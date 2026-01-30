import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import math
import re
from functools import partial
from torch.utils import checkpoint

from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

def repeat(tensor, pattern, **axes_lengths):
    """重复张量的辅助函数"""
    return tensor.repeat(*[axes_lengths.get(axis, 1) for axis in pattern.split()])

def build_norm_layer(norm_cfg, num_features, postfix=''):
    """构建归一化层"""
    if norm_cfg['type'] == 'BN':
        layer = nn.BatchNorm2d(num_features)
    elif norm_cfg['type'] == 'IN':
        layer = nn.InstanceNorm2d(num_features)
    else:
        raise NotImplementedError(f"Norm type {norm_cfg['type']} not implemented")
    return f'norm{postfix}', layer

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class BasicConv2d(nn.Module):
    """基础卷积模块"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAlignmentLayer(nn.Module):
    """通道对齐层 - 解决ResNet和SCP通道不匹配问题"""
    def __init__(self, scp_channels, resnet_channels):
        super().__init__()
        self.scp_channels = scp_channels
        self.resnet_channels = resnet_channels
        
        if scp_channels != resnet_channels:
            self.align_conv = nn.Sequential(
                nn.Conv2d(scp_channels, resnet_channels, 1, bias=False),
                nn.BatchNorm2d(resnet_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.align_conv = nn.Identity()
    
    def forward(self, x):
        return self.align_conv(x)

# 🔥 新增：通用2D Mamba Block
class Mamba2DBlock(nn.Module):
    """通用2D Mamba Block - 用于替换复杂模块"""
    def __init__(self, channels, d_state=16, d_conv=3, expand=2, drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # 转换为序列格式
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_seq = self.norm(x_seq)
        # Mamba处理
        y_seq = self.mamba(x_seq)
        # 转换回2D格式
        y = y_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # 残差连接
        return x + self.drop_path(y)

# 🔥 替换MSMambaLayer：用简单Mamba替换多尺度Mamba
class SimpleMambaLayer(nn.Module):
    """简化的Mamba Layer - 替换MSMambaLayer"""
    def __init__(self, input_channels, output_channels, d_state=16, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # 简单的通道对齐
        if input_channels != output_channels:
            self.proj = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        else:
            self.proj = nn.Identity()
            
        # 用单个Mamba替换复杂的多尺度结构
        self.mamba = Mamba2DBlock(output_channels, d_state=d_state, d_conv=d_conv, drop_path=drop_path)
            
    def forward(self, x):
        # 通道投影
        x = self.proj(x)
        # Mamba处理
        return self.mamba(x)

# 🔥 替换MSEncoder：用简化编码器替换多尺度编码器
class SimpleEncoder(nn.Module):
    """简化编码器：用普通Mamba替换MSEncoder"""
    def __init__(self, input_channels=3, depths=[2, 2, 9, 2], scp_channels=[64, 128, 256, 512], 
                 resnet_channels=[64, 128, 256, 512], d_state=16, drop_rate=0., drop_path_rate=0.2):
        super().__init__()
        self.num_layers = len(depths)
        self.scp_channels = scp_channels
        self.resnet_channels = resnet_channels
        
        # 初始卷积层
        self.stem_conv = nn.Sequential(
            nn.Conv2d(input_channels, scp_channels[0], 4, stride=4, bias=False),
            nn.BatchNorm2d(scp_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer_blocks = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                # 🔥 用SimpleMambaLayer替换MSMambaLayer
                block = SimpleMambaLayer(
                    input_channels=scp_channels[i_layer],
                    output_channels=scp_channels[i_layer],
                    d_state=d_state,
                    d_conv=4,
                    expand=2,
                    drop_path=dpr[sum(depths[:i_layer]) + i_block]
                )
                layer_blocks.append(block)
            self.layers.append(layer_blocks)
        
        # 下采样层
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            downsample = nn.Sequential(
                nn.Conv2d(scp_channels[i_layer], scp_channels[i_layer+1], 2, stride=2, bias=False),
                nn.BatchNorm2d(scp_channels[i_layer+1]),
                nn.ReLU(inplace=True)
            )
            self.downsamples.append(downsample)

    def forward(self, x):
        scp_features = []
        
        x = self.stem_conv(x)
        
        for i in range(self.num_layers):
            for block in self.layers[i]:
                x = block(x)
            
            scp_features.append(x)
            
            if i < self.num_layers - 1:
                x = self.downsamples[i](x)
        
        return scp_features

class PFA(nn.Module):
    """Point-wise Feature Attention"""
    def __init__(self, channels, act_layer=nn.ReLU6):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels, channels*4, 1, bias=False),
            nn.BatchNorm2d(channels*4),
            act_layer(),
            nn.Conv2d(channels*4, channels, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            act_layer(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.p_conv(x)
        att = att * self.se(x)
        x = x * self.gate_fn(att)
        return x

class LNA(nn.Module):
    """Local Neighborhood Attention"""
    def __init__(self, channels, act_layer=nn.ReLU6):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            act_layer(),
            nn.Conv2d(channels, channels, 1)
        )
        self.gate_fn = nn.Sigmoid()
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels),
            nn.BatchNorm2d(channels),
            act_layer(),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        att = self.local_conv(x) + self.dilated_conv(x)
        x = x * self.gate_fn(att)
        return x

class SRA(nn.Module):
    """Spatial Range Attention"""
    def __init__(self, channels, att_kernel=11):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, att_kernel, padding=att_kernel // 2, groups=channels),
            nn.BatchNorm2d(channels)
        )
        self.spatial_conv7 = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        att = self.spatial_conv(x) + self.spatial_conv7(x)
        x = x * self.act(att)
        return x

class GCA(nn.Module):
    """Global Context Attention"""
    def __init__(self, channels, act_layer=nn.ReLU6):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_pool_max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            act_layer(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.global_pool(x) + self.global_pool_max(x)
        att = self.fc(att)
        x = x * att
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class MPAM(nn.Module):
    """多尺度并行注意力融合模块"""
    def __init__(self, resnet_channels, scp_channels, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.ReLU6, stage=0):
        super().__init__()
        self.resnet_channels = resnet_channels
        self.scp_channels = scp_channels
        self.stage = stage
        
        # 🔥 修复：SCP特征投影到ResNet维度（使用正确的输入输出通道）
        self.scp_proj = nn.Conv2d(scp_channels, resnet_channels, 1, bias=False)
        self.scp_norm = nn.BatchNorm2d(resnet_channels)
        
        self.resnet_norm = nn.BatchNorm2d(resnet_channels)
        
        # 🔥 修复：根据stage选择注意力机制
        if stage == 0:
            self.attention = PFA(resnet_channels, act_layer)
        elif stage == 1:
            self.attention = LNA(resnet_channels, act_layer)
        elif stage == 2:
            self.attention = SRA(resnet_channels, att_kernel=11)
        elif stage == 3:
            self.attention = GCA(resnet_channels, act_layer)
        else:
            self.attention = PFA(resnet_channels, act_layer)
        
        mlp_hidden_dim = int(resnet_channels * mlp_ratio)
        self.mlp = Mlp(in_features=resnet_channels, hidden_features=mlp_hidden_dim, 
                      out_features=resnet_channels, act_layer=act_layer, drop=drop)
        self.norm1 = nn.BatchNorm2d(resnet_channels)
        self.norm2 = nn.BatchNorm2d(resnet_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.fusion_weight = nn.Parameter(torch.ones(2) * 0.5)
        reduction = max(resnet_channels // 4, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(resnet_channels, reduction, 1, bias=True),
            act_layer(),
            nn.Conv2d(reduction, resnet_channels, 1, bias=True),
            nn.Sigmoid()
        )
        self.spatial_attn = SpatialAttention()

    def forward(self, resnet_feat, scp_feat):
        resnet_norm = self.resnet_norm(resnet_feat)
        
        # 🔥 修复：确保输入输出维度正确
        scp_proj = self.scp_proj(scp_feat)
        scp_norm = self.scp_norm(scp_proj)
        
        if resnet_norm.shape[2:] != scp_norm.shape[2:]:
            scp_norm = F.interpolate(scp_norm, size=resnet_norm.shape[2:], 
                                   mode='bilinear', align_corners=False)
        
        weights = F.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * resnet_norm + weights[1] * scp_norm
        
        fused = fused * self.se(fused)
        fused = fused * self.spatial_attn(fused)
        
        attended = self.attention(fused)
        
        out = resnet_norm + self.drop_path(self.norm1(attended))
        mlp_out = self.mlp(out)
        out = out + self.drop_path(self.norm2(mlp_out))
        
        return out

class CMTF_E_FFN(nn.Module):
    """增强型前馈网络"""
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.):
        super(CMTF_E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=ksize, groups=hidden_features)
        self.conv2 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, groups=hidden_features)
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.fc2(x1 + x2)
        x = self.act(x)
        return x

class CrossAttentionFusion(nn.Module):
    """Cross Attention特征融合模块"""
    def __init__(self, channels, eps=1e-8):
        super(CrossAttentionFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(channels, channels, 5)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class CrossAttentionFusionDecoder(nn.Module):
    """Cross Attention融合解码器 - 用Mamba替换CrossAttentionMambaBlock"""
    def __init__(self, encoder_channels=(64, 128, 256, 512), decode_channels=512,
                 dilation=[[1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]],
                 fc_ratio=4, dropout=0.1, num_classes=6, use_mamba=True, d_state=16):
        super().__init__()
        
        self.Conv1 = nn.Conv2d(encoder_channels[3], decode_channels, 1)
        self.Conv2 = nn.Conv2d(encoder_channels[2], decode_channels, 1)
        self.Conv3 = nn.Conv2d(encoder_channels[1], decode_channels, 1)
        self.Conv4 = nn.Conv2d(encoder_channels[0], decode_channels, 1)
        
        # 🔥 替换CrossAttentionMambaBlock为简单的Mamba2DBlock
        if use_mamba:
            self.b4 = Mamba2DBlock(channels=decode_channels, d_state=d_state)
            self.b3 = Mamba2DBlock(channels=decode_channels, d_state=d_state)
            self.b2 = Mamba2DBlock(channels=decode_channels, d_state=d_state)
        
        self.p3 = CrossAttentionFusion(decode_channels)
        self.p2 = CrossAttentionFusion(decode_channels)
        self.p1 = CrossAttentionFusion(decode_channels)
        
        self.final_conv = nn.Conv2d(decode_channels, encoder_channels[0], 3, padding=1)
        self.seg_head = nn.Conv2d(encoder_channels[0], num_classes, 1)

        self.init_weight()

    def forward(self, features, target_size=None):
        res1, res2, res3, res4 = features
        
        if target_size is None:
            target_size = (res1.shape[2], res1.shape[3])
        h, w = target_size

        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)
        res2 = self.Conv3(res2)
        res1 = self.Conv4(res1)

        x = self.b4(res4)
        x = self.p3(x, res3)
        x = self.b3(x)
        x = self.p2(x, res2)
        x = self.b2(x)
        x = self.p1(x, res1)
        x = self.final_conv(x)
        x = self.seg_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class MSPAMamba(nn.Module):
    """MSPAMamba主模型 - 消融实验版本2：替换MSMamba和CAFMamba为Mamba"""
    def __init__(self, decode_channels=96, dropout=0.1, backbone_name='swsl_resnet18', 
                 pretrained=True, num_classes=6, embed_dim=96, depths=[2, 2, 9, 2], 
                 drop_path_rate=0.1, d_state=16):
        super().__init__()
        
        # ResNet编码器
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        all_backbone_channels = self.backbone.feature_info.channels()
        
        # 🔥 修复：正确获取ResNet的后4层通道
        if len(all_backbone_channels) >= 4:
            self.resnet_channels = all_backbone_channels[-4:]  # 取后4层：[64, 128, 256, 512]
        else:
            self.resnet_channels = all_backbone_channels + [all_backbone_channels[-1]] * (4 - len(all_backbone_channels))
        
        print(f"🔍 ResNet后4层通道: {self.resnet_channels}")
        
        # SCP通道与ResNet后4层对齐
        self.scp_channels = self.resnet_channels.copy()
        
        # 🔥 替换MSEncoder为SimpleEncoder
        self.ms_encoder = SimpleEncoder(
            input_channels=3,
            depths=depths,
            scp_channels=self.scp_channels,
            resnet_channels=self.resnet_channels,
            d_state=d_state,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate
        )
        
        # MPAM融合模块（保持不变）
        self.mpam_fusion = nn.ModuleList()
        for i in range(len(self.resnet_channels)):
            mpam = MPAM(
                resnet_channels=self.resnet_channels[i],
                scp_channels=self.scp_channels[i],
                mlp_ratio=4.,
                drop=dropout,
                drop_path=drop_path_rate,
                stage=i
            )
            self.mpam_fusion.append(mpam)
        
        # 解码器（已替换CrossAttentionMambaBlock为Mamba2DBlock）
        self.decoder = CrossAttentionFusionDecoder(
            encoder_channels=self.resnet_channels,
            decode_channels=decode_channels,
            dilation=[[1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]],
            fc_ratio=4,
            dropout=dropout,
            num_classes=num_classes,
            use_mamba=True,
            d_state=d_state
        )
        
        self.embed_dim = embed_dim
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.d_state = d_state
    
    def forward(self, x):
        # ResNet特征提取
        all_resnet_features = self.backbone(x)
        
        # 🔥 修复：只使用ResNet的后4层特征，与SCP对齐
        resnet_features = all_resnet_features[-4:]  # 取后4层
        
        # 简化编码器特征提取（替换了MSEncoder）
        ms_features = self.ms_encoder(x)
        
        # 🔥 修复：现在ResNet和Multi-Scale特征完全对齐
        fused_features = []
        for i, (resnet_feat, ms_feat, mpam) in enumerate(
            zip(resnet_features, ms_features, self.mpam_fusion)
        ):
            # 空间尺寸对齐
            if resnet_feat.shape[2:] != ms_feat.shape[2:]:
                ms_feat = F.interpolate(ms_feat, size=resnet_feat.shape[2:], 
                                       mode='bilinear', align_corners=False)
            
            # MPAM融合
            fused_feat = mpam(resnet_feat, ms_feat)
            fused_features.append(fused_feat)
        
        # 解码器处理
        output = self.decoder(fused_features, target_size=x.shape[2:])
        
        return output

def load_pretrained_ckpt(model, ckpt_path="./pretrain/vmamba_tiny_e292.pth"):
    """加载预训练权重"""
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias", 
                   "patch_embed.proj.weight", "patch_embed.proj.bias", 
                   "patch_embed.norm.weight", "patch_embed.norm.weight"]    

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_dict = model.state_dict()
    
    loaded_count = 0
    
    for k, v in ckpt['model'].items():
        if k in skip_params:
            continue
        
        kr1 = f"scp_encoder.{k}"
        kr2 = f"backbone.{k}"
        kr3 = k
        
        for kr in [kr1, kr2, kr3]:
            if kr in model_dict.keys():
                if v.shape == model_dict[kr].shape:
                    model_dict[kr] = v
                    loaded_count += 1
                    break
        
    model.load_state_dict(model_dict)
    return model