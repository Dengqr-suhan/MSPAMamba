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

class Inception(nn.Module):
    """Inception模块 - 多尺度特征提取"""
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return branch1, branch2, branch3, branch4

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

class MSMambaLayer(nn.Module):
    """Multi-Scale Mamba Layer - 多尺度Mamba核心编码模块"""
    def __init__(self, input_channels, output_channels, d_state=16, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # 🔥 修复：正确配置Inception维度
        base_dim = max(input_channels // 4, 16)
        ch1x1 = base_dim
        ch3x3red = base_dim // 2
        ch3x3 = base_dim
        ch5x5red = base_dim // 4
        ch5x5 = base_dim
        pool_proj = base_dim
        
        self.inception = Inception(input_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj)
        
        # 🔥 修复：正确计算拼接后的维度
        concat_dim = ch1x1 + ch3x3 + ch5x5 + pool_proj
        
        self.norm1 = nn.LayerNorm(ch1x1)
        self.norm2 = nn.LayerNorm(ch3x3) 
        self.norm3 = nn.LayerNorm(ch5x5)
        self.norm4 = nn.LayerNorm(pool_proj)
        self.norm = nn.LayerNorm(concat_dim)
        
        if Mamba is None:
            raise ImportError("mamba_ssm is required for MSMambaLayer")
            
        self.mamba1 = Mamba(d_model=ch1x1, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba2 = Mamba(d_model=ch3x3, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba3 = Mamba(d_model=ch5x5, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba4 = Mamba(d_model=pool_proj, d_state=d_state, d_conv=d_conv, expand=expand)
        
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 🔥 修复：输出投影到指定维度
        self.output_proj = nn.Conv2d(concat_dim, output_channels, 1, bias=False)
        
        if input_channels != output_channels:
            self.identity_proj = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        else:
            self.identity_proj = nn.Identity()
            
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        x1, x2, x3, x4 = self.inception(x)
        
        x1 = x1.reshape(B, -1, H*W).transpose(-1, -2)
        x1 = self.norm1(x1)
        x1 = self.mamba1(x1) + self.skip_scale * x1
        
        x2 = x2.reshape(B, -1, H*W).transpose(-1, -2)
        x2 = self.norm2(x2)
        x2 = self.mamba2(x2) + self.skip_scale * x2
        
        x3 = x3.reshape(B, -1, H*W).transpose(-1, -2)
        x3 = self.norm3(x3)
        x3 = self.mamba3(x3) + self.skip_scale * x3
        
        x4 = x4.reshape(B, -1, H*W).transpose(-1, -2)
        x4 = self.norm4(x4)
        x4 = self.mamba4(x4) + self.skip_scale * x4
        
        x_mamba = torch.cat([x1, x2, x3, x4], dim=2)
        x_mamba = self.norm(x_mamba)
        
        out = x_mamba.transpose(-1, -2).reshape(B, -1, H, W)
        out = self.output_proj(out)
        
        identity_proj = self.identity_proj(identity)
        out = identity_proj + self.drop_path(out)
            
        return out

class MSEncoder(nn.Module):
    """Multi-Scale Encoder：多尺度编码器核心"""
    def __init__(self, input_channels=3, depths=[2, 2, 9, 2], scp_channels=[64, 128, 256, 512], 
                 resnet_channels=[64, 128, 256, 512], d_state=16, drop_rate=0., drop_path_rate=0.2):
        super().__init__()
        self.num_layers = len(depths)
        self.scp_channels = scp_channels
        self.resnet_channels = resnet_channels
        
        # 🔥 修复：初始卷积层输出第一层SCP通道
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
                block = MSMambaLayer(
                    input_channels=scp_channels[i_layer],
                    output_channels=scp_channels[i_layer],  # 🔥 保持通道不变
                    d_state=d_state,
                    d_conv=4,
                    expand=2,
                    drop_path=dpr[sum(depths[:i_layer]) + i_block]
                )
                layer_blocks.append(block)
            self.layers.append(layer_blocks)
        
        # 🔥 修复：下采样层正确配置通道变化
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

# 🔥 替换MPAM：用Mamba融合ResNet和SCP特征
class MPAMMambaFusion(nn.Module):
    """用Mamba替换MPAM的融合模块"""
    def __init__(self, resnet_channels, scp_channels, d_state=16, drop_path=0., **kwargs):
        super().__init__()
        self.resnet_channels = resnet_channels
        self.scp_channels = scp_channels
        
        # 通道对齐
        self.scp_proj = nn.Conv2d(scp_channels, resnet_channels, 1, bias=False)
        self.scp_norm = nn.BatchNorm2d(resnet_channels)
        self.resnet_norm = nn.BatchNorm2d(resnet_channels)
        
        # 用Mamba处理融合特征
        self.mamba = Mamba2DBlock(resnet_channels, d_state=d_state, drop_path=drop_path)

    def forward(self, resnet_feat, scp_feat):
        # 归一化
        resnet_norm = self.resnet_norm(resnet_feat)
        scp_proj = self.scp_norm(self.scp_proj(scp_feat))
        
        # 空间尺寸对齐
        if resnet_norm.shape[2:] != scp_proj.shape[2:]:
            scp_proj = F.interpolate(scp_proj, size=resnet_norm.shape[2:], 
                                   mode='bilinear', align_corners=False)
        
        # 简单相加融合
        fused = resnet_norm + scp_proj
        
        # Mamba处理
        return self.mamba(fused)

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
    """MSPAMamba主模型 - 消融实验版本1：替换MPAM和CAFMamba为Mamba"""
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
        
        # Multi-Scale编码器
        self.ms_encoder = MSEncoder(
            input_channels=3,
            depths=depths,
            scp_channels=self.scp_channels,
            resnet_channels=self.resnet_channels,
            d_state=d_state,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate
        )
        
        # 🔥 替换MPAM为MPAMMambaFusion
        self.mpam_fusion = nn.ModuleList()
        for i in range(len(self.resnet_channels)):
            mpam = MPAMMambaFusion(
                resnet_channels=self.resnet_channels[i],
                scp_channels=self.scp_channels[i],
                d_state=d_state,
                drop_path=drop_path_rate
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
        
        # Multi-Scale特征提取
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
            
            # MPAM融合（现在是MPAMMambaFusion）
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