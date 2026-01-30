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

class SS2D(nn.Module):
    """Mamba选择性扫描2D模块"""
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        if selective_scan_fn is None:
            raise ImportError("mamba_ssm is required for complete Mamba implementation")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device, dtype=torch.float32)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) 
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) 
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
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

class SCPMambaLayer(nn.Module):
    """SCP Mamba Layer - 核心编码模块"""
    def __init__(self, input_channels, output_channels, d_state=16, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        base_dim = max(input_channels // 4, 16)
        inception_dims = [base_dim, base_dim, base_dim, base_dim, base_dim, base_dim]
        
        self.inception = Inception(input_channels, inception_dims[0], inception_dims[1], 
                                 inception_dims[2], inception_dims[3], inception_dims[4], inception_dims[5])
        
        concat_dim = sum(inception_dims[::2])  # 只取输出维度
        
        self.norm1 = nn.LayerNorm(inception_dims[0])
        self.norm2 = nn.LayerNorm(inception_dims[2]) 
        self.norm3 = nn.LayerNorm(inception_dims[4])
        self.norm4 = nn.LayerNorm(inception_dims[5])
        self.norm = nn.LayerNorm(concat_dim)
        
        if Mamba is None:
            raise ImportError("mamba_ssm is required for SCPMambaLayer")
            
        self.mamba1 = Mamba(d_model=inception_dims[0], d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba2 = Mamba(d_model=inception_dims[2], d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba3 = Mamba(d_model=inception_dims[4], d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba4 = Mamba(d_model=inception_dims[5], d_state=d_state, d_conv=d_conv, expand=expand)
        
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
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

class SCPEncoder(nn.Module):
    """SCP编码器：双分支编码器核心"""
    def __init__(self, input_channels=3, depths=[2, 2, 9, 2], scp_channels=[96, 192, 384, 768], 
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
        
        # 构建SCP层
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer_blocks = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                block = SCPMambaLayer(
                    input_channels=scp_channels[i_layer],
                    output_channels=scp_channels[i_layer],
                    d_state=d_state,
                    d_conv=4,
                    expand=2,
                    drop_path=dpr[sum(depths[:i_layer]) + i_block]
                )
                layer_blocks.append(block)
            self.layers.append(layer_blocks)
        
        # 通道对齐层
        self.channel_align = nn.ModuleList()
        for i in range(self.num_layers):
            align_layer = ChannelAlignmentLayer(scp_channels[i], resnet_channels[i])
            self.channel_align.append(align_layer)
        
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
            
            aligned_feature = self.channel_align[i](x)
            scp_features.append(aligned_feature)
            
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
    """多尺度并行注意力融合模块 - 解决双分支通道对齐问题"""
    def __init__(self, resnet_channels, scp_channels, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.ReLU6, stage=0):
        super().__init__()
        self.resnet_channels = resnet_channels
        self.scp_channels = scp_channels
        self.stage = stage
        
        # SCP特征投影到ResNet维度
        self.scp_proj = nn.Conv2d(scp_channels, resnet_channels, 1, bias=False)
        self.scp_norm = nn.BatchNorm2d(resnet_channels)
        
        # ResNet特征归一化
        self.resnet_norm = nn.BatchNorm2d(resnet_channels)
        
        # 根据stage选择注意力机制
        attention_modules = [PFA, LNA, SRA, GCA]
        self.attention = attention_modules[min(stage, 3)](resnet_channels, act_layer)
        
        # MLP层
        mlp_hidden_dim = int(resnet_channels * mlp_ratio)
        self.mlp = Mlp(in_features=resnet_channels, hidden_features=mlp_hidden_dim, 
                      out_features=resnet_channels, act_layer=act_layer, drop=drop)
        self.norm1 = nn.BatchNorm2d(resnet_channels)
        self.norm2 = nn.BatchNorm2d(resnet_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 特征融合权重
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
        # 归一化输入特征
        resnet_norm = self.resnet_norm(resnet_feat)
        
        # SCP特征投影并归一化
        scp_proj = self.scp_proj(scp_feat)
        scp_norm = self.scp_norm(scp_proj)
        
        # 空间尺寸对齐
        if resnet_norm.shape[2:] != scp_norm.shape[2:]:
            scp_norm = F.interpolate(scp_norm, size=resnet_norm.shape[2:], 
                                   mode='bilinear', align_corners=False)
        
        # 特征融合
        weights = F.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * resnet_norm + weights[1] * scp_norm
        
        # 通道和空间注意力
        fused = fused * self.se(fused)
        fused = fused * self.spatial_attn(fused)
        
        # 应用注意力机制
        attended = self.attention(fused)
        
        # 残差连接和MLP
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

class CMTF_MutilScal(nn.Module):
    """多尺度特征提取模块"""
    def __init__(self, channels=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(CMTF_MutilScal, self).__init__()
        self.conv0_1 = nn.Conv2d(channels, channels//fc_ratio, 1)
        self.bn0_1 = nn.BatchNorm2d(channels//fc_ratio)
        self.conv0_2 = nn.Conv2d(channels//fc_ratio, channels//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=channels //fc_ratio)
        self.bn0_2 = nn.BatchNorm2d(channels // fc_ratio)
        self.conv0_3 = nn.Conv2d(channels//fc_ratio, channels, 1)
        self.bn0_3 = nn.BatchNorm2d(channels)

        self.conv1_2 = nn.Conv2d(channels//fc_ratio, channels//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=channels // fc_ratio)
        self.bn1_2 = nn.BatchNorm2d(channels//fc_ratio)
        self.conv1_3 = nn.Conv2d(channels//fc_ratio, channels, 1)
        self.bn1_3 = nn.BatchNorm2d(channels)

        self.conv2_2 = nn.Conv2d(channels//fc_ratio, channels//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=channels//fc_ratio)
        self.bn2_2 = nn.BatchNorm2d(channels//fc_ratio)
        self.conv2_3 = nn.Conv2d(channels//fc_ratio, channels, 1)
        self.bn2_3 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU6()
        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()
        attn0_1 = self.relu(self.bn0_1(self.conv0_1(x)))
        attn0_2 = self.relu(self.bn0_2(self.conv0_2(attn0_1)))
        attn0_3 = self.relu(self.bn0_3(self.conv0_3(attn0_2)))
        attn1_2 = self.relu(self.bn1_2(self.conv1_2(attn0_1)))
        attn1_3 = self.relu(self.bn1_3(self.conv1_3(attn1_2)))
        attn2_2 = self.relu(self.bn2_2(self.conv2_2(attn0_1)))
        attn2_3 = self.relu(self.bn2_3(self.conv2_3(attn2_2)))
        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.relu(self.bn3(self.conv3(attn)))
        attn = attn * u
        pool = self.Avg(attn)
        return pool

class CrossAttention_MHSA(nn.Module):
    """Cross Attention多头自注意力"""
    def __init__(self, channels, num_heads, atten_drop=0., proj_drop=0., dilation=[3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(CrossAttention_MHSA, self).__init__()
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."
        self.channels = channels
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = CMTF_MutilScal(channels=channels, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=channels//fc_ratio, out_channels=channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(channels, 2 * channels, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)
        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn

class CrossAttentionMambaBlock(nn.Module):
    """融合Cross Attention和Mamba的混合块"""
    def __init__(self, channels=512, num_heads=16, mlp_ratio=4, pool_ratio=16, drop=0., 
                 dilation=[3, 5, 7], drop_path=0., act_layer=nn.ReLU6, 
                 norm_layer=nn.BatchNorm2d, d_state=16, d_conv=3, expand=2, fc_ratio=4):
        super().__init__()
        
        self.norm1 = norm_layer(channels)
        self.attn = CrossAttention_MHSA(channels, num_heads=num_heads, atten_drop=drop, 
                                       proj_drop=drop, dilation=dilation,
                                       pool_ratio=pool_ratio, fc_ratio=fc_ratio)
        
        self.norm2 = nn.LayerNorm(channels)
        self.mamba = SS2D(d_model=channels, d_state=d_state, d_conv=d_conv, 
                         expand=expand, dropout=drop)
        
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.fusion_norm = norm_layer(channels)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(channels // mlp_ratio)
        self.mlp = CMTF_E_FFN(in_features=channels, hidden_features=mlp_hidden_dim, 
                             out_features=channels, act_layer=act_layer, drop=drop)
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        attn_out = self.attn(self.norm1(x))
        
        x_mamba = x.permute(0, 2, 3, 1)
        x_mamba = self.norm2(x_mamba)
        mamba_out = self.mamba(x_mamba)
        mamba_out = mamba_out.permute(0, 3, 1, 2)
        
        fused = torch.cat([attn_out, mamba_out], dim=1)
        fused = self.fusion_norm(self.fusion_conv(fused))
        
        x = x + self.drop_path(self.alpha * attn_out + (1 - self.alpha) * fused)
        x = x + self.drop_path(self.mlp(x))
        
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
    """Cross Attention融合解码器"""
    def __init__(self, encoder_channels=(64, 128, 256, 512), decode_channels=512,
                 dilation=[[1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]],
                 fc_ratio=4, dropout=0.1, num_classes=6, use_mamba=True, d_state=16):
        super().__init__()
        
        self.Conv1 = nn.Conv2d(encoder_channels[3], decode_channels, 1)
        self.Conv2 = nn.Conv2d(encoder_channels[2], decode_channels, 1)
        self.Conv3 = nn.Conv2d(encoder_channels[1], decode_channels, 1)
        self.Conv4 = nn.Conv2d(encoder_channels[0], decode_channels, 1)
        
        if use_mamba:
            self.b4 = CrossAttentionMambaBlock(channels=decode_channels, dilation=dilation[3], 
                                             fc_ratio=fc_ratio, drop=dropout, d_state=d_state)
            self.b3 = CrossAttentionMambaBlock(channels=decode_channels, dilation=dilation[2], 
                                             fc_ratio=fc_ratio, drop=dropout, d_state=d_state)
            self.b2 = CrossAttentionMambaBlock(channels=decode_channels, dilation=dilation[1], 
                                             fc_ratio=fc_ratio, drop=dropout, d_state=d_state)
        
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
    """MSPAMamba主模型 - 双分支编码器架构"""
    def __init__(self, decode_channels=96, dropout=0.1, backbone_name='swsl_resnet18', 
                 pretrained=True, num_classes=6, embed_dim=96, depths=[2, 2, 9, 2], 
                 drop_path_rate=0.1, d_state=16):
        super().__init__()
        
        # ResNet编码器
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        all_backbone_channels = self.backbone.feature_info.channels()
        
        if len(all_backbone_channels) >= 4:
            self.resnet_channels = all_backbone_channels[-4:]
        else:
            self.resnet_channels = all_backbone_channels + [all_backbone_channels[-1]] * (4 - len(all_backbone_channels))
        
        # 根据embed_dim生成SCP通道
        if embed_dim != self.resnet_channels[0]:
            scale_factor = embed_dim / self.resnet_channels[0]
            self.scp_channels = [int(ch * scale_factor) for ch in self.resnet_channels]
        else:
            self.scp_channels = self.resnet_channels
        
        # SCP编码器
        self.scp_encoder = SCPEncoder(
            input_channels=3,
            depths=depths,
            scp_channels=self.scp_channels,
            resnet_channels=self.resnet_channels,
            d_state=d_state,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate
        )
        
        # MPAM融合模块
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
        
        # 解码器
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
        
        # 存储配置参数
        self.embed_dim = embed_dim
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.d_state = d_state
    
    def forward(self, x):
        # ResNet特征提取
        resnet_features = self.backbone(x)
        
        # SCP特征提取（已对齐到ResNet通道）
        scp_features = self.scp_encoder(x)
        
        # 双分支特征融合
        fused_features = []
        for i, (resnet_feat, scp_feat, mpam) in enumerate(
            zip(resnet_features, scp_features, self.mpam_fusion)
        ):
            # 空间尺寸对齐
            if resnet_feat.shape[2:] != scp_feat.shape[2:]:
                scp_feat = F.interpolate(scp_feat, size=resnet_feat.shape[2:], 
                                       mode='bilinear', align_corners=False)
            
            # MPAM融合
            fused_feat = mpam(resnet_feat, scp_feat)
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