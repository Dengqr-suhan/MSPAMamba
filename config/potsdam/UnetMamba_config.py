import torch
from torch.utils.data import DataLoader
import albumentations as albu
import numpy as np
import copy

from ours_env.losses import *
from ours_env.datasets.potsdam_dataset import *
from ours_env.datasets.transform import Compose, RandomScale, SmartCropV1
from ours_env.models.UNetMamba import UNetMamba
from tools.utils import Lookahead, process_model_params

# ====================== 1. 训练超参数 ======================

max_epoch = 50
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetmamba-U"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "unetmamba-U"
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

# ====================== 2. UNetMamba模型参数 ======================

# VSSM parameters
PATCH_SIZE = 4
IN_CHANS = 3
DEPTHS = [2, 2, 9, 2]
EMBED_DIM = 96
SSM_D_STATE = 16
SSM_RATIO = 2.0
SSM_RANK_RATIO = 2.0
SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"
SSM_CONV = 3
SSM_CONV_BIAS = True
SSM_DROP_RATE = 0.0
SSM_INIT = "v0"
SSM_FORWARDTYPE = "v4"
MLP_RATIO = 4.0
MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0
DROP_PATH_RATE = 0.1
PATCH_NORM = True
NORM_LAYER = "ln"
DOWNSAMPLE = "v2"
PATCHEMBED = "v2"
GMLP = False
USE_CHECKPOINT = False

# ====================== 3. 定义网络与损失 ======================

net = UNetMamba(pretrained=pretrained_ckpt_path,
                num_classes=num_classes,
                patch_size=PATCH_SIZE,
                in_chans=IN_CHANS,
                depths=DEPTHS,
                dims=EMBED_DIM,
                ssm_d_state=SSM_D_STATE,
                ssm_ratio=SSM_RATIO,
                ssm_rank_ratio=SSM_RANK_RATIO,
                ssm_dt_rank=("auto" if SSM_DT_RANK == "auto" else int(SSM_DT_RANK)),
                ssm_act_layer=SSM_ACT_LAYER,
                ssm_conv=SSM_CONV,
                ssm_conv_bias=SSM_CONV_BIAS,
                ssm_drop_rate=SSM_DROP_RATE,
                ssm_init=SSM_INIT,
                forward_type=SSM_FORWARDTYPE,
                mlp_ratio=MLP_RATIO,
                mlp_act_layer=MLP_ACT_LAYER,
                mlp_drop_rate=MLP_DROP_RATE,
                drop_path_rate=DROP_PATH_RATE,
                patch_norm=PATCH_NORM,
                norm_layer=NORM_LAYER,
                downsample_version=DOWNSAMPLE,
                patchembed_version=PATCHEMBED,
                gmlp=GMLP,
                use_checkpoint=USE_CHECKPOINT)

# define the loss
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True

# ====================== 4. 数据集与增广 ======================

def get_training_transform():
    """训练时的数据增广"""
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def train_aug(img, mask):
    """多尺度随机缩放 + 智能裁剪 + 归一化"""
    crop_aug = Compose([
        RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)
    ])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = albu.Compose([albu.Normalize()])(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

train_dataset = PotsdamDataset(data_root='data/potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(transform=val_aug)
test_dataset = PotsdamDataset(data_root='data/potsdam/test',
                              transform=val_aug)

# ====================== 5. 定义 Dataloader ======================

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# ====================== 6. 优化器和学习率调度 ======================

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)