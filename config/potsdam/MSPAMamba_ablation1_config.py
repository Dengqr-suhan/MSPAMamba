import albumentations as albu
import numpy as np
import torch
from torch.utils.data import DataLoader

from ours_env.losses import *
from ours_env.datasets.potsdam_dataset import *
from ours_env.datasets.transform import Compose, RandomScale, SmartCropV1
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

weights_name = "mspamamba-ablation1-U"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "mspamamba-ablation1-U"
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = 'auto'
resume_ckpt_path = None

# ====================== 2. 模型配置 ======================
# 消融实验1：移除MPAM和CAFMamba，保留MSMamba
from ours_env.models.MSPAMamba_ablation1 import MSPAMamba, load_pretrained_ckpt

net = MSPAMamba(
    decode_channels=256,  
    dropout=0.1,
    backbone_name='swsl_resnet18',
    pretrained=True,
    depths=[2, 2, 9, 2], 
    embed_dim=96,
    num_classes=6,
    drop_path_rate=0.1,
    d_state=16
)

# ====================== 3. 损失函数 ======================
use_aux_loss = False  
loss = UnetMambaLoss(ignore_index=ignore_index) 

# ====================== 4. 预训练权重加载 ======================
import sys
if 'train_supervision.py' in sys.argv[0] or 'train' in sys.argv:
    try:
        net = load_pretrained_ckpt(net, ckpt_path="pretrain/vmamba_tiny_e292.pth")
        print("Successfully loaded pretrained VMamba weights for Potsdam ablation1")
    except Exception as e:
        print(f"Warning: Could not load pretrained VMamba weights: {e}")
        print("Training Potsdam ablation1 from scratch.")
else:
    print("Testing mode: Skipping pretrained weight loading")

# ====================== 5. 数据增强 ======================
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

# ====================== 6. 数据集 ======================
train_dataset = PotsdamDataset(data_root='data/potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(transform=val_aug)
test_dataset = PotsdamDataset(data_root='data/potsdam/test',
                              transform=val_aug)

# ====================== 7. 数据加载器 ======================
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

# ====================== 8. 优化器和学习率调度器 ======================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)