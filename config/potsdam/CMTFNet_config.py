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

weights_name = "cmtfnet-U"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "last"
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = 'auto'
resume_ckpt_path = None

# ====================== 2. 定义网络与损失 ======================

from ours_env.models.CMTFNet import CMTFNet
from ours_env.losses.CrossEntropyLoss import CrossEntropyLoss

net = CMTFNet(num_classes=num_classes)
use_aux_loss = False
loss = CrossEntropyLoss(ignore_index=ignore_index, use_aux=use_aux_loss)

# ====================== 3. 数据集与增广 ======================

def get_training_transform():
    """简单版本的增广，和 unetformer.py 写法一致"""
    train_transform = [
        albu.HorizontalFlip(p=0.5),
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

# ====================== 调试信息：打印数据集详情 ======================

def print_dataset_info():
    """打印数据集的详细信息"""
    print("\n" + "="*60)
    print("数据集信息统计")
    print("="*60)
    
    # 训练集信息
    print(f"\n【训练集信息】")
    print(f"数据根目录: {train_dataset.data_root}")
    print(f"图像目录: {train_dataset.img_dir}")
    print(f"掩码目录: {train_dataset.mask_dir}")
    print(f"训练集图片总数: {len(train_dataset)}")
    
    # 分析训练集图片来源
    train_sources = {}
    for img_id in train_dataset.img_ids:
        # 提取大图名称（去掉坐标后缀）
        parts = img_id.split('_')
        if len(parts) >= 4:
            big_img_name = '_'.join(parts[:4])  # 例如: top_potsdam_2_10
            if big_img_name not in train_sources:
                train_sources[big_img_name] = 0
            train_sources[big_img_name] += 1
    
    print(f"\n训练集图片来源分布:")
    for big_img, count in sorted(train_sources.items()):
        print(f"  {big_img}: {count} 个小图块")
    print(f"训练集来自 {len(train_sources)} 张大图")
    
    # 测试集信息
    print(f"\n【测试集信息】")
    print(f"数据根目录: {test_dataset.data_root}")
    print(f"图像目录: {test_dataset.img_dir}")
    print(f"掩码目录: {test_dataset.mask_dir}")
    print(f"测试集图片总数: {len(test_dataset)}")
    
    # 分析测试集图片来源
    test_sources = {}
    for img_id in test_dataset.img_ids:
        # 提取大图名称（去掉坐标后缀）
        parts = img_id.split('_')
        if len(parts) >= 4:
            big_img_name = '_'.join(parts[:4])  # 例如: top_potsdam_2_13
            if big_img_name not in test_sources:
                test_sources[big_img_name] = 0
            test_sources[big_img_name] += 1
    
    print(f"\n测试集图片来源分布:")
    for big_img, count in sorted(test_sources.items()):
        print(f"  {big_img}: {count} 个小图块")
    print(f"测试集来自 {len(test_sources)} 张大图")
    
    # 验证集信息（如果与测试集不同）
    if val_dataset.data_root != test_dataset.data_root:
        print(f"\n【验证集信息】")
        print(f"数据根目录: {val_dataset.data_root}")
        print(f"验证集图片总数: {len(val_dataset)}")
    else:
        print(f"\n【验证集】使用与测试集相同的数据")
    
    print("\n" + "="*60)
    print("数据集信息统计完成")
    print("="*60 + "\n")

# 调用调试函数
print_dataset_info()

# ====================== 4. 定义 Dataloader ======================

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

# ====================== 5. 优化器和学习率调度 ======================

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)