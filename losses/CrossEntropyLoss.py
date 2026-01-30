
import torch
import torch.nn.functional as F
from torch import nn

class CrossEntropyLoss(nn.Module):
    """
    通用的语义分割 CrossEntropy 损失，支持是否带辅助输出 (aux).
    若 use_aux=True 并且传入 preds 为 (main_pred, aux_pred) 形式，
    则会把主分支损失和辅助分支损失一起计算，并按照 aux_weight 进行加和。
    """
    def __init__(self,
                 ignore_index=255,
                 weight=None,
                 reduction='mean',
                 use_aux=False,
                 aux_weight=0.4):
        """
        :param ignore_index: 忽略的类别索引，如 255
        :param weight: 类别权重，默认为 None
        :param reduction: 损失聚合方式，默认为 'mean'
        :param use_aux: 是否启用辅助分支损失计算
        :param aux_weight: 辅助分支损失的权重系数
        """
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.use_aux = use_aux
        self.aux_weight = aux_weight

    def forward(self, preds, target):
        """
        :param preds: 若 use_aux=True，则通常 preds = (main_pred, aux_pred)
                      否则为单个 tensor [N, C, H, W]
        :param target: [N, H, W] 的整型语义分割标签
        """
        # 如果启用辅助分支，并且 preds 是一个 tuple/list，说明存在 (main_pred, aux_pred)
        if self.use_aux and isinstance(preds, (tuple, list)):
            main_pred, aux_pred = preds[0], preds[1]
            loss_main = F.cross_entropy(main_pred, target, ignore_index=self.ignore_index,
                                        weight=self.weight, reduction=self.reduction)
            loss_aux = F.cross_entropy(aux_pred, target, ignore_index=self.ignore_index,
                                       weight=self.weight, reduction=self.reduction)
            return loss_main + self.aux_weight * loss_aux
        else:
            # 单输出（无辅助分支）
            return F.cross_entropy(preds, target, ignore_index=self.ignore_index,
                                   weight=self.weight, reduction=self.reduction)
