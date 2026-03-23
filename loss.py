import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    '''
    pred: tensor [B,1,H,W] logits
    target: tensor [B,1,H,W]  只有01
    '''
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice


def boundary_loss(pred, target, smooth=1e-6):
    """
    pred:  [B, 1, H, W] logits
    target: [B, 1, H, W] 真值 (0~1)

    用 F.smooth_l1_loss 替代硬 Sobel 边缘检测，
    让边界区域的误差权重更高，但不引入非连续操作。
    """
    pred = pred.float()
    target = target.float()

    # 软边缘权重：在预测值变化剧烈的区域（通过梯度检测）给予更高权重
    # 用 pred 自身计算梯度权重，而非 target
    grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # 水平梯度
    grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # 垂直梯度

    grad_x = F.pad(grad_pred_x, (0, 1, 0, 0), mode='replicate')
    grad_y = F.pad(grad_pred_y, (0, 0, 0, 1), mode='replicate')

    edge_weight = (grad_x.abs() + grad_y.abs()).clamp(min=smooth)
    edge_weight = edge_weight / (edge_weight.mean() + smooth)  # 归一化到平均值为 1
    loss_unweighted = F.smooth_l1_loss(pred, target, reduction='none')

    loss_weighted = (loss_unweighted * edge_weight).mean()

    return loss_weighted
