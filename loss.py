import torch
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, num_boxes=1.0, alpha=0.25, gamma=2.0):
    """
    inputs: logits [H*W]
    targets: targets [H*W]
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")  # 原始logits
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean() / num_boxes


def dice_loss(inputs, targets, num_boxes=1.0, smooth=1e-6):
    """
    inputs: logits [H*W]
    targets: [H*W]
    """
    inputs = torch.sigmoid(inputs).float()
    targets = targets.float()

    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return (1.0 - dice) / num_boxes


def boundary_loss(pred, target, smooth=1e-6):
    """
    pred:  [B, 1, H, W] logits
    target: [B, 1, H, W] 真值 (0~1)
    """
    pred = pred.float()
    target = target.float()

    grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    grad_x = F.pad(grad_pred_x, (0, 1, 0, 0), mode='replicate')
    grad_y = F.pad(grad_pred_y, (0, 0, 0, 1), mode='replicate')

    edge_weight = (grad_x.abs() + grad_y.abs()).clamp(min=smooth)
    edge_weight = edge_weight / (edge_weight.mean() + smooth)

    loss_unweighted = F.smooth_l1_loss(pred, target, reduction='none')
    loss_weighted = (loss_unweighted * edge_weight).mean()

    return loss_weighted
