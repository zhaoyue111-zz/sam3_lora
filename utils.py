import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch.utils.data import DataLoader
from transformers import Sam3Model, Sam3Processor
from dataset import SourceDataset, split_train_val
from torch.nn import DataParallel
import torch.nn.functional as F
from loss import sigmoid_focal_loss, dice_loss


def load_model():
    path = "/home/data4/zy/weight/sam3"
    base_model = Sam3Model.from_pretrained(path)
    processor = Sam3Processor.from_pretrained(path)

    lora_config = LoraConfig(
        r=14,
        lora_alpha=28,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    lora_model = get_peft_model(base_model, lora_config)

    lora_model = lora_model.to('cuda:0')
    lora_model = DataParallel(lora_model, device_ids=[0, 1, 2, 3])

    for name, param in lora_model.module.named_parameters():
        if 'geometry_encoder' in name:
            param.requires_grad = False
        if 'vision_encoder.backbone.layers' in name:
            layer_num = int(name.split('layers.')[1].split('.')[0])
            if layer_num < 16:
                param.requires_grad = False

    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return processor, lora_model


def load_data(processor):
    df = pd.read_excel("/home/data4/zy/data/CT_MRI_DATA/MRI_mapping_info_labeled.csv")
    image_root = "/home/data4/zy/data/CT_MRI_DATA/images/Delay/pngs"
    mask_root = "/home/data4/zy/data/CT_MRI_DATA/labels/Delay/pngs"
    train_uids, val_uids = split_train_val(df, train_ratio=0.8, subset=0)

    train_dataset = SourceDataset(df, image_root, mask_root, processor, subset=0, uids=train_uids)
    val_dataset = SourceDataset(df, image_root, mask_root, processor, subset=0, uids=val_uids)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def save_checkpoint(lora_model, optimizer, epoch, save_path):
    model_to_save = lora_model.module if isinstance(lora_model, DataParallel) else lora_model

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"检查点已保存: {save_path}")


def process_outputs(outputs, target, threshold=0.3,weight_dice=0.5,mode="train"):
    '''
    outputs
    target: [B,1,H,W]
    '''
    pred_logits = outputs.pred_logits  # [B, N]
    pred_masks = outputs.pred_masks  # [B, N, H', W']

    batch_losses_focal = []
    batch_losses_dice = []
    batch_preds = []

    B, C, H, W = target.shape

    for b in range(B):
        pred_logits_b = pred_logits[b]  # [N]
        pred_masks_b = pred_masks[b]  # [N, H', W']
        pred_probs = torch.sigmoid(pred_logits_b)  # [N]

        weight = pred_probs.clamp(min=threshold, max=1.0)
        weight = weight / (weight.sum() + 1e-8)
        pred_mask = (pred_masks_b * weight.view(-1, 1, 1)).sum(dim=0)  # [H', W']

        if pred_mask.shape[-2:] != (H, W):
            pred_mask = F.interpolate(
                pred_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, H', W']
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # [H, W]

        target_mask = target[b, 0, :, :]  # [H, W]
        # print(f"[process] target_masks==1: {(target_mask==1.0).sum()}")
        pred_flat = pred_mask.view(-1)  # [H*W]
        target_flat = target_mask.view(-1)  # [H*W]

        loss_focal_b = sigmoid_focal_loss(pred_flat, target_flat, num_boxes=1.0)
        loss_dice_b = dice_loss(pred_flat, target_flat, num_boxes=1.0)

        batch_losses_focal.append(loss_focal_b)
        batch_losses_dice.append(loss_dice_b)
        batch_preds.append(pred_mask)  # [H, W]

    loss_focal = sum(batch_losses_focal) / B
    loss_dice = sum(batch_losses_dice) / B
    loss = loss_focal + weight_dice * loss_dice

    if mode == "test":
        return loss, loss_focal, loss_dice, batch_preds
    else:
        return loss, loss_focal, loss_dice

def compute_metrics(pred, target, smooth=1e-6):
    '''
    pred: tensor [B, H, W] logits值（NOT sigmoid后）
    target: tensor [B, H, W] 01的GT
    '''
    pred = torch.sigmoid(pred).detach()
    target = target.float()

    pred= (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou = intersection / (union - intersection + smooth)

    return iou.item(), dice.item()
