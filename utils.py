import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch.utils.data import DataLoader
from transformers import Sam3Model, Sam3Processor
from dataset import SourceDataset, split_train_val
from torch.nn import DataParallel
import torch.nn.functional as F

def load_model():
    path = "/home/data4/zy/weight/sam3"
    base_model = Sam3Model.from_pretrained(path)
    processor = Sam3Processor.from_pretrained(path)

    lora_config = LoraConfig(
        r=12,
        lora_alpha=24,
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
            if layer_num < 16:  #  冻结前 16 层，只训练后 16 层
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
    # 如果使用 DataParallel，需要获取底层模型
    model_to_save = lora_model.module if isinstance(lora_model, DataParallel) else lora_model

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"检查点已保存: {save_path}")


def post_process_batch_outputs(outputs, target_sizes, device, threshold=0.5, mask_threshold=0.5):
    """
    参考Sam3ImageProcessorFast.post_process_instance_segmentation实现
    返回：
    - pred_logits: 融合后的 logits [B, 1, H', W']
    - pred_masks: sigmoid后的[B, 1, H', W']
    """
    pred_masks_logits = outputs.pred_masks  # [B, num_queries, H, W]
    pred_logits = outputs.pred_logits  # [B, num_queries]
    presence_logits = outputs.presence_logits # [B, 1] or None
    batch_size = pred_logits.shape[0]

    batch_logits = []
    batch_masks = []

    for b in range(batch_size):
        scores = pred_logits[b].float().sigmoid()  # [num_queries]
        if presence_logits is not None:
            presence_score = presence_logits[b].float().sigmoid()  # [1]
            scores = scores * presence_score  # Broadcast: [num_queries] * [1] = [num_queries]

        masks_logits = pred_masks_logits[b]  # [num_queries, H, W]

        #  阈值筛选（对应源码中的 keep = scores > threshold）
        keep = scores > threshold
        if keep.sum() == 0:
            # 没有 score > threshold，直接返回全 0 logits（不经过索引操作）
            h, w = target_sizes[b]
            combined_logits = torch.zeros((h, w), device=masks_logits.device, requires_grad=True)
        else:
            # 有有效 score，正常加权平均
            valid_scores = scores[keep]  # [num_keep]
            valid_masks_logits = masks_logits[keep]  # [num_keep, H, W]
            target_size = tuple(target_sizes[b])
            logits_resized = F.interpolate(
                valid_masks_logits.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            scores_sum = valid_scores.sum()

            print(f"scores_sum: {scores_sum}")
            if scores_sum > 1e-8:
                combined_logits = (logits_resized * valid_scores.view(-1, 1, 1)).sum(dim=0) / scores_sum
            else:
                # scores_sum 接近 0，返回全 0
                h, w = target_sizes[b]
                combined_logits = torch.zeros((h, w), device=masks_logits.device, requires_grad=True)

        batch_logits.append(combined_logits.unsqueeze(0))
        batch_masks.append(combined_logits.sigmoid().unsqueeze(0))

    pred_logits = torch.stack(batch_logits, dim=0)  # [B, 1, H, W]
    pred_masks = torch.stack(batch_masks, dim=0)  # [B, 1, H, W]

    pred_logits = pred_logits.to(device)
    pred_masks = pred_masks.to(device)

    assert pred_logits.requires_grad, "pred_logits_combined 没有梯度"

    return pred_logits, pred_masks


def compute_metrics(pred, target,smooth=1e-6):
    '''
        pred: tensor [B,1,H,W] 只有01
        target: tensor [B,1,H,W]  只有01
    '''
    pred = (pred>0.5).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou=intersection / union if union > 0 else 1.0

    return iou,dice