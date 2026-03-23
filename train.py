import os
import torch
import torchvision
from torch.amp import GradScaler,autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from loss import dice_loss, boundary_loss
from utils import load_model, load_data, post_process_batch_outputs, save_checkpoint, compute_metrics

torch.cuda.empty_cache()


def train_epoch(lora_model, train_dataloader, optimizer, scaler, epoch, num_epochs, writer, device):
    lora_model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch_idx, (inputs, target_masks) in enumerate(progress_bar):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target_masks = target_masks.to(device).float() # [B,1,256,256]

        B, C, H, W = target_masks.shape
        target_sizes = torch.tensor([H, W]).expand(B, 2)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = lora_model(**inputs)

            pred_logits,pred_masks = post_process_batch_outputs(outputs,target_sizes,device, threshold=0.5,mask_threshold=0.5)

            loss_bce = F.binary_cross_entropy_with_logits(pred_logits, target_masks)
            loss_boundary = boundary_loss(pred_masks, target_masks)
            loss = loss_bce + loss_boundary

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # 缩放后反向传播
        scaler.step(optimizer) # 检查是否有 NaN/Inf
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_train_loss = epoch_loss / len(train_dataloader)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    print(f"Epoch {epoch + 1} 完成，训练平均损失: {avg_train_loss:.4f}")


@torch.no_grad()
def valid_epoch(lora_model, val_dataloader, epoch, writer, device):
    lora_model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_mio=0.0

    for batch_idx, (inputs, target_masks) in enumerate(val_dataloader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target_masks = target_masks.to(device)
        B, C, H, W = target_masks.shape
        target_sizes = torch.tensor([H, W]).expand(B, 2)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = lora_model(**inputs)

            pred_logits,pred_masks = post_process_batch_outputs(outputs,target_sizes,device, threshold=0.5,mask_threshold=0.5)

            loss_bce = F.binary_cross_entropy_with_logits(pred_logits, target_masks)
            loss_boundary = boundary_loss(pred_masks, target_masks)
            loss = loss_bce + loss_boundary

            miou,dice=compute_metrics(pred_masks,target_masks)

        val_loss += loss.item()
        val_mio+=miou
        val_dice+=dice

        # 保存部分样本到 TensorBoard
        if batch_idx < 2:
            preds_bin = (pred_masks.sigmoid() > 0.5).float() * 255
            gts_bin = target_masks.float() * 255

            grid_pred = torchvision.utils.make_grid(preds_bin, normalize=False)
            grid_gt = torchvision.utils.make_grid(gts_bin, normalize=False)

            writer.add_image(f"Val/Pred_epoch{epoch + 1}_batch{batch_idx}", grid_pred, epoch)
            writer.add_image(f"Val/GT_epoch{epoch + 1}_batch{batch_idx}", grid_gt, epoch)

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_dice = val_dice / len(val_dataloader)
    avg_val_miou=val_mio / len(val_dataloader)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Dice/val", avg_val_dice, epoch)
    writer.add_scalar("MIou/val", avg_val_miou, epoch)
    print(f"验证集: 平均损失={avg_val_loss:.4f}, 平均Dice={avg_val_dice:.4f}, MIou={avg_val_miou:.4f}")

    return avg_val_dice


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ============ 加载模型和数据 ============
    processor, lora_model = load_model()
    train_dataloader, val_dataloader = load_data(processor)

    # ============ 优化器和 Scaler ============
    scaler = GradScaler(device="cuda")

    optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.01
    )
    # processor.post_process_instance_segmentation
    # ============ TensorBoard ============
    writer = SummaryWriter(log_dir="./runs/sam3_lora")

    # ============ 训练循环 ============
    num_epochs = 20
    best_val_dice = 0.0
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # ----------- Training -----------
        train_epoch(lora_model, train_dataloader, optimizer, scaler, epoch, num_epochs, writer, device)

        # ----------- Validation -----------
        avg_val_dice = valid_epoch(lora_model, val_dataloader, epoch, writer, device)

        # ----------- Save best model -----------
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_path = os.path.join(checkpoint_dir, f"best_model_dice_{best_val_dice:.4f}.pt")
            save_checkpoint(lora_model, optimizer, epoch, save_path)
            print(f"保存最佳模型，Dice={best_val_dice:.4f}")

    writer.close()
    print("训练完成！")


if __name__ == '__main__':
    main()