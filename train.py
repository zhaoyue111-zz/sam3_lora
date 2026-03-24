import os
import torch
import torchvision
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import load_model, load_data, save_checkpoint, compute_metrics, process_outputs

torch.cuda.empty_cache()


def train_epoch(lora_model, train_dataloader, optimizer, scaler, epoch, num_epochs, writer, device):
    lora_model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch_idx, (inputs, target_masks) in enumerate(progress_bar):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target_masks = target_masks.to(device).float()  # [B, 1, H_gt, W_gt]

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = lora_model(**inputs)
            loss, loss_focal, loss_dice = process_outputs(outputs, target_masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Focal": f"{loss_focal.item():.4f}",
            "Dice": f"{loss_dice.item():.4f}"
        })

    avg_train_loss = epoch_loss / len(train_dataloader)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    print(f"Epoch {epoch + 1} 完成，训练平均损失: {avg_train_loss:.4f}")


@torch.no_grad()
def valid_epoch(lora_model, val_dataloader, epoch, writer, device):
    lora_model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    for batch_idx, (inputs, target_masks) in enumerate(val_dataloader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target_masks = target_masks.to(device).float()

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = lora_model(**inputs)
            loss, loss_focal, loss_dice, batch_preds = process_outputs(outputs, target_masks, mode="test")

            preds_stack = torch.stack(batch_preds, dim=0)  # [B, H, W]
            # preds_stack=preds_stack.unsqueeze(1) # [B,1,H,W]
            targets_stack = target_masks.squeeze(1)  # [B, H, W]
            iou_b, dice_b = compute_metrics(preds_stack, targets_stack)

        val_loss += loss.item()
        val_dice += dice_b
        val_iou += iou_b

        if 5< batch_idx < 11:
            preds_vis = (torch.sigmoid(preds_stack) > 0.5).float() * 255
            gts_vis = targets_stack.float() * 255

            grid_pred = torchvision.utils.make_grid(preds_vis.unsqueeze(1), normalize=False)
            grid_gt = torchvision.utils.make_grid(gts_vis.unsqueeze(1), normalize=False)

            writer.add_image(f"Val/Pred_epoch{epoch + 1}_batch{batch_idx}", grid_pred, epoch)
            writer.add_image(f"Val/GT_epoch{epoch + 1}_batch{batch_idx}", grid_gt, epoch)

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_dice = val_dice / len(val_dataloader)
    avg_val_iou = val_iou / len(val_dataloader)

    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Dice/val", avg_val_dice, epoch)
    writer.add_scalar("IOU/val", avg_val_iou, epoch)

    print(f"验证集: 损失={avg_val_loss:.4f}, Dice={avg_val_dice:.4f}, IOU={avg_val_iou:.4f}")

    return avg_val_dice


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processor, lora_model = load_model()
    train_dataloader, val_dataloader = load_data(processor)

    scaler = GradScaler(device="cuda")

    optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01
    )

    writer = SummaryWriter(log_dir="./runs/sam3_lora")

    num_epochs = 20
    best_val_dice = 0.0
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_epoch(lora_model, train_dataloader, optimizer, scaler, epoch, num_epochs, writer, device)
        avg_val_dice = valid_epoch(lora_model, val_dataloader, epoch, writer, device)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_path = os.path.join(checkpoint_dir, f"best_model_dice_{best_val_dice:.4f}.pt")
            save_checkpoint(lora_model, optimizer, epoch, save_path)
            print(f"保存最佳模型，Dice={best_val_dice:.4f}")

    writer.close()
    print("训练完成！")


if __name__ == '__main__':
    main()
