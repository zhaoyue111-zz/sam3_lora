import random
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Sam3Processor

class SourceDataset(Dataset):
    def __init__(self, df, image_root, mask_root, processor, text="liver", sequence="Delay",transform=None, subset=0):
        """
        df: 从Excel读取的DataFrame
        image_root: /home/data4/zy/data/CT_MRI_DATA/images/Delay/pngs
        mask_root: /home/data4/zy/data/CT_MRI_DATA/labels/Delay/pngs
        subset: 0表示训练集，1表示验集集，2表示测试集
        """
        self.processor = processor
        self.transform = transform
        self.text = text

        df_subset = df[(df["subset"] == subset) & (df[sequence] == 1)]

        self.image_paths = []
        self.mask_paths = []

        for _, row in df_subset.iterrows():
            uid = row["uid"]

            slice_imgs = sorted(Path(image_root).glob(f"{uid}_*.png"))
            slice_masks = sorted(Path(mask_root).glob(f"{uid}_*.png"))

            # 确保数量匹配
            assert len(slice_imgs) == len(slice_masks), f"{uid} 图像和掩码数量不匹配！"

            self.image_paths.extend(slice_imgs)
            self.mask_paths.extend(slice_masks)

        assert len(self.image_paths) == len(self.mask_paths), "整体图像和掩码数量不匹配！"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR) # ndarray:(1024,1024,3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # rgb 3通道
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE) # ndarray: (256,256)
        mask = (mask>0).astype(np.uint8)

        inputs = self.processor(images=image, text=self.text, return_tensors="pt")
        # {'pixel_values': [1,3,1008,1008], 'original_sizes':[[1024,1024]], 'input_ids':[1,32], 'attention_mask':[1,32]}

        target_mask = torch.from_numpy(mask).unsqueeze(0).float() # [1,256,256]
        if target_mask.shape[-1] != 256:
            target_mask = F.interpolate(target_mask.unsqueeze(0), size=(256, 256),
                                        mode='bilinear', align_corners=False).squeeze(0)

        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        # {'pixel_values': [3,1008,1008], 'original_sizes':[1024,1024], 'input_ids':[32,], 'attention_mask':[32,]}

        return inputs, target_mask

class SourceDatasetLong(Dataset):
    def __init__(self, df, image_root, mask_root, prompt_root, processor, text="liver", sequence="Delay",transform=None, subset=0):
        """
        df: 从Excel读取的DataFrame
        image_root: /home/data4/zy/data/CT_MRI_DATA/images/Delay/pngs
        mask_root: /home/data4/zy/data/CT_MRI_DATA/labels/Delay/pngs
        promot_root:/home/data4/zy/data/CT_MRI_DATA/prompt/train.csv
        subset: 0表示训练集，1表示验集集，2表示测试集
        """
        self.processor = processor
        self.transform = transform
        self.text = text

        df_subset = df[(df["subset"] == subset) & (df[sequence] == 1)]

        self.image_paths = []
        self.mask_paths = []

        for _, row in df_subset.iterrows():
            uid = row["uid"]

            slice_imgs = sorted(Path(image_root).glob(f"{uid}_*.png"))
            slice_masks = sorted(Path(mask_root).glob(f"{uid}_*.png"))

            # 确保数量匹配
            assert len(slice_imgs) == len(slice_masks), f"{uid} 图像和掩码数量不匹配！"

            self.image_paths.extend(slice_imgs)
            self.mask_paths.extend(slice_masks)

        assert len(self.image_paths) == len(self.mask_paths), "整体图像和掩码数量不匹配！"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR) # ndarray:(1024,1024,3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # rgb 3通道
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE) # ndarray: (256,256)
        mask = (mask>0).astype(np.uint8)

        inputs = self.processor(images=image, text=self.text, return_tensors="pt")
        # {'pixel_values': [1,3,1008,1008], 'original_sizes':[[1024,1024]], 'input_ids':[1,32], 'attention_mask':[1,32]}

        target_mask = torch.from_numpy(mask).unsqueeze(0).float() # [1,256,256]
        if target_mask.shape[-1] != 256:
            target_mask = F.interpolate(target_mask.unsqueeze(0), size=(256, 256),
                                        mode='bilinear', align_corners=False).squeeze(0)

        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        # {'pixel_values': [3,1008,1008], 'original_sizes':[1024,1024], 'input_ids':[32,], 'attention_mask':[32,]}

        return inputs, target_mask

if __name__ == '__main__':
    df = pd.read_excel("/home/data4/zy/data/CT_MRI_DATA/MRI_mapping_info_labeled.csv")
    image_root = "/home/data4/zy/data/CT_MRI_DATA/images/Delay/pngs"
    mask_root = "/home/data4/zy/data/CT_MRI_DATA/labels/Delay/pngs"
    # train_uids, val_uids = split_train_val(df, train_ratio=0.8, subset=0)

    path = "/home/data4/zy/weight/sam3"
    processor = Sam3Processor.from_pretrained(path)

    train_dataset = SourceDataset(df, image_root, mask_root, processor, subset=0)
    val_dataset = SourceDataset(df, image_root, mask_root, processor, subset=1)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    for inputs, targets in train_dataloader:
        for i in range(4):
            print((targets[i]==0).sum(), (targets[i]==1).sum())
        break

    print(len(train_dataset))
    print(len(val_dataset))
