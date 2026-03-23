import os
import nibabel as nib
import numpy as np
from PIL import Image

import os

def count_files(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

def clean_img_dir(img_dir, gt_dir):
    """
    删除 img_dir 中没有对应 gt_dir 的 nii.gz 文件。
    gt 文件名 = img 文件名 + "_mask"
    """
    # 获取 gt 文件的病例名（去掉 _mask）
    gt_cases = set()
    for fname in os.listdir(gt_dir):
        if fname.endswith(".nii.gz"):
            if fname.endswith("_mask.nii.gz"):
                case_name = fname.replace("_mask.nii.gz", "")
                gt_cases.add(case_name)
            else:
                print(f"{fname} dose not end with _mask.nii.gz")
        else:
            print(f"{fname} is not a nii fill")

    # 遍历 img_dir，删除没有对应 gt 的文件
    for fname in os.listdir(img_dir):
        if fname.endswith(".nii.gz"):
            case_name = fname.replace(".nii.gz", "")
            if case_name not in gt_cases:
                file_path = os.path.join(img_dir, fname)
                os.remove(file_path)
                print(f"Deleted {file_path}")

    if count_files(img_dir) == count_files(gt_dir):
        print(f"ok! file numbers = {count_files(img_dir)}")
    else:
        print(f"error! img_dir = {img_dir}, gt_dir = {gt_dir}")


def process_nii_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.endswith(".nii.gz"):
            print(f"{fname} is not a nii fill")
            continue

        # 提取病例名
        case_name = fname.split("_DCE")[0]

        # 读取 nii.gz 文件
        nii_path = os.path.join(input_folder, fname)
        img = nib.load(nii_path)
        data = img.get_fdata()  # shape: (H, W, Z)

        # 遍历 z 轴 slice
        for idx in range(data.shape[2]):
            slice_data = data[:, :, idx]

            # 原始范围 [-1,1] → 映射到 [0,255]
            print(f"original size of {fname}: {slice_data.shape}")
            # slice_norm = ((slice_data + 1) / 2.0) * 255.0
            # slice_norm = np.clip(slice_norm, 0, 255).astype(np.uint8)
            liver_mask = np.zeros_like(slice_data, dtype=np.uint8)
            liver_mask[slice_data == 5] = 225

            # 转成 PIL Image
            im = Image.fromarray(liver_mask)

            # resize 到 1024×1024
            # im = im.resize((1024, 1024), resample=Image.BILINEAR)

            im = im.resize((256, 256), resample=Image.NEAREST)

            # 保存文件，序号三位补零
            out_name = f"{case_name}_{idx:03d}.png"
            out_path = os.path.join(output_folder, out_name)
            im.save(out_path)

            print(f"Saved {out_path}")

        print(f"image numbers: {count_files(output_folder)}")

if __name__ == "__main__":
    # img_dir = "/home/data4/zy/data/CT_MRI_DATA/images/T2"
    gt_dir = "/home/data4/zy/data/CT_MRI_DATA/labels/EAP"
    # clean_img_dir(img_dir, gt_dir)

    # input_folder = "/home/data4/zy/data/MRI_DATA/Source/train/nii"   # 输入nii.gz文件夹
    output_folder = "/home/data4/zy/data/CT_MRI_DATA/labels/EAP/pngs"                                        # 输出文件夹
    process_nii_folder(gt_dir, output_folder)