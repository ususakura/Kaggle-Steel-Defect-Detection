# coding: utf-8

import os
import cv2
import warnings
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset, sampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import sys

sys.path.append('.')
from utils.data_augmentation import data_augmentation
from utils.rle_parse import mask2rle, make_mask
from utils.visualize import image_with_mask_torch
warnings.filterwarnings("ignore")


# === 在这里给出一个调试打印函数，可以更方便加前缀
def debug_print(msg):
    print(f"[DEBUG] {msg}")

# Dataset Segmentation
class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase, crop=False, height=None, width=None):
        super(SteelDataset, self).__init__()
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms
        self.crop = crop
        self.height = height
        self.width = width
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask_data = make_mask(idx, self.df)

        # === 构造图像路径
        image_path = os.path.join(self.root, "train_images", image_id)
        img = cv2.imread(image_path)  # BGR格式，shape=(H,W,3) or None if not found

        # === 调试打印：查看我们读到的 img 是否为空
        if img is None:
            debug_print(f"Failed to load image from path: {image_path}")
        else:
            debug_print(f"Loaded image from {image_path}, shape={img.shape}")

        # === 这里 mask_data 是个 numpy array (H,W,4) 吗？取决于 make_mask 的实现
        #     先打印一下它的 shape
        debug_print(f"Mask shape before transform: {mask_data.shape}")

        # === 调用 transforms
        img, mask = self.transforms(
            self.phase, 
            img, 
            mask_data, 
            self.mean, 
            self.std, 
            crop=self.crop, 
            height=self.height, 
            width=self.width
        )

        # === transforms 之后的 img 是个 Tensor, mask 也是个 Tensor (或 numpy->Tensor)
        #     对于分割任务，mask 通常是 shape=(C,H,W)
        #     这里你再做一次 permute(2,0,1) 会不会出错？看你原先代码想做什么
        mask = mask.permute(2, 0, 1)

        return img, mask

    def __len__(self):
        return len(self.fnames)


# Dataset Classification
class SteelClassDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase, crop=False, height=None, width=None):
        super(SteelClassDataset, self).__init__()
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms
        self.crop = crop
        self.height = height
        self.width = width
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask_data = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images", image_id)
        img = cv2.imread(image_path)

        # === 调试打印
        if img is None:
            debug_print(f"Failed to load image from path: {image_path}")
        else:
            debug_print(f"Loaded image from {image_path}, shape={img.shape}")
        debug_print(f"Mask shape before transform: {mask_data.shape}")

        img, mask = self.transforms(
            self.phase, 
            img, 
            mask_data, 
            self.mean, 
            self.std, 
            crop=self.crop, 
            height=self.height, 
            width=self.width
        )

        # 下面这行 permute 只对 shape=(H,W,4) 之类的才适用
        mask = mask.permute(2, 0, 1)  # 4x256x1600

        # 这里又做了个 reshape，逻辑看你原本想干啥
        mask = mask.view(mask.size(0), -1)
        mask = torch.sum(mask, dim=1)
        mask = mask > 0
        return img, mask.float()

    def __len__(self):
        return len(self.fnames)


class TestDataset(Dataset):
    '''Dataset for test prediction'''

    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image)
        return fname, images

    def __len__(self):
        return self.num_samples


def augmentation(image, mask, crop=False, height=None, width=None):
    """进行数据增强
    Args:
        image: 原始图像 (numpy array)
        mask: 原始掩膜 (numpy array)
    Return:
        image_aug: 增强后的图像，PIL图像
        mask_aug: 增强后的掩膜 (numpy array)
    """
    image_aug, mask_aug = data_augmentation(
        image, 
        mask, 
        crop=crop, 
        height=height, 
        width=width
    )
    image_aug = Image.fromarray(image_aug)
    return image_aug, mask_aug


def get_transforms(phase, image, mask, mean, std, crop=False, height=1600, width=256):
    """把数据增强 (train时) + ToTensor + Normalize 串起来"""

    # === 在这里打印一下，看它实际拿到的 height 和 width 多少
    debug_print(f"get_transforms: phase={phase}, height={height}, width={width}, image_type={type(image)}, mask_type={type(mask)}")

    if phase == 'train':
        image, mask = augmentation(image, mask, crop=crop, height=height, width=width)

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    transform_compose = transforms.Compose([to_tensor, normalize])
    image = transform_compose(image)

    # === mask是numpy array，对其转成PyTorch的Tensor
    mask = torch.from_numpy(mask)
    return image, mask


def mask_only_collate_fun(batch):
    """自定义collate_fn函数，用于从一个batch中去除没有掩膜的样本"""
    batch_scale = []
    for image, mask in batch:
        mask_pixel_num = torch.sum(mask)
        if mask_pixel_num > 0:
            batch_scale.append([image, mask])
    if len(batch_scale) > 0:
        batch_scale = default_collate(batch_scale)
    else:
        batch_scale = torch.tensor(batch_scale)
    return batch_scale


def provider(
    data_folder,
    df_path,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
    n_splits=0,
    mask_only=False,
    crop=False, 
    height=None,
    width=None
):
    """返回数据加载器，用于分割模型
    """
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    
    # 将数据集划分为n_split份
    train_dfs = []
    val_dfs = []
    if n_splits != 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=69)
        for train_df_index, val_df_index in skf.split(df, df['defects']):
            train_dfs.append(df.loc[df.index[train_df_index]])
            val_dfs.append(df.loc[df.index[val_df_index]])
    else:        
        df_temp = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
        train_dfs.append(df_temp[0])
        val_dfs.append(df_temp[1])

    dataloaders = []
    for df_index, (train_df, val_df) in enumerate(zip(train_dfs, val_dfs)):
        train_dataset = SteelDataset(
            train_df, 
            data_folder, 
            mean, 
            std, 
            'train', 
            crop=crop, 
            height=height, 
            width=width
        )
        val_dataset = SteelDataset(val_df, data_folder, mean, std, 'val')
        if mask_only:
            print('Segmentation modle: only masked data.')
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                collate_fn=mask_only_collate_fun,
                pin_memory=True, 
                shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size, 
                num_workers=num_workers,
                collate_fn=mask_only_collate_fun, 
                pin_memory=True, 
                shuffle=False
            )
        else:    
            print('Segmentation model: all data.')
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                pin_memory=True, 
                shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size, 
                num_workers=num_workers, 
                pin_memory=True, 
                shuffle=False
            )
        dataloaders.append([train_dataloader, val_dataloader])

    return dataloaders


def classify_provider(
    data_folder,
    df_path,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
    n_splits=0,
    crop=False,
    height=None,
    width=None
):
    """返回数据加载器，用于分类模型
    """
    df = pd.read_csv(df_path)
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    
    train_dfs = []
    val_dfs = []
    if n_splits != 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=69)
        for train_df_index, val_df_index in skf.split(df, df['defects']):
            train_dfs.append(df.loc[df.index[train_df_index]])
            val_dfs.append(df.loc[df.index[val_df_index]])
    else:        
        df_temp = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
        train_dfs.append(df_temp[0])
        val_dfs.append(df_temp[1])
    
    dataloaders = []
    for df_index, (train_df, val_df) in enumerate(zip(train_dfs, val_dfs)):
        train_dataset = SteelClassDataset(
            train_df, 
            data_folder, 
            mean, 
            std, 
            'train', 
            crop=crop, 
            height=height, 
            width=width
        )
        val_dataset = SteelClassDataset(val_df, data_folder, mean, std, 'val')

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            shuffle=False
        )
        dataloaders.append([train_dataloader, val_dataloader])

    return dataloaders


if __name__ == "__main__":
    data_folder = "datasets/Steel_data"
    df_path = "datasets/Steel_data/train.csv"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 12
    num_workers = 4
    n_splits = 1
    mask_only = False
    crop = False
    height = 256
    width = 512
    # 测试分割数据集
    dataloader = provider(
        data_folder, df_path, mean, std, 
        batch_size, num_workers, 
        n_splits, mask_only=mask_only, 
        crop=crop, height=height, width=width
    )
    class_dataloader = classify_provider(
        data_folder, df_path, mean, std, 
        batch_size, num_workers, 
        n_splits
    )
    for fold_index, [[train_dataloader, val_dataloader], [classify_train_dataloader, classify_val_dataloader]] in enumerate(zip(dataloader, class_dataloader)):
        val_bar = tqdm(val_dataloader)
        classify_val_bar = tqdm(classify_val_dataloader)
        class_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [139, 0, 139]]
        for [[images, targets], [classify_images, classify_targets]] in zip(val_bar, classify_val_bar):
            image = images[0]
            target = targets[0]
            image = image_with_mask_torch(image, target, mean, std)['image']
            classify_target = classify_targets[0]
            position_x = 10
            for i in range(classify_target.size(0)):
                color = class_color[i]
                position_x += 50
                position = (position_x, 50)
                if classify_target[i] != 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    image = cv2.putText(image, str(i+1), position, font, 1.2, color, 2) 
            cv2.imshow('win', image)
            cv2.waitKey(0)

    pass
