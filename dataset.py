"""
CheXpert数据集
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CheXpertDataset(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        """
        Args:
            cfg: Config对象
            split: "train" 或 "valid"
            transform: 图像变换
        """
        self.cfg = cfg
        self.transform = transform
        self.labels = cfg.train_labels
        
        # 读取CSV
        csv_path = os.path.join(cfg.data.csv_dir, 
                                cfg.data.train_csv if split == "train" else cfg.data.valid_csv)
        self.df = pd.read_csv(csv_path)
        
        # 预处理标签：-1 根据配置处理
        self.df = self._preprocess_labels()
        
        print(f"[{split}] 共 {len(self.df)} 张图像, {len(self.labels)} 个标签")
    
    def _preprocess_labels(self):
        """处理不确定值(-1)"""
        df = self.df.copy()
        
        # 确保所有标签列都存在
        for label in self.labels:
            if label not in df.columns:
                df[label] = 0.0
            else:
                df[label] = pd.to_numeric(df[label], errors="coerce").fillna(0.0)
        
        # 处理不确定值
        if self.cfg.uncertain_handling == "ignore":
            # -1 当作 0
            for label in self.labels:
                df[label] = df[label].apply(lambda x: 0.0 if x == -1 else x)
        elif self.cfg.uncertain_handling == "ones":
            # -1 当作 1
            for label in self.labels:
                df[label] = df[label].apply(lambda x: 1.0 if x == -1 else x)
        
        return df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 读取图像
        img_path = os.path.join(self.cfg.data.image_dir, row["Path"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # 图像读取失败，返回黑图
            image = Image.new("RGB", (self.cfg.training.image_size, self.cfg.training.image_size), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # 标签
        labels = torch.tensor([row[label] for label in self.labels], dtype=torch.float32)
        
        return image, labels


def get_transforms(cfg, split="train"):
    """获取数据增强"""
    size = cfg.training.image_size
    
    if split == "train":
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
