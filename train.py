"""
训练和评估
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import Config
from dataset import CheXpertDataset, get_transforms
from model import CheXpertModel


def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # 确保lam >= 0.5
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = CheXpertModel(cfg).to(self.device)
        
        # 损失函数（BCEWithLogitsLoss，多标签分类标准损失）
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = None  # 在训练开始时根据数据量设置
        
        # 最佳指标
        self.best_auroc = 0.0
        
        # 输出目录
        os.makedirs(cfg.output.save_dir, exist_ok=True)
    
    def setup_scheduler(self, num_training_steps):
        """设置学习率调度器"""
        if self.cfg.training.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=self.cfg.training.learning_rate * 0.01,
            )
    
    def train_one_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg.training.epochs} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Mixup
            if self.cfg.training.mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, self.cfg.training.mixup_alpha
                )
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            
            if self.cfg.training.mixup_alpha > 0:
                loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
            else:
                loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        
        pbar = tqdm(val_loader, desc="[Valid]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
        
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # 计算AUROC（每个标签分别计算，然后取平均）
        auroc_per_label = []
        for i, label_name in enumerate(self.cfg.train_labels):
            try:
                auroc = roc_auc_score(all_labels[:, i], all_logits[:, i])
                auroc_per_label.append((label_name, auroc))
            except ValueError:
                auroc_per_label.append((label_name, 0.0))
        
        avg_auroc = np.mean([v for _, v in auroc_per_label])
        
        return total_loss / len(val_loader), avg_auroc, auroc_per_label
    
    def save_checkpoint(self, epoch, avg_auroc, is_best=False):
        """保存模型"""
        path = os.path.join(
            self.cfg.output.save_dir,
            f"{self.cfg.output.model_name}_epoch{epoch:02d}.pth"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "auroc": avg_auroc,
            "config": {
                "model_name": self.cfg.model.name,
                "num_classes": self.cfg.model.num_classes,
                "train_labels": self.cfg.train_labels,
            }
        }, path)
        
        if is_best:
            best_path = os.path.join(
                self.cfg.output.save_dir,
                f"{self.cfg.output.model_name}_best.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "auroc": avg_auroc,
                "config": {
                    "model_name": self.cfg.model.name,
                    "num_classes": self.cfg.model.num_classes,
                    "train_labels": self.cfg.train_labels,
                }
            }, best_path)
    
    def train(self):
        """完整训练流程"""
        # 数据加载
        train_dataset = CheXpertDataset(self.cfg, split="train", transform=get_transforms(self.cfg, "train"))
        val_dataset = CheXpertDataset(self.cfg, split="valid", transform=get_transforms(self.cfg, "valid"))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.training.num_workers > 0 else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.training.batch_size * 2,  # 验证时用更大batch
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.training.num_workers > 0 else False,
        )
        
        # 设置学习率调度器
        total_steps = self.cfg.training.epochs * len(train_loader)
        self.setup_scheduler(total_steps)
        
        print(f"\n{'='*60}")
        print(f"开始训练: {self.cfg.model.name}")
        print(f"训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张")
        print(f"标签: {self.cfg.train_labels}")
        print(f"Epochs: {self.cfg.training.epochs} | Batch: {self.cfg.training.batch_size}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.cfg.training.epochs):
            start_time = time.time()
            
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, avg_auroc, auroc_per_label = self.validate(val_loader)
            epoch_time = time.time() - start_time
            
            # 打印每个标签的AUROC
            label_str = " | ".join([f"{n}: {v:.3f}" for n, v in auroc_per_label])
            print(f"\nEpoch {epoch+1} | Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Avg AUROC: {avg_auroc:.4f}")
            print(f"  Per-label: {label_str}")
            
            # 保存checkpoint
            is_best = avg_auroc > self.best_auroc
            if is_best:
                self.best_auroc = avg_auroc
                print(f"  ★ 新最佳 AUROC: {avg_auroc:.4f}")
            
            self.save_checkpoint(epoch, avg_auroc, is_best)
            print()
        
        print(f"训练完成! 最佳 AUROC: {self.best_auroc:.4f}")
        print(f"最佳模型保存在: {self.cfg.output.save_dir}/{self.cfg.output.model_name}_best.pth")


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    trainer = Trainer(cfg)
    trainer.train()
