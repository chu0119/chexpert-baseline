"""
模型定义
"""
import torch
import torch.nn as nn
import timm


class CheXpertModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 使用timm加载预训练模型
        self.backbone = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=0,  # 去掉分类头，只取特征
            drop_rate=cfg.model.dropout,
        )
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.Dropout(cfg.model.dropout / 2),
            nn.Linear(self.feature_dim // 2, cfg.model.num_classes),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """获取特征向量（用于Grad-CAM）"""
        return self.backbone(x)
