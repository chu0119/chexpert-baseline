"""
数据集和模型配置
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    csv_dir: str = "E:/yibao/archive"
    image_dir: str = "E:/yibao/archive"
    train_csv: str = "train.csv"
    valid_csv: str = "valid.csv"


@dataclass
class ModelConfig:
    name: str = "tf_efficientnetv2_s"
    pretrained: bool = True
    num_classes: int = 7
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    image_size: int = 320
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    mixup_alpha: float = 0.2
    label_smoothing: float = 0.1


@dataclass
class OutputConfig:
    save_dir: str = "outputs"
    model_name: str = "efficientnetv2_chexpert"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    train_labels: List[str] = field(default_factory=lambda: [
        "Atelectasis", "Cardiomegaly", "Consolidation",
        "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"
    ])
    uncertain_handling: str = "ignore"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        
        cfg = cls()
        if "data" in d:
            for k, v in d["data"].items():
                setattr(cfg.data, k, v)
        if "model" in d:
            for k, v in d["model"].items():
                setattr(cfg.model, k, v)
        if "training" in d:
            for k, v in d["training"].items():
                setattr(cfg.training, k, v)
        if "output" in d:
            for k, v in d["output"].items():
                setattr(cfg.output, k, v)
        if "train_labels" in d:
            cfg.train_labels = d["train_labels"]
        if "uncertain_handling" in d:
            cfg.uncertain_handling = d["uncertain_handling"]
        
        cfg.model.num_classes = len(cfg.train_labels)
        return cfg
