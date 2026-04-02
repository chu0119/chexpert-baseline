# CheXpert Baseline - 全国医保影像AI识图大赛 赛道八

基于EfficientNetV2的胸部X光多疾病检测baseline项目。

## 快速开始

### 1. 环境准备

```bash
conda activate yibao
pip install -r requirements.txt
```

### 2. 修改配置

编辑 `config.yaml`，确认数据路径正确：
```yaml
data:
  csv_dir: "E:/yibao/archive"    # 你的CheXpert数据路径
  image_dir: "E:/yibao/archive"
```

### 3. 训练

```bash
python train.py
```

训练日志会实时显示每个epoch的loss和各疾病AUROC。

### 4. Grad-CAM可视化

```bash
python gradcam.py --image path/to/xray.jpg --checkpoint outputs/efficientnetv2_chexpert_best.pth --label_idx 0
```

`--label_idx` 对应：
- 0: Atelectasis（肺不张）
- 1: Cardiomegaly（心肥大）
- 2: Consolidation（肺实变）
- 3: Edema（肺水肿）
- 4: Pleural Effusion（胸腔积液）
- 5: Pneumonia（肺炎）
- 6: Pneumothorax（气胸）

## 项目结构

```
chexpert-baseline/
├── config.yaml          # 配置文件
├── config.py            # 配置类
├── dataset.py           # 数据集
├── model.py             # 模型定义
├── train.py             # 训练脚本
├── gradcam.py           # Grad-CAM可视化
├── requirements.txt     # 依赖
└── outputs/             # 输出目录（训练后生成）
```

## 技术路线

- **Backbone**: EfficientNetV2-S (timm预训练)
- **分类头**: 自定义两层MLP + BatchNorm + Dropout
- **损失函数**: BCEWithLogitsLoss (多标签分类)
- **数据增强**: RandomFlip + Rotation + ColorJitter + Mixup
- **优化器**: AdamW + CosineAnnealing
- **评价指标**: AUROC (每个疾病独立计算 + 平均)
