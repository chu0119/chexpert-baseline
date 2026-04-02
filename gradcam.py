"""
Grad-CAM 可视化
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import Config
from model import CheXpertModel


def visualize_gradcam(
    image_path: str,
    model: CheXpertModel,
    cfg: Config,
    label_idx: int = 0,
    save_path: str = "gradcam_output.png",
):
    """
    对单张图像生成Grad-CAM热力图
    
    Args:
        image_path: 图像路径
        model: 加载好权重的模型
        cfg: 配置
        label_idx: 要可视化的标签索引
        save_path: 保存路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((cfg.training.image_size, cfg.training.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Grad-CAM目标：最大化指定标签的输出
    targets = [ClassifierOutputTarget(label_idx)]
    
    # 选择最后一层卷积作为目标层
    # timm模型的最后一层卷积通常叫这个
    target_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        print("找不到卷积层！")
        return
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # 生成CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]
    
    # 可视化
    # 将原始图像归一化到0-1
    rgb_image = np.array(image.resize((cfg.training.image_size, cfg.training.image_size))) / 255.0
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    
    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image.resize((cfg.training.image_size, cfg.training.image_size)))
    axes[0].set_title("原始图像")
    axes[0].axis("off")
    
    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title(f"热力图: {cfg.train_labels[label_idx]}")
    axes[1].axis("off")
    
    axes[2].imshow(visualization)
    axes[2].set_title("叠加结果")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM可视化")
    parser.add_argument("--image", type=str, required=True, help="图像路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件")
    parser.add_argument("--label_idx", type=int, default=0, help="标签索引")
    parser.add_argument("--output", type=str, default="gradcam_output.png", help="输出路径")
    args = parser.parse_args()
    
    cfg = Config.from_yaml(args.config)
    
    # 加载模型
    model = CheXpertModel(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"标签列表: {cfg.train_labels}")
    print(f"可视化标签: {cfg.train_labels[args.label_idx]} (索引 {args.label_idx})")
    
    visualize_gradcam(args.image, model, cfg, args.label_idx, args.output)


if __name__ == "__main__":
    main()
