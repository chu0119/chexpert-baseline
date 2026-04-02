"""
CheXpert 可视化仪表板
科技感深色主题，实时展示训练状态、Grad-CAM热力图和推理结果
"""
import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from torchvision import transforms
from datetime import datetime

from config import Config
from model import CheXpertModel

app = Flask(__name__)

# 全局配置和模型
cfg = None
model = None
device = None
LABEL_COLORS = [
    "#00f5d4", "#00bbf9", "#fee440", "#f15bb5",
    "#9b5de5", "#00f5d4", "#fb5607"
]


def load_model(checkpoint_path):
    global model, cfg, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config.from_yaml("config.yaml")
    model = CheXpertModel(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ 模型加载成功: {checkpoint_path} (AUROC: {checkpoint.get('auroc', 'N/A')})")


def get_transform():
    return transforms.Compose([
        transforms.Resize((cfg.training.image_size, cfg.training.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


def load_training_history():
    """从训练日志中解析历史数据"""
    history = {"epochs": [], "train_loss": [], "val_loss": [], "avg_auroc": [], "per_label": {}}
    # 尝试从JSON文件加载
    json_path = os.path.join(cfg.output.save_dir, "training_history.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            history = json.load(f)
    return history


def find_best_checkpoint():
    """找到最佳模型"""
    pattern = os.path.join(cfg.output.save_dir, f"{cfg.output.model_name}_best.pth")
    if os.path.exists(pattern):
        return pattern
    # 找最新的checkpoint
    checkpoints = glob.glob(os.path.join(cfg.output.save_dir, "*.pth"))
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)
    return None


# ============ HTML Templates ============

INDEX_HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CheXpert AI Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --bg-primary: #0a0a1a;
            --bg-secondary: #111128;
            --bg-card: #161638;
            --bg-hover: #1e1e4a;
            --accent-cyan: #00f5d4;
            --accent-blue: #00bbf9;
            --accent-purple: #9b5de5;
            --accent-pink: #f15bb5;
            --accent-yellow: #fee440;
            --accent-orange: #fb5607;
            --text-primary: #e8e8f0;
            --text-secondary: #8888aa;
            --text-dim: #555577;
            --border: #2a2a5a;
            --glow-cyan: 0 0 20px rgba(0, 245, 212, 0.3);
            --glow-blue: 0 0 20px rgba(0, 187, 249, 0.3);
        }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* 背景动效 */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                radial-gradient(ellipse at 20% 50%, rgba(0, 245, 212, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(155, 93, 229, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(0, 187, 249, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        
        /* 导航 */
        .nav {
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 32px;
            background: rgba(10, 10, 26, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
        }
        
        .nav-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 18px;
            color: var(--accent-cyan);
        }
        
        .nav-brand .icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        
        .nav-links {
            display: flex;
            gap: 8px;
        }
        
        .nav-links a {
            padding: 8px 20px;
            border-radius: 8px;
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
            border: 1px solid transparent;
        }
        
        .nav-links a:hover, .nav-links a.active {
            color: var(--accent-cyan);
            background: rgba(0, 245, 212, 0.1);
            border-color: rgba(0, 245, 212, 0.2);
        }
        
        /* 主体 */
        .container {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px 32px;
        }
        
        .page { display: none; }
        .page.active { display: block; }
        
        /* 卡片 */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            transition: all 0.3s;
        }
        
        .card:hover {
            border-color: rgba(0, 245, 212, 0.3);
            box-shadow: var(--glow-cyan);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        
        .card-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .card-title .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-cyan);
            box-shadow: 0 0 10px var(--accent-cyan);
        }
        
        /* 状态卡片网格 */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        }
        
        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-label {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        
        /* 指标条 */
        .metric-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid rgba(42, 42, 90, 0.5);
        }
        
        .metric-bar:last-child { border-bottom: none; }
        
        .metric-name {
            width: 140px;
            font-size: 13px;
            color: var(--text-secondary);
            flex-shrink: 0;
        }
        
        .metric-bar-bg {
            flex: 1;
            height: 8px;
            background: rgba(42, 42, 90, 0.5);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .metric-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
        }
        
        .metric-value {
            width: 60px;
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 600;
            color: var(--accent-cyan);
        }
        
        /* 上传区域 */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: 16px;
            padding: 60px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
        }
        
        .upload-zone:hover {
            border-color: var(--accent-cyan);
            background: rgba(0, 245, 212, 0.05);
        }
        
        .upload-zone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(0, 245, 212, 0.1);
            box-shadow: var(--glow-cyan);
        }
        
        .upload-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        
        .upload-text {
            font-size: 16px;
            color: var(--text-secondary);
        }
        
        .upload-hint {
            font-size: 13px;
            color: var(--text-dim);
            margin-top: 8px;
        }
        
        /* 推理结果 */
        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        
        .result-image {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        
        .result-image img {
            width: 100%;
            display: block;
        }
        
        .prediction-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 16px;
            border-radius: 10px;
            margin-bottom: 8px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            transition: all 0.3s;
        }
        
        .prediction-item.high-risk {
            border-color: rgba(241, 91, 181, 0.5);
            background: rgba(241, 91, 181, 0.1);
        }
        
        .prediction-name {
            flex: 1;
            font-size: 14px;
        }
        
        .prediction-prob {
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 700;
        }
        
        .prediction-badge {
            padding: 2px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .badge-positive { background: rgba(241, 91, 181, 0.2); color: var(--accent-pink); }
        .badge-negative { background: rgba(0, 245, 212, 0.2); color: var(--accent-cyan); }
        
        /* 图表容器 */
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        .chart-container canvas {
            width: 100% !important;
            height: 100% !important;
        }
        
        /* 加载动画 */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 40px;
            color: var(--text-secondary);
        }
        
        .spinner {
            width: 24px;
            height: 24px;
            border: 3px solid var(--border);
            border-top-color: var(--accent-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* 脉冲动画 */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .pulse { animation: pulse 2s ease-in-out infinite; }
        
        /* 滚动条 */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
    </style>
</head>
<body>
    <!-- 导航 -->
    <nav class="nav">
        <div class="nav-brand">
            <div class="icon">🫁</div>
            <span>CheXpert AI</span>
        </div>
        <div class="nav-links">
            <a href="#" class="active" onclick="showPage('dashboard')">📊 仪表盘</a>
            <a href="#" onclick="showPage('predict')">🔍 智能诊断</a>
            <a href="#" onclick="showPage('gradcam')">🔥 热力图</a>
        </div>
    </nav>
    
    <div class="container">
        <!-- 仪表盘页面 -->
        <div id="page-dashboard" class="page active">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ best_auroc }}</div>
                    <div class="stat-label">最佳 AUROC</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ current_epoch }}/{{ total_epochs }}</div>
                    <div class="stat-label">训练轮次</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ model_name }}</div>
                    <div class="stat-label">模型架构</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ num_labels }}</div>
                    <div class="stat-label">检测疾病数</div>
                </div>
            </div>
            
            <!-- 各疾病AUROC -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><span class="dot"></span> 各疾病检测性能</div>
                </div>
                <div>
                    {% for label, auroc in per_label_auroc.items() %}
                    <div class="metric-bar">
                        <div class="metric-name">{{ label }}</div>
                        <div class="metric-bar-bg">
                            <div class="metric-bar-fill" style="width: {{ auroc * 100 }}%; background: linear-gradient(90deg, {{ colors[loop.index0] }}, {{ colors[(loop.index0+1) % colors|length] }})"></div>
                        </div>
                        <div class="metric-value" style="color: {{ colors[loop.index0] }}">{{ "%.4f"|format(auroc) }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <!-- 训练曲线 -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><span class="dot" style="background: var(--accent-cyan)"></span> Loss 曲线</div>
                    </div>
                    <div class="chart-container">
                        <canvas id="lossChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><span class="dot" style="background: var(--accent-purple)"></span> AUROC 曲线</div>
                    </div>
                    <div class="chart-container">
                        <canvas id="aurocChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 智能诊断页面 -->
        <div id="page-predict" class="page">
            <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <div class="upload-text">拖拽X光片到此处，或点击上传</div>
                <div class="upload-hint">支持 JPG / PNG 格式</div>
                <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="predictImage(this.files[0])">
            </div>
            
            <div id="predictResult" style="display:none; margin-top:24px;">
                <div class="result-grid">
                    <div>
                        <div class="card">
                            <div class="card-header">
                                <div class="card-title"><span class="dot"></span> 输入图像</div>
                            </div>
                            <div class="result-image">
                                <img id="previewImage" src="" alt="">
                            </div>
                        </div>
                    </div>
                    <div>
                        <div class="card">
                            <div class="card-header">
                                <div class="card-title"><span class="dot" style="background: var(--accent-pink)"></span> 诊断结果</div>
                                <span id="predictTime" style="font-size:12px; color:var(--text-dim); font-family:'JetBrains Mono',monospace;"></span>
                            </div>
                            <div id="predictionList"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Grad-CAM页面 -->
        <div id="page-gradcam" class="page">
            <div class="upload-zone" id="gradcamUploadZone" onclick="document.getElementById('gradcamFileInput').click()">
                <div class="upload-icon">🔥</div>
                <div class="upload-text">上传X光片生成热力图</div>
                <div class="upload-hint">选择要分析的目标疾病</div>
                <input type="file" id="gradcamFileInput" accept="image/*" style="display:none">
            </div>
            
            <div style="margin-top:16px;">
                <label style="font-size:14px; color:var(--text-secondary);">目标疾病：</label>
                <select id="gradcamLabel" style="margin-left:8px; padding:8px 16px; background:var(--bg-card); color:var(--text-primary); border:1px solid var(--border); border-radius:8px; font-size:14px;">
                    {% for label in labels %}
                    <option value="{{ loop.index0 }}">{{ label }}</option>
                    {% endfor %}
                </select>
            </div>
            
            <div id="gradcamResult" style="display:none; margin-top:24px;">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><span class="dot" style="background: var(--accent-orange)"></span> Grad-CAM 热力图分析</div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px;">
                        <div>
                            <div style="font-size:12px; color:var(--text-dim); margin-bottom:8px; text-align:center;">原始图像</div>
                            <div class="result-image"><img id="gradcamOriginal" src="" alt=""></div>
                        </div>
                        <div>
                            <div style="font-size:12px; color:var(--text-dim); margin-bottom:8px; text-align:center;">注意力热力图</div>
                            <div class="result-image"><img id="gradcamHeatmap" src="" alt=""></div>
                        </div>
                        <div>
                            <div style="font-size:12px; color:var(--text-dim); margin-bottom:8px; text-align:center;">叠加结果</div>
                            <div class="result-image"><img id="gradcamOverlay" src="" alt=""></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script>
        // 页面切换
        function showPage(page) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
            document.getElementById('page-' + page).classList.add('active');
            event.target.classList.add('active');
        }
        
        // 训练数据
        const epochs = {{ epochs | tojson }};
        const trainLoss = {{ train_loss | tojson }};
        const valLoss = {{ val_loss | tojson }};
        const avgAuroc = {{ avg_auroc | tojson }};
        
        // Loss图表
        new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    { label: 'Train Loss', data: trainLoss, borderColor: '#00f5d4', backgroundColor: 'rgba(0,245,212,0.1)', fill: true, tension: 0.4 },
                    { label: 'Val Loss', data: valLoss, borderColor: '#9b5de5', backgroundColor: 'rgba(155,93,229,0.1)', fill: true, tension: 0.4 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#8888aa', font: { family: 'JetBrains Mono' } } } },
                scales: {
                    x: { grid: { color: 'rgba(42,42,90,0.3)' }, ticks: { color: '#555577' } },
                    y: { grid: { color: 'rgba(42,42,90,0.3)' }, ticks: { color: '#555577' } }
                }
            }
        });
        
        // AUROC图表
        new Chart(document.getElementById('aurocChart'), {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    { label: 'Avg AUROC', data: avgAuroc, borderColor: '#f15bb5', backgroundColor: 'rgba(241,91,181,0.1)', fill: true, tension: 0.4 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#8888aa', font: { family: 'JetBrains Mono' } } } },
                scales: {
                    x: { grid: { color: 'rgba(42,42,90,0.3)' }, ticks: { color: '#555577' } },
                    y: { grid: { color: 'rgba(42,42,90,0.3)' }, ticks: { color: '#555577' }, min: 0.5, max: 1.0 }
                }
            }
        });
        
        // 拖拽上传
        ['uploadZone', 'gradcamUploadZone'].forEach(id => {
            const zone = document.getElementById(id);
            if (!zone) return;
            zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
            zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
            zone.addEventListener('drop', e => {
                e.preventDefault();
                zone.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (id === 'uploadZone') predictImage(file);
                else generateGradCAM(file);
            });
        });
        
        // 智能诊断
        async function predictImage(file) {
            if (!file) return;
            const formData = new FormData();
            formData.append('image', file);
            
            document.getElementById('previewImage').src = URL.createObjectURL(file);
            document.getElementById('predictResult').style.display = 'block';
            document.getElementById('predictionList').innerHTML = '<div class="loading"><div class="spinner"></div>分析中...</div>';
            
            const res = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await res.json();
            document.getElementById('predictTime').textContent = data.time || '';
            
            const list = document.getElementById('predictionList');
            list.innerHTML = data.predictions.map(p => `
                <div class="prediction-item ${p.prob > 0.5 ? 'high-risk' : ''}">
                    <div class="prediction-name">${p.label}</div>
                    <div class="prediction-prob" style="color: ${p.prob > 0.5 ? '#f15bb5' : '#00f5d4'}">${(p.prob * 100).toFixed(1)}%</div>
                    <span class="prediction-badge ${p.prob > 0.5 ? 'badge-positive' : 'badge-negative'}">${p.prob > 0.5 ? '阳性' : '阴性'}</span>
                </div>
            `).join('');
        }
        
        // Grad-CAM
        async function generateGradCAM(file) {
            if (!file) return;
            const formData = new FormData();
            formData.append('image', file);
            formData.append('label_idx', document.getElementById('gradcamLabel').value);
            
            document.getElementById('gradcamResult').style.display = 'block';
            document.getElementById('gradcamOriginal').src = URL.createObjectURL(file);
            
            const res = await fetch('/api/gradcam', { method: 'POST', body: formData });
            const data = await res.json();
            
            document.getElementById('gradcamHeatmap').src = data.heatmap;
            document.getElementById('gradcamOverlay').src = data.overlay;
        }
        
        // Grad-CAM文件选择
        document.getElementById('gradcamFileInput').addEventListener('change', function() {
            generateGradCAM(this.files[0]);
        });
    </script>
</body>
</html>
'''


@app.route("/")
def index():
    history = load_training_history()
    
    # 如果没有历史，用默认值
    if not history["epochs"]:
        history = {
            "epochs": list(range(1, 2)),
            "train_loss": [0.5],
            "val_loss": [0.5],
            "avg_auroc": [0.5],
            "per_label": {label: 0.5 for label in (cfg.train_labels if cfg else [])}
        }
    
    # 找最佳checkpoint信息
    best_auroc = max(history["avg_auroc"]) if history["avg_auroc"] else 0.0
    current_epoch = len(history["epochs"])
    
    return render_template_string(
        INDEX_HTML,
        best_auroc=f"{best_auroc:.4f}",
        current_epoch=current_epoch,
        total_epochs=cfg.training.epochs if cfg else 50,
        model_name=cfg.model.name if cfg else "N/A",
        num_labels=len(cfg.train_labels) if cfg else 7,
        epochs=history["epochs"],
        train_loss=history["train_loss"],
        val_loss=history["val_loss"],
        avg_auroc=history["avg_auroc"],
        per_label_auroc=history.get("per_label", {}),
        colors=LABEL_COLORS,
        labels=cfg.train_labels if cfg else [],
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "模型未加载"}), 400
    
    import time
    start = time.time()
    
    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    elapsed = f"{(time.time() - start) * 1000:.0f}ms"
    
    predictions = [
        {"label": label, "prob": float(prob)}
        for label, prob in zip(cfg.train_labels, probs)
    ]
    predictions.sort(key=lambda x: x["prob"], reverse=True)
    
    return jsonify({"predictions": predictions, "time": elapsed})


@app.route("/api/gradcam", methods=["POST"])
def gradcam():
    if not model:
        return jsonify({"error": "模型未加载"}), 400
    
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    import base64, io
    
    file = request.files["image"]
    label_idx = int(request.form.get("label_idx", 0))
    image = Image.open(file.stream).convert("RGB")
    
    # 生成热力图
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 找最后一层卷积
    target_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(label_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    
    # 叠加
    resized = image.resize((cfg.training.image_size, cfg.training.image_size))
    rgb_image = np.array(resized) / 255.0
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    
    # 保存并返回base64
    heatmap_img = Image.fromarray((grayscale_cam * 255).astype(np.uint8)).convert("RGB")
    overlay_img = Image.fromarray((visualization * 255).astype(np.uint8)).convert("RGB")
    
    def to_base64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    
    return jsonify({
        "heatmap": f"data:image/png;base64,{to_base64(heatmap_img)}",
        "overlay": f"data:image/png;base64,{to_base64(overlay_img)}",
    })


def main():
    parser = argparse.ArgumentParser(description="CheXpert AI Dashboard")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型权重路径")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件")
    parser.add_argument("--port", type=int, default=7860, help="端口")
    args = parser.parse_args()
    
    # 加载配置
    global cfg
    cfg = Config.from_yaml(args.config)
    
    # 加载模型
    checkpoint = args.checkpoint or find_best_checkpoint()
    if checkpoint:
        load_model(checkpoint)
    else:
        print("⚠️ 未找到模型权重，推理功能不可用。仪表盘仍可正常显示。")
    
    print(f"\n🚀 CheXpert AI Dashboard")
    print(f"   http://localhost:{args.port}")
    print(f"   模型: {cfg.model.name}")
    print(f"   标签: {cfg.train_labels}")
    print()
    
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
