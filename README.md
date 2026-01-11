# HyperFormer: 基于交叉注意力的高光谱与多模态数据融合分类

## 📋 项目概述

HyperFormer 是一个基于深度学习的多模态遥感图像分类框架，基于 **CrossAttn（交叉注意力）** 架构，用于高光谱图像（HSI）与LiDAR/SAR数据的融合分类。

### 核心架构：CrossAttn (JViT)

BASIC ARCHITECTURE

1. **双流Transformer结构**：分别处理高光谱和辅助模态数据
2. **双向交叉注意力机制**：实现模态间的信息交互
3. **三阶段处理**：从粗到细的特征提取和融合
4. **可学习位置编码**：保留空间位置信息

### 模型对比

| 模型 | 架构特点 | 适用场景 |
|------|---------|---------|
| **JViT (CrossAttn)** | Transformer + CrossAttention | 多模态融合分类 |
| **S2ENet** | CNN + SAEM/SEEM模块 | 传统多模态融合 |

---

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 1.9.0
torchsummary
spectral
scikit-learn
numpy
scipy
matplotlib
seaborn
tqdm
```

### 安装依赖

```bash
pip install torch torchvision
pip install torchsummary spectral scikit-learn numpy scipy matplotlib seaborn tqdm
```

### 运行训练

使用提供的脚本快速训练：

```bash
# 使用默认配置训练 Berlin 数据集
bash Run.sh
```

或直接运行：

```python
python train.py \
    --dataset Berlin \
    --model JViT \
    --patch_size 7 \
    --epoch 150 \
    --lr 5e-3 \
    --batch_size 256 \
    --cuda 0 \
    --flip_augmentation
```

---

## 📁 数据准备

### 数据集文件夹结构

将数据集放在 `./Datasets/` 目录下：

```
Datasets/
├── Houston/
│   ├── HSI.mat          # 高光谱数据
│   ├── LiDAR.mat        # LiDAR数据
│   └── gt.mat           # 标签数据
│
├── Trento/
│   ├── HSI_Trento.mat
│   ├── Lidar_Trento.mat
│   └── GT_Trento.mat
│
├── Augsburg/
│   ├── data_HS_LR.mat   # 高光谱数据
│   ├── data_SAR_HR.mat  # SAR数据
│   ├── TrainImage.mat   # 训练标签
│   └── TestImage.mat    # 测试标签
│
├── Berlin/
│   ├── data_HS_LR.mat
│   ├── data_SAR_HR.mat
│   ├── TrainImage.mat
│   └── TestImage.mat
│
└── MUUFL/
    └── (MUUFL数据集文件)
```

### 数据格式说明

- **HSI/LiDAR/SAR数据**: `.mat` 文件，包含 `data` 或对应的键名
- **标签数据**: `.mat` 文件，包含 `gt`、`TRLabel`、`TSLabel` 等键名
- **自动归一化**: 程序会自动将数据归一化到 [0, 1] 范围

### 自定义数据集

如需添加自定义数据集，请在 `datasets.py` 的 `DATASETS_CONFIG` 字典中添加配置：

```python
DATASETS_CONFIG = {
    "YourDataset": {
        "urls": [],  # 下载链接（可选）
        "folder": "YourDataset/",  # 数据夹名称
    }
}
```

---

## ⚙️ 训练参数说明

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | string | 必填 | 数据集名称：Houston/Trento/Augsburg/Berlin/MUUFL |
| `--model` | string | 必填 | 模型名称：JViT/S2ENet |
| `--cuda` | int | 1 | CUDA设备索引（-1表示使用CPU） |
| `--runs` | int | 1 | 运行次数（用于多次实验取平均） |
| `--seed` | int | 0 | 随机种子（控制实验可重复性） |

### 数据集参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--folder` | string | "./Datasets/" | 数据集根目录 |
| `--train_set` | string | None | 训练标签文件路径（.mat格式） |
| `--test_set` | string | None | 测试标签文件路径（.mat格式） |
| `--train_val_split` | float | 0.8 | 训练集内部验证集划分比例 |
| `--training_sample` | float | 0.99 | 从标注点中采样的训练比例 |
| `--sampling_mode` | string | "random" | 采样模式：random/fixed/disjoint |

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--patch_size` | int | 7 | 空间邻域大小（奇数） |
| `--n_classes` | int | 自动 | 分类数量（从数据集自动获取） |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epoch` | int | 128 | 训练轮数 |
| `--lr` | float | 0.001 | 初始学习率 |
| `--batch_size` | int | 128 | 批次大小 |
| `--class_balancing` | flag | False | 是否启用类别平衡（逆中频加权） |
| `--test_stride` | int | 1 | 测试时滑窗步长 |

### 学习率调度

默认使用 `MultiStepLR`：

```python
milestones = [90, 150, 180]
gamma = 0.1
```

即在第90、150、180轮时学习率乘以0.1。

### 优化器

- **JViT**: AdamW (weight_decay=1e-4)
- **S2ENet**: Adam

### 损失函数

默认使用 **CrossEntropyLoss**，支持类别平衡权重。

---

## 🔧 数据增强

| 参数 | 说明 |
|------|------|
| `--flip_augmentation` | 随机翻转增强（水平+垂直） |
| `--radiation_augmentation` | 辐射噪声增强（10%概率） |
| `--mixture_augmentation` | 混合增强（20%概率） |

示例：

```bash
python train.py \
    --dataset Berlin \
    --model JViT \
    --flip_augmentation \
    --radiation_augmentation \
    --mixture_augmentation
```

---

## 📊 评估指标

程序会自动计算并记录以下指标：

| 指标 | 说明 |
|------|------|
| **OA (Overall Accuracy)** | 总体准确率 |
| **AA (Average Accuracy)** | 平均准确率（各类召回率的均值） |
| **Kappa** | Kappa系数 |
| **Per-class Accuracy** | 各类别准确率 |
| **Loss** | 训练/验证损失 |

### 输出文件

训练日志保存在 `runs/` 目录下：

```
runs/
├── {dataset}_{model}_seed{seed}/
│   └── {timestamp}/
│       ├── metrics_epoch.csv      # 每轮指标
│       ├── per_class_epoch.csv    # 逐类指标
│       └── events.out.tfevents.*  # TensorBoard日志
```

### TensorBoard 可视化

```bash
tensorboard --logdir runs --port 6006
```

然后访问 http://localhost:6006 查看训练曲线。

---

## 🏗️ 模型架构详情

### CrossAttn (JViT) 结构

![Architect](1.png)

### 关键组件

1. **SelfAttnBlock**: 标准Transformer编码器块
   - LayerNorm → MultiheadAttention → Dropout → FFN

2. **CrossAttnBlock**: 交叉注意力块
   - Q来自目标模态，K,V来自源模态
   - 支持不同维度模态间的注意力计算

3. **TwoStreamStage**: 双流处理阶段
   - A流自注意力
   - B流自注意力
   - A←B 交叉注意力
   - B←A 交叉注意力

---

## 📝 使用示例

### 示例1：基础训练

```bash
python train.py \
    --dataset Houston \
    --model JViT \
    --patch_size 7 \
    --epoch 150 \
    --lr 0.005 \
    --batch_size 256 \
    --cuda 0
```

### 示例2：带数据增强

```bash
python train.py \
    --dataset Trento \
    --model JViT \
    --patch_size 9 \
    --epoch 200 \
    --lr 0.001 \
    --batch_size 128 \
    --cuda 0 \
    --flip_augmentation \
    --radiation_augmentation
```

### 示例3：多次运行取平均

```bash
python train.py \
    --dataset Berlin \
    --model JViT \
    --runs 5 \
    --seed 42 \
    --epoch 150 \
    --batch_size 256 \
    --cuda 0
```

### 示例4：使用S2ENet模型

```bash
python train.py \
    --dataset Augsburg \
    --model S2ENet \
    --patch_size 7 \
    --epoch 128 \
    --lr 0.001 \
    --batch_size 64 \
    --cuda 0
```

---

## 🔍 文件结构

```
HyperFormer/
├── train.py           # 主训练脚本
├── model_utils.py     # 模型工厂函数
├── datasets.py        # 数据集加载与处理
├── losses.py          # 损失函数定义
├── utils.py           # 工具函数
├── Run.sh             # 快速运行脚本
│
└── Model/
    ├── CrossAttn.py   # CrossAttn (JViT) 架构
    └── S2ENet.py      # S2ENet 基线模型
```

---

## 📦 依赖版本

```
torch >= 1.9.0
torchsummary
spectral
scikit-learn
numpy
scipy
matplotlib
seaborn
tqdm
```

---

## 📄 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{HyperFormer,
  title={HyperFormer: Cross-Attention based Multi-modal Fusion for Hyperspectral Classification},
  author={ChangYi,Xiao;ChengYu,Yang},
  year={2026}
}
```

---

## 📧 联系方式

如有问题或建议，请提交Issue或联系作者。
