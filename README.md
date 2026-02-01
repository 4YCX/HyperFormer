# HyperFormer: 基于交叉注意力的高光谱与多模态数据融合分类

## 📋 项目概述

HyperFormer 是一个基于深度学习的多模态遥感图像分类框架，**核心创新是提出了 CroSSM（Cross-State Space Model，交叉状态空间模型）架构**，用于高光谱图像（HSI）与 LiDAR/SAR 数据的融合分类。

相比传统的 Transformer 方法，CroSSM 使用 **Mamba 状态空间模型** 替代了自注意力机制，在保持跨模态信息交互能力的同时，实现了线性复杂度 O(N) 的序列建模，显著提升了多模态融合分类的性能。

### 核心架构

| 模型 | 架构特点 | 定位 |
|------|---------|------|
| **CroSSM** | **Mamba + 交叉注意力（主要贡献）** | 主模型，效果更好 |
| JViT (CrossAttn) | Transformer + 交叉注意力 | 对比 Baseline |
| S2ENet | CNN + SAEM/SEEM 模块 | 传统 Baseline |

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
# 安装 PyTorch
pip install torch torchvision

# 安装核心依赖
pip install torchsummary spectral scikit-learn numpy scipy matplotlib seaborn tqdm

# 可选：安装 TensorBoard
pip install tensorboard

# 可选：安装 mamba-ssm（CroSSM 最佳性能，如安装失败会自动回退到简化版）
pip install mamba-ssm
```

### 运行训练（推荐使用 CroSSM）

使用提供的脚本快速训练：

```bash
bash Run.sh
```

或直接运行：

```bash
# 使用 CroSSM（主要贡献模型，效果更好）
python train.py \
    --dataset Houston \
    --model CSSM \
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

---

## ⚙️ 训练参数说明

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | string | 必填 | 数据集名称：Houston/Trento/Augsburg/Berlin/MUUFL |
| `--model` | string | 必填 | 模型名称：**CSSM**（推荐）/ JViT / S2ENet |
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

- **CroSSM / JViT**: AdamW (weight_decay=1e-4)
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
    --model CSSM \
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

## 🏗️ CroSSM 架构详情（主要贡献）

![Architect](CSSM.png)

### 核心创新

CroSSM（Cross-State Space Model）是本项目的主要贡献，其核心创新包括：

1. **Mamba 替代自注意力**
   - 使用 Mamba 状态空间模型替代 Transformer 的自注意力机制
   - 复杂度从 O(N²) 降低到 O(N)，更适合长序列建模
   - 保持全局感受野和长期依赖建模能力

2. **保留交叉注意力机制**
   - 维持双流架构中的双向交叉注意力
   - 实现高光谱与 LiDAR/SAR 模态间的信息交互
   - 可学习的融合门控（Sigmoid gating）

3. **LiDAR 引导的波段门控**
   - 使用辅助模态（LiDAR/SAR）生成门控信号
   - 自适应选择高光谱特征波段
   - 增强跨模态特征对齐

### 网络结构

```
输入: HSI (B, C1, H, W), LiDAR/SAR (B, C2, H, W)
         ↓
    LiDAR-guided Band Gate
         ↓
    Token Embedding (1×1 Conv)
         ↓
    A: (B, N, 128), B: (B, N, 8)
         ↓
    Stage 1: MambaBlock + CrossAttn (双向)
         ↓
    投影 + 位置编码
         ↓
    Stage 2: MambaBlock + CrossAttn (双向)
         ↓
    投影 + 位置编码
         ↓
    Stage 3: MambaBlock + CrossAttn (双向)
         ↓
    FusionLayer (Conv1×1 + BN + ReLU)
         ↓
    AvgPool + FC
         ↓
    输出: (B, n_classes)
```

### 关键组件

1. **MambaBlock**: Mamba 状态空间块
   - LayerNorm → Mamba → 残差连接
   - LayerNorm → FFN → 残差连接
   - 支持真实 Mamba（mamba-ssm）或简化版 fallback

2. **CrossAttnBlock**: 交叉注意力块
   - Q 来自目标模态，K/V 来自源模态
   - 支持不同维度模态间的注意力计算
   - 可学习的融合强度

3. **TwoStreamStage**: 双流处理阶段
   - A 流：Mamba 块处理 HSI
   - B 流：Mamba 块处理 LiDAR/SAR
   - A←B 交叉注意力
   - B←A 交叉注意力

### 相比 JViT 的优势

| 特性 | CroSSM | JViT |
|------|--------|------|
| 序列建模 | Mamba (O(N)) | Self-Attention (O(N²)) |
| 长序列处理 | 更高效 | 显存开销大 |
| 全局感受野 | ✓ | ✓ |
| 交叉注意力 | ✓ | ✓ |
| 分类性能 | **更优** | 良好 |

---

## 📝 使用示例

### 示例1：使用 CroSSM 训练（推荐）

```bash
python train.py \
    --dataset Houston \
    --model CSSM \
    --patch_size 7 \
    --epoch 150 \
    --lr 0.005 \
    --batch_size 256 \
    --cuda 0 \
    --flip_augmentation
```

### 示例2：带数据增强

```bash
python train.py \
    --dataset Trento \
    --model CSSM \
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
    --model CSSM \
    --runs 5 \
    --seed 42 \
    --epoch 150 \
    --batch_size 256 \
    --cuda 0
```

### 示例4：使用 JViT 模型（对比 Baseline）

```bash
python train.py \
    --dataset Augsburg \
    --model JViT \
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
├── train.py           # 主训练脚本（推荐，支持 TensorBoard）
├── visdom_main.py     # 主脚本（支持 Visdom 可视化）
├── model_utils.py     # 模型工厂函数
├── datasets.py        # 数据集加载与处理
├── losses.py          # 损失函数定义
├── utils.py           # 工具函数
├── Run.sh             # 快速运行脚本
│
├── Model/
│   ├── CroSSM.py      # CroSSM 架构（主要贡献）
│   ├── CrossAttn.py   # JViT 架构
│   └── S2ENet.py      # S2ENet Baseline模型
│
├── Datasets/          # 数据集目录
├── checkpoints/       # 模型检查点
├── runs/              # 训练日志
└── Results/           # 结果输出
```

---

## 📦 依赖版本

### 核心依赖

```bash
pip install torch torchvision
pip install torchsummary spectral scikit-learn numpy scipy matplotlib seaborn tqdm
pip install tensorboard  # 用于 train.py
```

### 可选依赖

```bash
pip install visdom       # 用于 visdom_main.py 可视化
pip install mamba-ssm    # 用于 CroSSM 获得最佳性能（如未安装会自动回退到简化版）
```

---

## 🎯 模型选择建议

- **推荐使用 CroSSM（CSSM）**：效果更好，复杂度更低
- **JViT**：作为对比 Baseline，基于传统 Transformer
- **S2ENet**：轻量级 CNN Baseline

---

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系作者。
