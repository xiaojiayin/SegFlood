# MA-XAttn：面向洪水与地表水映射的光学–SAR统一基准与融合框架

> **English documentation**: [README.md](README.md)

SegFlood 是一个面向**多模态洪水（或地表水）语义分割**的研究代码库，以统一的 Hydra + PyTorch Lightning 流水线为骨架。它支持：

- **多模态输入**：光学影像（如 RGB+NIR）与 SAR（VV/VH 极化）的双流处理
- **单模态基线**：仅光学或仅 SAR 模式
- **轻量级融合系统**：支持 `concat`（通道拼接）、`add`（逐元素相加）、`xattn`（MA-XAttn 跨模态注意力）三种策略
- **完整的训练、评估与推理工具**：逐块预测 + 可选马赛克（GeoTIFF 拼接）导出

---

## 目录

1. [项目背景](#1-项目背景)
2. [仓库目录结构](#2-仓库目录结构)
3. [数据集说明](#3-数据集说明)
4. [模型架构](#4-模型架构)
   - 4.1 [编码器（Encoder）](#41-编码器encoder)
   - 4.2 [融合模块（Fusion）](#42-融合模块fusion)
   - 4.3 [解码器（Decoder）](#43-解码器decoder)
5. [对齐正则化（MA-XAttn Alignment）](#5-对齐正则化ma-xattn-alignment)
6. [配置系统（Hydra）](#6-配置系统hydra)
7. [环境安装](#7-环境安装)
8. [快速启动](#8-快速启动)
   - 8.1 [训练](#81-训练)
   - 8.2 [评估](#82-评估)
   - 8.3 [推理](#83-推理)
9. [实验配置详解](#9-实验配置详解)
10. [指标体系](#10-指标体系)
11. [致谢](#11-致谢)

---

## 1 项目背景

遥感洪水制图面临两个核心挑战：

1. **多模态差异**：光学卫星（如 Sentinel-2、GF-2）受云雾遮挡影响大，而 SAR（如 Sentinel-1、GF-3）可全天候成像，两者特征空间差距显著。
2. **跨数据集泛化**：不同数据集的传感器、空间分辨率、标注语义各不相同，难以用单一模型统一处理。

本项目提出 **MA-XAttn**（Multi-scale Attention Cross-Attention）融合框架，在特征金字塔的每个尺度上对光学与 SAR 特征进行双向跨模态注意力，同时引入 **SigLIP 风格的 PatchNCE 对齐正则化**，鼓励两个模态的特征在共享投影空间中语义对齐，从而提升融合质量。

---

## 2 仓库目录结构

```text
SegFlood/
├── configs/                   # Hydra 配置根目录
│   ├── train.yaml             # 训练入口配置（默认组合）
│   ├── eval.yaml              # 评估入口配置
│   ├── data/                  # 各数据集的 DataModule 配置
│   │   ├── gf_floodnet.yaml
│   │   ├── cau_flood.yaml
│   │   ├── kurosiwo.yaml
│   │   ├── worldfloodsv2.yaml
│   │   └── s1s2_water.yaml
│   ├── model/                 # 模型骨架配置（编码器+融合+解码器）
│   │   ├── resnet50_resnet50.yaml
│   │   ├── dinov3_dinov3.yaml
│   │   ├── efficientnetb4_mobilenetv3.yaml
│   │   └── sam2_sam2.yaml
│   ├── experiment/            # 完整实验覆盖配置（数据+模型+训练超参）
│   ├── trainer/               # Lightning Trainer 配置
│   ├── callbacks/             # 回调函数配置（ModelCheckpoint 等）
│   ├── logger/                # 日志配置（TensorBoard 等）
│   ├── paths/                 # 路径配置（PROJECT_ROOT 等）
│   ├── extras/                # 辅助配置（打印树、询问标签等）
│   └── hydra/                 # Hydra 输出目录规则
│
├── src/                       # 核心源代码
│   ├── train.py               # 训练主入口（Hydra 驱动）
│   ├── eval.py                # 评估主入口（Hydra 驱动）
│   ├── data/
│   │   ├── datasets/          # torch.utils.data.Dataset 实现（每个数据集一个文件）
│   │   └── datamodules/       # Lightning DataModule 实现（每个数据集一个文件）
│   ├── models/
│   │   ├── lightning_module.py   # 核心 LightningModule（MultiModalSegmentationModule）
│   │   ├── encoders/             # 编码器（双流 timm backbone）
│   │   ├── fusion/               # 融合模块（concat / add / MA-XAttn）
│   │   └── decoders/             # 解码器（动态 U-Net）
│   ├── infer/                 # 推理写出器（每个数据集一个 PredictWriter）
│   └── utils/                 # 日志、实例化工具等
│
├── scripts/
│   ├── run/                   # SLURM 训练脚本（sbatch）
│   ├── infer/                 # SLURM 推理脚本 + 薄 Python 入口
│   └── data_pre/              # 数据预处理脚本
│
├── assets/                    # README 所用图片
├── requirements.txt           # Python 依赖
└── .project-root              # rootutils 项目根标识文件
```

---

## 3 数据集说明

所有数据集均转换为**基于图像块（patch）的样本**进行训练与评估。

| 数据集          | 传感器模态                             | 训练集   | 验证集  | 测试集  | 任务                             |
| --------------- | -------------------------------------- | -------: | ------: | ------: | -------------------------------- |
| **GF-FloodNet** | GF-3 SAR（VV）+ GF-2 光学             |  9,372   |  2,678  |  1,338  | SAR–光学洪水分割                 |
| **CAU-Flood**   | S2 光学（事前）+ S1 SAR（事后）        | 13,328   |  1,903  |  3,071  | 光学–SAR 洪水变化检测            |
| **Kuro Siwo**   | S1 GRD SAR + DEM                       | 21,273   |  5,031  | 13,630  | 纯 SAR 水体映射                  |
| **WorldFloods v2** | S2 L1C 光学（BGRI）               | 65,582   |  3,524  |  4,703  | 纯光学洪水与地表水分割           |
| **S1S2-Water**  | 配准的 S1 SAR + S2 光学（+ DEM）       | 61,017   | 29,584  | 29,584  | 全球永久水体分割                 |

### 标签语义（重要！勿混淆）

- **CAU-Flood**：变化检测任务，标签 `1` 表示**新增洪水像素**（永久水体为背景 `0`）。
- **GF-FloodNet**：分割任务，标签 `1` 表示事后影像中的**所有水体**（包括永久水体）。
- 所有数据集均使用 `ignore_index = -1` 标记无效/忽略像素。

### 数据集根目录配置

通过 `data.root=...` 参数指定各数据集的本地路径，例如：

```bash
python src/train.py experiment=resnet50_resnet50_gffloodnet data.root=/path/to/GFFloodNet
```

或直接修改 `configs/data/<dataset>.yaml` 中的 `root` 字段。

---

## 4 模型架构

整体流水线为：**双流编码器 → 特征融合 → U-Net 解码器 → 分割头**。

```
光学输入  ──►  光学编码器  ──►  [C1, C2, C3, C4]  ─┐
                                                      ├──►  FeatureFusion  ──►  [F1, F2, F3, F4]  ──►  UNetDecoder  ──►  logits
SAR 输入  ──►  SAR 编码器  ──►  [C1, C2, C3, C4]  ─┘
```

所有组件均通过 Hydra 配置字典（含 `_target_` 字段）动态实例化，使检查点（checkpoint）完全自包含——加载检查点时无需额外传入构造参数。

### 4.1 编码器（Encoder）

**源文件**：`src/models/encoders/encoders.py`

核心类为 `DualStreamEncoder`，它为光学与 SAR 各创建一个独立的 timm backbone（`features_only=True` 模式，输出特征金字塔）：

```python
# 示例：双流 ResNet-50 编码器
encoder:
  _target_: src.models.encoders.DualStreamEncoder
  model_name: "resnet50"      # 两路共用同一骨干名，也可分别指定 optical_model_name / sar_model_name
  optical_channels: 3         # 光学输入波段数
  sar_channels: 1             # SAR 输入波段数
  optical_pretrained: true    # 光学骨干使用 ImageNet 预训练权重
  sar_pretrained: false       # SAR 骨干随机初始化（通道数不匹配 ImageNet）
```

- **特征金字塔对齐**：通过 `feature_info`（timm API）自动获取各层的通道数与空间降采样倍数，确保两路特征图在融合前尺寸一致。
- **单模态模式**：当 `sar_channels=0` 或 `optical_channels=0` 时，自动退化为单流编码器。
- **支持的骨干**：所有 timm 支持 `features_only=True` 的模型，如 ResNet、EfficientNet、MobileNetV3、DINOv3（ViT-based）、SAM2 等。

### 4.2 融合模块（Fusion）

**源文件**：`src/models/fusion/fusion.py`、`src/models/fusion/strategies/`

融合模块采用**策略模式**（Strategy Pattern），通过 `fusion_type` 参数切换具体融合策略：

#### (a) Concat（通道拼接）

```yaml
fusion:
  _target_: src.models.fusion.FeatureFusion
  fusion_type: "concat"
```

在每个特征金字塔层上，将光学与 SAR 特征**沿通道维度拼接**，然后通过 1×1 卷积降维至统一宽度。

#### (b) Add（逐元素相加）

```yaml
fusion:
  fusion_type: "add"
  projection_norm: "gn"   # 投影归一化：gn（GroupNorm）或 bn（BatchNorm）
```

先将两路特征通过独立投影头映射到相同宽度，再逐元素相加。

#### (c) MA-XAttn（多尺度跨模态注意力，本文核心方法）

```yaml
fusion:
  fusion_type: "xattn"
  xattn_apply_levels: [-1, -2]   # 在倒数第 1、2 个特征层应用 XAttn，其余层用 Add
  xattn_reduction: 8             # 注意力通道压缩比
  xattn_gamma_init: 0.0          # 注意力残差权重初始化（0=恒等，逐渐学习）
  xattn_align_bias: true         # 启用对齐偏置引导
  projection_norm: "gn"
```

MA-XAttn 在指定的深层特征层执行**双向跨模态注意力**：

1. **投影**：光学与 SAR 特征分别经 `DWResidualProjection`（深度可分离残差投影）映射到共享维度。
2. **空间注意力（SpatialAttBlock）**：以一路特征为 Query，另一路为 Key/Value，计算空间位置上的注意力权重；支持通过 `align_bias` 将对齐损失的梯度信号作为额外引导。
3. **通道注意力（ChannelAttBlock）**：在通道维度上对融合后的特征进行重标定。
4. **残差连接**：融合增量通过可学习的 γ 参数缩放后，加回到原始拼接特征的投影。

对于**未应用 XAttn 的特征层**，退化为 Add 策略（投影后相加）。

#### 单模态前向路径

当输入只有一个模态时（`optical_features=None` 或 `sar_features=None`），融合模块自动切换到单流投影路径，输出维度与双模态一致，从而支持单模态/双模态的统一训练接口。

### 4.3 解码器（Decoder）

**源文件**：`src/models/decoders/decoders.py`

使用**动态 U-Net 解码器**（`UNetDecoder`），自动适应编码器输出的任意特征金字塔深度：

```
F4（最深）──► up ──► 拼接 F3 ──► conv_block ──► up ──► 拼接 F2 ──► ... ──► 分割头 ──► logits
```

- **上采样方式**：支持双线性插值（`up_mode="bilinear"`，推荐）和转置卷积（`up_mode="convtrans"`，固定 2×）。
- **分割头**（`SegmentationHead`）：Conv→BN→ReLU→Dropout→1×1 Conv，输出 `(B, num_classes, H, W)` 的 logits。
- 解码器最终将 logits 双线性插值回输入分辨率（`final_upsample_to_input=True`）。

---

## 5 对齐正则化（MA-XAttn Alignment）

**源文件**：`src/models/lightning_module.py`（`_compute_alignment_loss`、`_patch_siglip_loss`）

为进一步拉近光学与 SAR 特征的语义距离，本项目引入 **SigLIP 风格的 PatchNCE 对齐正则化**，仅在**训练阶段**、仅在**双模态输入**时计算。

### 实现原理

1. **特征采样**：在选定的特征层（如最后一层）上，随机采样 K 个空间位置（默认 K=256）。
2. **投影**：复用 MA-XAttn 的共享投影头，将光学/SAR 特征映射到对齐空间，并做 L2 归一化。
3. **相似度矩阵**：计算 `[n×K, n×K]` 的余弦相似度矩阵，再除以温度系数 τ。
4. **正样本掩码**：对角块（同一图像、同一位置）以及切比雪夫距离 ≤ r 的空间邻域都视为正样本（PatchNCE 局部邻域）；其余为负样本。
5. **SigLIP 损失**：以 `BCEWithLogitsLoss`（带正负样本权重平衡）替代传统 softmax-NCE，实现多正例建模。
6. **损失上限（Cap）**：通过 EMA 追踪主任务损失，将对齐损失上限钳制为主损失的 `align_cap_ratio` 倍（默认 0.3），防止对齐项主导训练。

### 配置示例

```yaml
model:
  alignment_enabled: true
  alignment_target_weight: 0.05     # 对齐损失基础权重 λ
  alignment_layers: [-1]            # 在倒数第 1 层计算
  alignment_temperature: 0.08       # 温度系数 τ
  alignment_num_samples: 256        # 每张图采样空间位置数 K
  alignment_patch_radius: 1         # 正样本邻域半径 r（切比雪夫距离）
  align_cap_ratio: 0.3              # 对齐损失上限 = cap_ratio × EMA(主损失)
```

---

## 6 配置系统（Hydra）

本项目使用 [Hydra](https://hydra.cc/) 管理所有超参数。配置以**组合（Composition）**方式组织：

```
configs/train.yaml
  ├── data: gf_floodnet          # 选用的数据集配置
  ├── model: resnet50_resnet50   # 选用的模型配置
  ├── callbacks: default         # 回调函数
  ├── logger: tensorboard        # 日志
  ├── trainer: default           # Lightning Trainer
  ├── paths: default             # 路径（PROJECT_ROOT）
  ├── extras: default            # 辅助功能
  ├── hydra: default             # Hydra 输出目录规则
  └── experiment: null           # 实验覆盖（可选）
```

**实验配置**（`configs/experiment/*.yaml`）通过 `# @package _global_` 声明可覆盖任意顶层字段，是最常用的实验管理方式：

```yaml
# configs/experiment/resnet50_resnet50_gffloodnet.yaml
defaults:
  - override /data: gf_floodnet
  - override /model: resnet50_resnet50
  - override /trainer: gpu

model:
  fusion:
    fusion_type: "xattn"
    xattn_apply_levels: [-1, -2]
  alignment_enabled: true
```

**Hydra 插值**：在 YAML 中可使用 `${data.optical_channels}` 等插值表达式，确保模型通道数与数据集自动匹配，无需手动同步。

---

## 7 环境安装

```bash
# 创建并激活 conda 环境（Python 3.11）
conda create -n segflood python=3.11 -y
conda activate segflood

# 升级基础工具
pip install -U pip setuptools wheel

# 安装所有依赖
pip install -r requirements.txt
```

> **HPC 注意事项**：在集群上建议通过 `conda-forge` 渠道安装 `rasterio` / `gdal`，以避免与系统 GDAL 冲突：
> ```bash
> conda install -c conda-forge rasterio gdal -y
> ```

**主要依赖说明**：

| 包                    | 用途                                      |
| --------------------- | ----------------------------------------- |
| `torch` / `torchvision` | 深度学习框架                            |
| `lightning`           | PyTorch Lightning 训练流水线              |
| `hydra-core`          | 配置管理系统                              |
| `timm`                | 预训练骨干网络（ResNet、ViT 等）          |
| `rasterio`            | GeoTIFF 读写与地理信息处理                |
| `segmentation-models-pytorch` | FocalLoss、DiceLoss 等损失函数  |
| `torchmetrics`        | IoU、F1、精度等评估指标                   |
| `compress-pickle`     | Kuro Siwo 数据集的压缩 pickle 索引        |

---

## 8 快速启动

### 前置条件：设置 PROJECT_ROOT

Hydra 路径解析依赖 `PROJECT_ROOT` 环境变量：

```bash
export PROJECT_ROOT="$(pwd)"   # 在仓库根目录执行
```

或直接将其写入 shell 配置文件（`.bashrc` / `.zshrc`）。

---

### 8.1 训练

#### 本地运行（验证配置是否正确，跳过实际训练）

```bash
python src/train.py train=False test=False
```

#### 运行指定实验（本地 GPU）

```bash
python src/train.py experiment=resnet50_resnet50_gffloodnet
```

#### 覆盖单个参数

```bash
# 修改批量大小和学习率
python src/train.py experiment=resnet50_resnet50_gffloodnet \
    data.batch_size=64 \
    model.learning_rate=1e-3
```

#### 通过 SLURM 提交（HPC 集群）

```bash
# 提交单个实验
sbatch scripts/run/train_experiment.sh resnet50_resnet50_gffloodnet

# 批量提交所有实验
bash scripts/run/submit_batch_experiments.sh
```

> `scripts/run/train_experiment.sh` 会根据实验名称自动推断数据集 key（`cauflood | gffloodnet | s1s2water | kurosiwo | worldfloodsv2`），并设置对应的 `data.root`。

---

### 8.2 评估

使用已保存的检查点在测试集上进行评估：

```bash
python src/eval.py \
    experiment=resnet50_resnet50_gffloodnet \
    ckpt_path=/path/to/checkpoint.ckpt
```

---

### 8.3 推理

推理脚本位于 `scripts/infer/`，每个数据集有独立的入口：

```bash
# GF-FloodNet 推理（逐块预测 + GeoTIFF 输出 + 马赛克拼接）
python scripts/infer/infer_gffloodnet.py \
    --checkpoint /path/to/ckpt.ckpt \
    --dataset-path /path/to/GFFloodNet \
    --output-dir outputs/infer/gffloodnet \
    --save-predictions \
    --save-format tif

# CAU-Flood 推理
python scripts/infer/infer_cauflood.py \
    --checkpoint /path/to/ckpt.ckpt \
    --dataset-path /path/to/CAUFlood \
    --output-dir outputs/infer/cauflood

# Kuro Siwo（滑窗推理，全场景 mosaic）
python scripts/infer/infer_kurosiwo.py \
    --checkpoint /path/to/ckpt.ckpt \
    --root /path/to/KuroSiwoGRD \
    --out_dir outputs/infer/kurosiwo \
    --tile_size 256 --overlap 64

# S1S2-Water 推理
python scripts/infer/infer_s1s2water.py \
    --checkpoint /path/to/ckpt.ckpt \
    --data-root /path/to/S1S2Water \
    --out-dir outputs/infer/s1s2water

# WorldFloodsv2 推理
python scripts/infer/infer_worldfloodsv2.py \
    --checkpoint /path/to/ckpt.ckpt \
    --dataset-path /path/to/WorldFloodsV2 \
    --output-dir outputs/infer/worldfloodsv2
```

#### 推理输出

| 文件 / 目录              | 说明                                          |
| ------------------------ | --------------------------------------------- |
| `predictions/`           | 逐块预测掩码（GeoTIFF 或 PNG，含颜色表）      |
| `mosaics/`               | 事件级全景马赛克（部分数据集支持）            |
| `overall_metrics.xlsx`   | 全测试集整体指标汇总                          |
| `detailed_samples.xlsx`  | 每个样本的逐块指标明细                        |

---

## 9 实验配置详解

所有实验配置位于 `configs/experiment/`，命名规则为：

```
<光学骨干>_<SAR骨干 或 early_fusion>_<数据集>.yaml
```

例如：

| 实验名                              | 光学骨干         | SAR 骨干       | 数据集          | 说明                   |
| ----------------------------------- | ---------------- | -------------- | --------------- | ---------------------- |
| `resnet50_resnet50_gffloodnet`      | ResNet-50        | ResNet-50      | GF-FloodNet     | 双流 + MA-XAttn        |
| `resnet50_early_fusion_gffloodnet`  | ResNet-50        | 无（早期融合） | GF-FloodNet     | 早期融合（单流）       |
| `resnet50_optical_gffloodnet`       | ResNet-50        | 无             | GF-FloodNet     | 纯光学单模态           |
| `resnet50_sar_gffloodnet`           | 无               | ResNet-50      | GF-FloodNet     | 纯 SAR 单模态          |
| `dinov3_dinov3_cauflood`            | DINOv3（ViT）    | DINOv3（ViT）  | CAU-Flood       | 双流 ViT + MA-XAttn    |
| `efficientnetb4_mobilenetv3_s1s2water` | EfficientNet-B4 | MobileNetV3  | S1S2-Water      | 双流轻量级骨干         |
| `sam2_sam2_gffloodnet`              | SAM2             | SAM2           | GF-FloodNet     | 双流 SAM2              |
| `resnet50_kurosiwo`                 | —                | ResNet-50      | Kuro Siwo       | 纯 SAR（单流）         |
| `resnet50_worldfloodsv2`            | ResNet-50        | —              | WorldFloods v2  | 纯光学（单流）         |

### 关键可调超参数

```yaml
model:
  # 编码器
  encoder:
    drop_path_rate: 0.1        # 随机深度（Stochastic Depth）正则化

  # 融合
  fusion:
    fusion_type: "xattn"       # concat | add | xattn
    xattn_apply_levels: [-1, -2]
    xattn_reduction: 8         # 注意力降维比
    xattn_gamma_init: 0.0      # 初始残差权重

  # 解码器
  decoder:
    dropout: 0.1               # Dropout 率
    up_mode: "bilinear"        # 上采样方式

  # 主损失
  loss_fn:
    _target_: segmentation_models_pytorch.losses.FocalLoss
    alpha: [0.15, 0.85]        # 类别权重（0=背景, 1=水体）
    gamma: 2.0                 # 难样本聚焦系数

  # 辅助损失（可选）
  aux_loss_fn:
    _target_: segmentation_models_pytorch.losses.DiceLoss
  aux_loss_weight: 0.3

  # 优化器
  learning_rate: 5e-4
  weight_decay: 1e-4

  # 对齐正则化（可选，仅双模态 + xattn 时有效）
  alignment_enabled: true
  alignment_target_weight: 0.05
```

---

## 10 指标体系

训练过程中记录的指标（通过 TensorBoard 查看）：

| 指标键                 | 说明                                                     |
| ---------------------- | -------------------------------------------------------- |
| `train/loss_main`      | 主损失（FocalLoss 或 CrossEntropyLoss）                  |
| `train/loss_aux`       | 辅助损失（DiceLoss，仅训练阶段）                         |
| `train/loss_alignment` | 对齐正则化损失（仅双模态 + 启用时）                     |
| `val/iou`              | 验证集宏平均 mIoU（`JaccardIndex`）                      |
| `val/water_iou`        | 水体类 IoU                                               |
| `val/f1`               | 宏平均 F1 分数                                           |
| `val/recall`           | 逐类召回率（包含洪水类与背景类）                         |
| `val/specificity`      | 逐类特异性                                               |
| `test/iou`             | 测试集 mIoU（训练结束后用最佳检查点计算）                |

推理阶段的指标输出在 `overall_metrics.xlsx` 中，包括：
`iou`、`water_iou`、`bg_iou`、`f1`、`accuracy`、`precision`、`recall`、`specificity`、`bg_false_alarm_rate`（背景误报率）、`flood_miss_rate`（洪水漏检率）。

---

## 11 致谢

- 本仓库的 Backbone 封装参考并重构自 [timm](https://github.com/huggingface/pytorch-image-models/) 项目。
- 训练流水线结构与 Hydra 配置模式参考自 [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/)。
