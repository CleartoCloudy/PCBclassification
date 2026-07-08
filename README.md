# PCB缺陷分类系统 - 基于多种模式识别方法的比较研究

> Author: CleartoCloudy

模式识别课程设计大作业。实现 HOG+SVM、LBP+SVM、ResNet18、MobileNetV3 对 PCB 缺陷进行分类，并用 Faster R-CNN 进行目标检测，完成性能比较分析。

## 项目结构

```
PCB_Classification/
├── config.py                  # 全局配置（路径、超参数）
├── main.py                    # 统一入口脚本
├── requirements.txt           # Python 依赖
├── README.md                  # 本文件
│
├── datasets/                  # 数据集目录
│   ├── DeepPCB_original/      # [需手动准备] 原始 DeepPCB 数据集
│   └── PCB_Classification/    # [自动生成] 处理后的分类数据集
│       ├── train/             #   train_clahe/  train_augmented/
│       ├── val/               #   val_clahe/
│       └── test/              #   test_clahe/
│
├── preprocess/                # 数据预处理模块
│   ├── crop_dataset.py        #   解析标注 → 裁剪缺陷 ROI → 划分 train/val/test
│   ├── clahe.py               #   CLAHE 自适应直方图均衡化增强
│   └── augment.py             #   数据增强（旋转、翻转、亮度、噪声）
│
├── traditional/               # 传统模式识别方法
│   ├── features.py            #   特征提取工具（HOG / LBP）
│   ├── hog_svm.py             #   实验一：HOG + SVM
│   └── lbp_svm.py             #   实验二：LBP + SVM
│
├── deep_learning/             # 深度学习方法
│   ├── dataset.py             #   PyTorch Dataset 与 DataLoader
│   ├── train.py               #   统一训练脚本（支持5种模型）
│   ├── predict.py             #   单图 / 批量推理
│   └── models/
│       ├── attention.py       #   SE 与 CBAM 注意力模块
│       ├── resnet18.py        #   ResNet18（基础/SE/CBAM）
│       └── mobilenetv3.py     #   MobileNetV3 Small
│
├── detection/                 # 目标检测模块
│   ├── dataset.py             #   检测数据集（大图 + bbox 标注）
│   ├── model.py               #   Faster R-CNN (ResNet50-FPN, 小目标锚框优化)
│   ├── train.py               #   训练脚本（mAP@0.5 评估 + 早停）
│   ├── predict.py             #   推理 + 在原图画检测框
│   └── evaluate.py            #   测试集 mAP 评估
│
├── evaluation/                # 评估与可视化
│   ├── evaluate.py            #   统一评估：所有模型测试集指标
│   ├── visualize.py           #   训练曲线、混淆矩阵、对比图表
│   └── report.py              #   实验报告生成器 (Markdown + JSON)
│
├── results/                   # [自动生成] 所有输出
│   ├── models/                #   训练好的模型权重 (.pth / .pkl)
│   ├── figures/               #   对比图表、训练曲线
│   ├── confusion_matrix/      #   各模型混淆矩阵图
│   ├── logs/                  #   训练日志、评估指标 JSON
│   └── reports/               #   实验报告 (Markdown + JSON, 带时间戳)
│
└── report/                    # 课程报告、答辩 PPT 存放目录
```

## 实验设计

| 实验编号 | 方法 | 类型 |
|---------|------|------|
| 实验一 | HOG + SVM | 传统模式识别 Baseline |
| 实验二 | LBP + SVM | 纹理特征分类 Baseline |
| 实验三 | ResNet18 (CLAHE预处理) | 基础深度学习 |
| 实验四 | MobileNetV3 (CLAHE预处理) | 轻量级深度学习 |
| 实验五 | ResNet18 + SE/CBAM 注意力 | 网络结构优化改进 |
| 实验六 | Faster R-CNN (ResNet50-FPN) | 目标检测（大图→缺陷位置+类别） |

## 前置工作：环境准备

### 1. 安装 Python 3.10+

确保 Python 版本 >= 3.10：

```bash
python --version
# 应输出 Python 3.10.x 或更高版本
```

### 2. 创建虚拟环境（强烈推荐）

**Windows (cmd / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意：** PyTorch 安装可能因系统而异。如果 `pip install -r requirements.txt` 安装 PyTorch 失败，请前往 [pytorch.org](https://pytorch.org/get-started/locally/) 根据你的 CUDA 版本（或无 GPU 选 CPU）获取安装命令。例如：
>
> **CUDA 11.8:**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> **CPU only:**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

## 前置工作：准备 DeepPCB 数据集

### 方式一：从 GitHub 下载（推荐）

```bash
# 在项目根目录下执行
cd datasets/DeepPCB_original/

# 克隆 DeepPCB 数据集仓库
git clone https://github.com/tangsanli5201/DeepPCB.git .

cd ../..
```

**克隆完成后无需移动文件**，代码会自动识别 `DeepPCB_original/DeepPCB/PCBData/group*/` 的嵌套结构。

### 方式二：手动下载 ZIP

在浏览器打开 https://github.com/tangsanli5201/DeepPCB ，点击 "Code" → "Download ZIP"。解压后将 **整个 DeepPCB 文件夹**（或内部 group* 文件夹）放入 `datasets/DeepPCB_original/`。

### 方式三：已有数据集

如果你已经有 DeepPCB 的 `group*` 目录，直接放在 `datasets/DeepPCB_original/` 下即可。代码支持以下两种目录结构：

```
# 结构 A（GitHub 克隆后）
datasets/DeepPCB_original/DeepPCB/PCBData/group00041/...

# 结构 B（直接放置）
datasets/DeepPCB_original/group00041/...
```

### 验证数据集

数据集应为如下结构（任一层次均可）：
```
DeepPCB_original/
└── DeepPCB/                   ← 可能有一层 DeepPCB 父目录
    └── PCBData/
        ├── group00041/
        │   ├── 00041/         ← 图像（00041000_test.jpg, 00041000_temp.jpg）
        │   └── 00041_not/     ← 标注文件（00041000.txt, ...）
        ├── group20085/
        │   ├── 20085/
        │   └── 20085_not/
        └── ...
```

标注文件 (`.txt`) 格式（空格分隔）：
```
x1 y1 x2 y2 type
```
其中 `type` 为 **1-6**，对应缺陷类型：open(1), short(2), mousebite(3), spur(4), pinhole(5), spurious_copper(6)

## 运行流程

### 一键运行（推荐）

如果你已经准备好数据集且依赖安装完毕，直接执行：

```bash
python main.py run
```

这会按顺序自动完成：数据预处理 → 训练所有模型（5个实验）→ 评估 → 生成可视化图表 → 生成实验报告。

整个过程在 GPU (RTX3060) 上大约需要 1-2 小时，CPU 上可能需要 4-6 小时。

### 分步运行

如果你只想执行其中某些步骤，可以按以下方式操作。

---

#### 步骤1：数据预处理

```bash
python main.py preprocess
```

这一步会自动完成：
1. **裁剪缺陷区域**：扫描 DeepPCB 标注文件，从有缺陷的图像中裁剪出每个缺陷的 ROI 区域
2. **数据集划分**：按 7:1.5:1.5 比例划分为 train / val / test
3. **CLAHE 增强**：对裁剪后的图像应用自适应直方图均衡化，增强局部对比度
4. **数据增强**：对训练集进行旋转、翻转、亮度变化、高斯噪声等增强

处理后的数据保存在 `datasets/PCB_Classification/` 下：
- `train/` — 原始训练集
- `train_clahe/` — CLAHE 增强后的训练集
- `train_augmented/` — CLAHE + 数据增强后的最终训练集
- `val/` — 原始验证集
- `val_clahe/` — CLAHE 增强后的验证集
- `test/` — 原始测试集
- `test_clahe/` — CLAHE 增强后的测试集

---

#### 步骤2：训练模型

训练所有模型（5个实验）：
```bash
python main.py train all
```

你也可以单独训练某个模型：
```bash
python main.py train hog_svm       # 仅实验一: HOG+SVM
python main.py train lbp_svm       # 仅实验二: LBP+SVM
python main.py train resnet18      # 仅实验三: ResNet18
python main.py train mobilenetv3   # 仅实验四: MobileNetV3
python main.py train resnet18_se   # 仅实验五: ResNet18+SE注意力
python main.py train resnet18_cbam # 仅实验五: ResNet18+CBAM注意力
```

训练后的模型保存在 `results/models/` 目录下。

> **目标检测实验**：训练 Faster R-CNN，对整张 PCB 大图检测所有缺陷的位置和类别：
> ```bash
> python main.py detect train      # 训练检测模型
> python main.py detect evaluate   # 测试集 mAP 评估
> ```

---

#### 步骤3：评估模型并生成报告

```bash
python main.py evaluate
```

在测试集上评估所有已训练好的模型，输出 Accuracy、Precision、Recall、F1-Score 等指标，并**自动生成实验报告**（Markdown + JSON）。

- 评估指标汇总 → `results/logs/all_results_summary.json`
- 实验报告 → `results/reports/experiment_report_YYYYMMDD_HHMMSS.md`
- 报告 JSON → `results/reports/experiment_report_YYYYMMDD_HHMMSS.json`

> 如果只想基于已有评估结果重新生成报告（不重新跑评估），执行：
> ```bash
> python main.py report
> ```

---

#### 步骤4：生成可视化图表

```bash
python main.py visualize
```

生成以下图表：
- 各模型的训练曲线（loss + accuracy）→ `results/figures/`
- 各模型的混淆矩阵 → `results/confusion_matrix/`
- 模型性能对比柱状图 → `results/figures/comparison_*.png`
- 模型效率对比图（参数量 vs 推理时间）→ `results/figures/efficiency_comparison.png`

---

#### 步骤5：查看实验报告

报告在 `results/reports/` 目录下，每次运行生成带时间戳的新文件：

```bash
python main.py report
```

报告包含六大部分：实验环境、数据集统计、模型性能总览（★标注最佳）、各模型详细指标+混淆矩阵、效率对比、结论总结。

---

#### 步骤6：单图预测（可选）

用训练好的模型对单张 PCB 缺陷图像进行分类：

```bash
python main.py predict --model resnet18 --image path/to/your/defect_image.jpg
```

支持的 model 参数：`hog_svm`, `lbp_svm`, `resnet18`, `resnet18_se`, `resnet18_cbam`, `mobilenetv3`

> **检测预测**：对整张 PCB 大图进行缺陷检测（画框 + 标签）：
> ```bash
> python main.py detect predict --image path/to/pcb_image.jpg
> python main.py detect predict --image datasets/DeepPCB_original/DeepPCB/PCBData/group00041/00041/
> ```
> 结果保存在 `results/reports/detection_YYYYMMDD_HHMMSS/`（标注图 + JSON）

---

## 部署到 Ubuntu 云端运行

### 1. 上传项目

将整个 `PCB_Classification` 文件夹上传到 Ubuntu 服务器：

```bash
# 方式一：scp（在本地 Windows 终端执行）
scp -r PCB_Classification/ user@your-server-ip:/home/user/

# 方式二：使用 WinSCP / FileZilla 等工具上传
# 方式三：通过 Git 仓库中转
```

### 2. 服务器环境配置

```bash
# SSH 连接到服务器
ssh user@your-server-ip

# 进入项目目录
cd ~/PCB_Classification

# 安装 Python 依赖（Ubuntu 通常已有 Python3）
sudo apt update
sudo apt install python3-pip python3-venv -y

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备数据集

在服务器上同样需要下载 DeepPCB 数据集：

```bash
cd datasets/DeepPCB_original/
git clone https://github.com/tangsanli5201/DeepPCB.git .
cd ../..
```

### 4. 后台运行（防止SSH断开中断训练）

```bash
# 使用 nohup 后台运行
nohup python main.py run > run.log 2>&1 &

# 查看运行状态
tail -f run.log

# 查看后台任务
jobs -l
```

或使用 `tmux` / `screen`：

```bash
tmux new -s pcb
python main.py run
# 按 Ctrl+B 然后按 D 断开（任务继续运行）
# 重新连接: tmux attach -t pcb
```

---

## 部署到其他 Windows 电脑

### 1. 复制项目文件夹

将整个 `PCB_Classification` 文件夹复制到目标电脑。

### 2. 环境配置

```bash
# 在项目根目录打开终端
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 准备数据集

同上，将 DeepPCB 数据集放入 `datasets/DeepPCB_original/`。

> **注意：** Windows 下 PyTorch DataLoader 的 `num_workers` 会自动设为 0，避免多进程问题。如果手动运行深度学习训练脚本遇到多进程错误，请确保 `num_workers=0`。

---

## 配置修改

所有可调参数集中在 `config.py`，你可以根据需要修改：

```python
# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 图像尺寸
IMG_SIZE_TRAD = 128   # 传统方法
IMG_SIZE_DL = 224     # 深度学习方法

# 数据增强
AUG_PER_SAMPLE = 4    # 每张原始图生成的增强样本数

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# SVM 参数
SVM_C = 10.0
SVM_KERNEL = "rbf"
```

---

## 输出结果说明

训练和评估完成后，查看以下文件：

| 内容 | 位置 |
|------|------|
| 训练好的模型 | `results/models/*.pth`, `*.pkl` |
| 评估指标汇总 | `results/logs/all_results_summary.json` |
| 各模型详细结果 | `results/logs/{model_name}_result.json` |
| 训练曲线图 | `results/figures/*_training_curves.png` |
| 混淆矩阵图 | `results/confusion_matrix/*.png` |
| 模型对比图 | `results/figures/comparison_*.png` |
| 效率对比图 | `results/figures/efficiency_comparison.png` |
| 实验报告 (Markdown) | `results/reports/experiment_report_*.md` |
| 实验报告 (JSON) | `results/reports/experiment_report_*.json` |
| 最新报告快捷副本 | `results/reports/latest_report.md` |
| 检测模型权重 | `results/models/detection_faster_rcnn_best.pth` |
| 检测历史/评估 | `results/logs/detection_history.json`, `detection_result.json` |
| 检测结果标注图 | `results/reports/detection_*/det_*.jpg` |

---

## 实验结果查看与分析指南

训练和评估全部完成后，你需要系统性地查看和分析实验结果，为课程报告提供数据支撑。以下是一套完整的分析流程。

### 第一步：查看评估指标汇总

打开 `results/logs/all_results_summary.json`，这是所有模型在测试集上的最终表现汇总。JSON 结构如下：

```json
{
  "HOG+SVM": {
    "accuracy": 0.xxxx,
    "precision": 0.xxxx,
    "recall": 0.xxxx,
    "f1": 0.xxxx,
    "params_M": 0,
    "inference_time_ms": 0,
    "type": "traditional",
    "confusion_matrix": [[...], ...]
  },
  "LBP+SVM": { ... },
  "ResNet18": { ... },
  "MobileNetV3": { ... },
  "ResNet18+SE": { ... },
  "ResNet18+CBAM": { ... }
}
```

**关注要点**：
- **Accuracy**：整体分类正确率，最直观的指标
- **F1-Score**：精确率和召回率的调和平均，在类别不均衡时比 Accuracy 更可靠
- **Precision / Recall**：分别反映"查得准"和"查得全"的能力

### 第二步：查看单个模型详细结果

每个模型还有独立的详细结果文件：

| 文件 | 内容 |
|------|------|
| `results/logs/hog_svm_result.json` | HOG+SVM 完整评估 |
| `results/logs/lbp_svm_result.json` | LBP+SVM 完整评估 |
| `results/logs/resnet18_result.json` | ResNet18 完整评估 |
| `results/logs/mobilenetv3_result.json` | MobileNetV3 完整评估 |
| `results/logs/resnet18_se_result.json` | ResNet18+SE 完整评估 |
| `results/logs/resnet18_cbam_result.json` | ResNet18+CBAM 完整评估 |

每个文件包含 `train_acc`、`val_acc`、`test_acc`、`train_time_s`、`params_M`、`inference_time_ms`、`history` 等完整信息。

### 第三步：分析混淆矩阵

混淆矩阵图位于 `results/confusion_matrix/` 目录，展示了每个模型对 6 类缺陷的分类详情。

**阅读方法**：
- **对角线**上的数字表示正确分类的样本数，颜色越深、数值越大越好
- **非对角线**上的数字表示混淆的样本，例如 open 被误判为 short 的数量
- 重点关注哪些缺陷类型之间容易混淆（如 mousebite 和 spur 经常互混），这是报告中的重要分析点

### 第四步：分析训练曲线

训练曲线图位于 `results/figures/`，命名格式为 `{模型名}_training_curves.png`。

**阅读方法**：
- **Loss 曲线（左图）**：训练集和验证集的损失下降趋势
  - 两条曲线都在下降且趋于平稳 → 训练正常
  - 训练 Loss 下降但验证 Loss 上升 → **过拟合**
  - 两条曲线都降不下去 → **欠拟合**或学习率不合适
- **Accuracy 曲线（右图）**：训练集和验证集的准确率上升趋势
  - train acc 远高于 val acc → **过拟合**
  - 两条曲线接近且 plateau → 模型已收敛

### 第五步：模型对比分析

`results/figures/` 目录下的对比图：

| 图表文件 | 分析用途 |
|---------|---------|
| `comparison_accuracy.png` | 所有模型准确率柱状图对比 |
| `comparison_f1.png` | 所有模型 F1-Score 柱状图对比 |
| `comprehensive_comparison.png` | 四合一综合对比（Accuracy / Precision / Recall / F1） |
| `efficiency_comparison.png` | 深度学习模型的参数量 vs 推理时间双轴图 |

### 第六步：撰写实验分析（课程报告素材）

分析实验结果是课程报告的核心章节。建议按以下维度展开：

#### 6.1 传统方法 vs 深度学习方法

对比 HOG+SVM、LBP+SVM 与 ResNet18、MobileNetV3 在各指标上的差距：

```
示例分析模板：
"从实验结果可以看出，深度学习方法（ResNet18 准确率 xx.x%、MobileNetV3 准确率 xx.x%）
整体优于传统方法（HOG+SVM 准确率 xx.x%、LBP+SVM 准确率 xx.x%）。
这说明 PCB 缺陷的纹理和形态特征较为复杂，手工设计的 HOG/LBP 特征难以
完整描述缺陷的判别信息，而 CNN 能够自动学习更具区分力的深层特征。"
```

#### 6.2 HOG vs LBP 特征分析

对比两种手工特征的差异：
- HOG 捕获梯度方向信息，对边缘和形状敏感
- LBP 捕获局部纹理模式，对纹理细节敏感
- 分析哪种特征更适合 PCB 缺陷分类

#### 6.3 ResNet18 vs MobileNetV3 对比

从精度和效率两个维度分析：
- 哪个模型准确率更高？
- MobileNetV3 的参数量和推理速度优势有多大？
- 精度损失是否在可接受范围内？适合什么场景？

#### 6.4 注意力机制的效果

对比基础 ResNet18、ResNet18+SE、ResNet18+CBAM：
- SE/CBAM 是否带来了精度提升？提升了多少？
- SE 和 CBAM 哪个效果更好？
- 参数量增加了多少？是否值得？

#### 6.5 错误分析

结合混淆矩阵，重点分析：
- 哪些类别容易混淆？（如 spur 和 spurious_copper）
- 混淆的原因可能是什么？（形态相似、样本不足、图像质量等）
- 如何改进？

### 第七步：关键数据填表模板

课程报告中一般需要以下表格，可直接从 `all_results_summary.json` 提取数据填入：

**表1：模型性能对比表**

| 模型 | Accuracy | Precision | Recall | F1-Score | 参数量 | 推理时间(ms) |
|------|----------|-----------|--------|----------|--------|-------------|
| HOG+SVM | | | | | — | — |
| LBP+SVM | | | | | — | — |
| ResNet18 | | | | | | |
| MobileNetV3 | | | | | | |
| ResNet18+SE | | | | | | |
| ResNet18+CBAM | | | | | | |

**表2：注意力机制消融实验**

| 模型 | Accuracy | vs Baseline |
|------|----------|-------------|
| ResNet18 (Baseline) | | — |
| ResNet18 + SE | | +x.x% |
| ResNet18 + CBAM | | +x.x% |

---

## 常见问题

**Q: 报错 `ImportError: No module named 'torch'`**
A: PyTorch 未正确安装。请根据你的环境（GPU/CPU）重新安装，参考上方"安装依赖"部分。

**Q: 报错 `CUDA out of memory`**
A: 减小 `config.py` 中的 `BATCH_SIZE`（如改为 16 或 8）。

**Q: 报错 `在 DeepPCB_original 中未找到 DeepPCB 数据集文件`**
A: 确认数据集已正确放入 `datasets/DeepPCB_original/`，且其子文件夹中包含 `.txt` 标注文件和 `.jpg` 图像文件。

**Q: 没有 GPU 能运行吗？**
A: 可以，代码会自动检测 CUDA 可用性，无 GPU 时使用 CPU 训练。深度学习模型在 CPU 上训练会慢很多（可能需要数小时）。

**Q: 显存不够，如何只训练部分模型？**
A: 使用 `python main.py train <model_name>` 逐个训练，而不是 `train all`。

**Q: matplotlib 中文显示为方块？**
A: 代码已设置中文字体回退机制。如果仍有问题，在 `evaluation/visualize.py` 中修改 `plt.rcParams["font.sans-serif"]`。

**Q: HOG+SVM 或 LBP+SVM 在 SVM 训练阶段卡住很久？**
A: RBF 核 SVM 在大数据集上非常慢。已默认使用 `LinearSVC`（liblinear），训练大数据集只需数秒。如需切换回 RBF 核，将 `config.py` 中 `SVM_KERNEL` 改为 `"rbf"`（不推荐在数据量 > 10000 时使用）。

**Q: Windows 下训练时 DataLoader 卡住？**
A: 代码在 Windows 下自动设置 `num_workers=0`，如果仍有问题，检查是否有杀毒软件干扰。
