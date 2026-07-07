# 基于多种模式识别方法的PCB缺陷分类系统设计与实现——传统特征与深度学习模型的比较研究
## 一、选题背景、意义
印刷电路板（Printed Circuit Board，PCB）是电子产品的重要组成部分，其制造质量直接影响电子设备的可靠性和稳定性。在PCB生产过程中，由于材料、加工工艺以及设备等因素影响，容易产生开路（Open）、短路（Short）、毛刺（Spur）、鼠咬（Mouse Bite）、针孔（Pinhole）以及多余铜（Spurious Copper）等多种缺陷。如果这些缺陷不能及时发现，不仅会影响产品性能，还可能造成较大的经济损失。因此，实现PCB缺陷的自动识别与分类，对提高工业检测效率和产品质量具有重要意义。

目前，PCB缺陷识别方法主要包括传统模式识别方法和基于深度学习的方法。传统方法通常采用人工设计特征，如方向梯度直方图（HOG）、局部二值模式（LBP）等，再结合支持向量机（SVM）等分类器完成识别。这类方法具有模型简单、计算速度快、可解释性强等优点，但对复杂纹理、小尺寸缺陷及光照变化的适应能力有限。

近年来，卷积神经网络（CNN）在图像分类领域取得了显著成果。ResNet、MobileNet等深度学习模型能够自动学习图像深层特征，在工业视觉检测中表现出更高的识别精度。然而，不同模型在识别性能、计算复杂度和推理速度方面存在一定差异，针对PCB缺陷这一典型工业场景，不同方法之间的性能比较仍具有研究价值。

因此，本课题拟构建一套PCB缺陷分类系统，以公开数据集DeepPCB为基础，对传统模式识别算法（HOG+SVM、LBP+SVM）和深度学习算法（ResNet18、MobileNetV3）进行统一实验比较，并结合图像增强技术对深度学习模型进行优化，分析不同算法的识别性能及适用场景，为PCB缺陷自动分类提供一种完整的模式识别解决方案。

## 二、拟达到的研究目标和内容
### （一）研究目标
本课题拟设计并实现一套PCB缺陷分类系统，实现PCB缺陷图像的自动分类识别。系统能够完成数据读取、图像预处理、特征提取、模型训练、缺陷分类、结果可视化等功能，并对传统模式识别算法和深度学习算法进行实验比较，分析不同算法在PCB缺陷分类中的优缺点，最终获得识别精度较高、运行稳定、具有一定工程应用价值的PCB缺陷分类模型。

### （二）研究内容
1. **PCB缺陷数据集构建**
采用公开的DeepPCB数据集，根据数据集中提供的缺陷标注信息提取缺陷区域，构建适用于分类任务的数据集，并按照训练集、验证集和测试集进行划分。

2. **图像预处理与数据增强**
为了提高模型的鲁棒性，对PCB缺陷图像进行统一尺寸调整、归一化处理，并采用CLAHE（自适应直方图均衡化）增强图像局部对比度。同时采用旋转、翻转、亮度变化、随机噪声等数据增强方式，提高模型对复杂工业场景的适应能力。

3. **传统模式识别算法实现**
实现两种经典模式识别方法：
① HOG（Histogram of Oriented Gradients）特征提取结合SVM分类器；
② LBP（Local Binary Pattern）纹理特征结合SVM分类器。
分析两种人工特征在PCB缺陷分类中的识别效果。

4. **深度学习分类模型实现**
采用两种典型卷积神经网络完成PCB缺陷分类：
① ResNet18网络；
② MobileNetV3轻量化网络。
比较两种深度学习模型在分类精度、训练效率及推理速度等方面的差异。

5. **模型优化**
在深度学习模型基础上，引入CLAHE图像增强及数据增强策略，对模型输入进行优化，提高模型对细小缺陷及复杂背景的识别能力，并分析改进前后的分类性能变化。

6. **系统设计与实验分析**
完成PCB缺陷分类系统设计，实现PCB图片读取、分类识别、分类结果显示及结果保存等功能。
采用Accuracy、Precision、Recall、F1-score、Confusion Matrix（混淆矩阵）等评价指标，对不同算法进行性能分析，并从分类准确率、模型复杂度及运行效率等方面进行综合比较。

## 三、系统总体设计
整个系统按照模式识别经典流程设计。
```text
PCB缺陷图像
        │
        ▼
数据读取
        │
        ▼
图像预处理
（归一化、CLAHE）
        │
        ▼
数据增强
（旋转、翻转、亮度、噪声）
        │
        ▼
特征提取模块
        │
 ┌──────┼────────┬────────────┬────────────┐
 │      │        │            │
 ▼      ▼        ▼            ▼
HOG     LBP   ResNet18   MobileNetV3
 │      │        │            │
 ▼      ▼        ▼            ▼
SVM     SVM    Softmax     Softmax
        │
        ▼
PCB缺陷分类结果
        │
        ▼
性能评价与结果分析
```

## 四、实验方案（优化调整建议）
### 优化思路
将CLAHE作为所有深度学习模型统一预处理手段，区分数据预处理优化与网络结构优化，新增注意力机制实验；若项目时间不足，可退回原CLAHE+ResNet18方案。

| 实验编号 | 方法 | 目的 |
| ---- | ---------------- | -------------- |
| 实验一 | HOG + SVM | 传统模式识别Baseline |
| 实验二 | LBP + SVM | 纹理特征分类Baseline |
| 实验三 | ResNet18（CLAHE预处理） | 基础深度学习分类模型 |
| 实验四 | MobileNetV3（CLAHE预处理） | 轻量级深度学习模型 |
| 实验五 | ResNet18+SE/CBAM注意力机制 | 网络结构优化改进实验 |

实验约束：统一数据集、数据集划分、评价指标，保证实验结果可对比。

## 五、环境配置
### （一）硬件环境
| 项目 | 配置 |
| --- | ---------------------------------- |
| CPU | Intel Core i5 / i7 或 AMD Ryzen 5以上 |
| 内存 | 16GB |
| GPU | NVIDIA RTX3060（若无GPU可采用CPU训练） |
| 硬盘 | 20GB以上可用空间 |

### （二）软件环境
| 软件 | 用途 |
| -------------------------- | ------------- |
| Windows 11 | 操作系统 |
| Python 3.10 | 开发语言 |
| PyTorch | 深度学习框架 |
| OpenCV | 图像处理 |
| Scikit-learn | HOG、LBP、SVM实现 |
| NumPy | 数据处理 |
| Matplotlib | 可视化分析 |
| Jupyter Notebook / VS Code | 开发平台 |

## 六、小组分工方案
| 成员 | 负责模块 | 具体工作内容 | 工作量 |
| --- | ------- | ------------------------------------------- | --- |
| 组员1 | 数据处理模块 | 数据集整理、缺陷区域提取、CLAHE、数据增强、数据集划分 | 30% |
| 组员2 | 算法实现模块 | HOG+SVM、LBP+SVM、ResNet18、MobileNetV3、注意力模型搭建及训练 | 40% |
| 组员3 | 系统与实验模块 | 分类系统实现、实验对比、结果可视化、论文撰写、PPT制作 | 30% |

## 七、预期成果
1. 完成PCB缺陷分类系统设计与实现；
2. 实现HOG+SVM、LBP+SVM、ResNet18和MobileNetV3四种基础分类模型；
3. 实现基于CLAHE预处理与注意力机制的改进分类模型；
4. 完成多种模式识别算法的性能比较与分析；
5. 绘制训练曲线、混淆矩阵、准确率对比图等实验结果；
6. 完成课程设计报告、系统演示及答辩PPT。

## 八、课题特色与创新点
1. **多算法对比**：构建传统模式识别算法（HOG+SVM、LBP+SVM）与深度学习算法（ResNet18、MobileNetV3）的统一实验平台，系统比较不同算法在PCB缺陷分类中的性能差异。
2. **兼顾准确率与效率**：不仅比较分类精度，还综合分析模型参数量、推理速度和计算复杂度，为不同应用场景选择合适模型提供参考。
3. **双层优化策略**：采用CLAHE全域图像预处理+注意力网络结构优化，从数据、模型两个维度提升细小缺陷识别能力。
4. **完整模式识别流程**：系统涵盖数据预处理、特征提取、分类器设计、模型优化、性能评价和结果可视化，完整覆盖模式识别课程核心流程。

## 九、参考文献（建议）
[1] Dalal N, Triggs B. Histograms of Oriented Gradients for Human Detection[C]. CVPR, 2005.
[2] Ojala T, Pietikäinen M, Mäenpää T. Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns[J]. IEEE TPAMI, 2002.
[3] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. CVPR, 2016.
[4] Howard A G, et al. Searching for MobileNetV3[C]. ICCV, 2019.
[5] Song K, Yan Y. A Noise Robust Method Based on Completed Local Binary Patterns for Hot-Rolled Steel Strip Surface Defect Classification[J]. Applied Surface Science, 2013.
[6] 李航.《统计学习方法》. 清华大学出版社.
[7] 周志华.《机器学习》. 清华大学出版社.
[8] 冈萨雷斯, 伍兹.《数字图像处理》. 电子工业出版社.

## 十、项目完整文件架构
```
PCB_Classification/
│
├── datasets/                 # 数据集
│   ├── DeepPCB_original/     # 原始数据
│   └── PCB_Classification/   # 处理后分类数据集（train/val/test）
│
├── preprocess/               # 数据预处理
│   ├── crop_dataset.py       # 缺陷裁剪、数据集拆分
│   ├── clahe.py              # CLAHE图像增强
│   └── augment.py            # 随机数据增强脚本
│
├── traditional/              # 传统模式识别算法
│   ├── hog_svm.py
│   ├── lbp_svm.py
│   └── features.py           # 通用特征提取工具
│
├── deep_learning/
│   ├── models/               # 网络定义（ResNet18、MobileNetV3、注意力模块）
│   ├── train_resnet18.py
│   ├── train_mobilenet.py
│   └── predict.py            # 批量/单图推理预测
│
├── results/
│   ├── figures/              # 训练曲线、对比图表
│   ├── confusion_matrix/     # 各模型混淆矩阵图
│   └── logs/                 # 训练日志、指标记录
│
├── report/                   # 课程报告、答辩PPT存放目录
│
└── requirements.txt          # 项目依赖配置文件
```