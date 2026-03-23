# CAB-DNet: Category-Aware Band Selection based Detail-Enhanced Network

---

## 简介 | Introduction

**CAB-DNet（Category-Aware Band Selection based Detail-Enhanced Network）** 是一种用于高光谱图像分类的新型深度学习模型。该模型通过结合类别感知波段选择（SWL）模块与细节增强卷积（DEConv）模块，有效解决了高光谱数据冗余和传统卷积对高频细节表征能力有限两大问题。

**CAB-DNet (Category-Aware Band Selection based Detail-Enhanced Network)** is a novel deep learning model for hyperspectral image classification. By integrating the Spectral Weight Learning (SWL) module with the Detail-Enhanced Convolution (DEConv) module, the model effectively addresses two major challenges: spectral redundancy in hyperspectral data and the limited capability of traditional convolutions to represent high-frequency details.

---

## 主要特性 | Key Features

| 特性 | Feature | 说明 | Description |
|------|---------|------|-------------|
| **类别感知波段选择** | **Category-Aware Band Selection (SWL)** | 通过光谱注意力机制和类别一致性损失，学习每个类别的专属光谱权重，实现关键波段的精准选择与增强 | Learns class-specific spectral weights through spectral attention and category consistency loss, enabling precise selection and enhancement of critical bands |
| **细节增强卷积** | **Detail-Enhanced Convolution (DEConv)** | 并行集成常规卷积与四种差分卷积，同时捕获低频强度与高频梯度特征 | Integrates vanilla convolution with four difference convolutions in parallel, capturing both low-frequency intensity and high-frequency gradient features |
| **重参数化技术** | **Re-parameterization** | 训练时保留并行结构，推理时等效转换为单层卷积，实现精度与效率的统一 | Preserves parallel structure during training and converts to single-layer convolution during inference, achieving balance between accuracy and efficiency |
| **端到端训练** | **End-to-End Training** | 支持完整的训练、测试与预测流程 | Supports complete training, testing, and prediction pipeline |

---

## 项目结构 | Project Structure

```
CAB-DNet/
├── CAB_DNet.py          # 模型核心网络结构 | Core network architecture
├── data_loader.py       # 数据加载与预处理 | Data loading and preprocessing
├── main_train.py        # 模型训练主脚本 | Main training script
├── main_test.py         # 模型测试主脚本 | Main testing script
├── main_os.py           # 操作系统级配置入口 | OS-level configuration entry
├── pred_all_new1.py     # 预测脚本 | Prediction script
└── README.md            # 项目说明文档 | Project documentation
```

---

## 文件说明 | File Description

| 文件 | File | 功能 | Function |
|------|------|------|----------|
| **CAB_DNet.py** | **CAB_DNet.py** | 定义CAB-DNet网络架构，包含SWL模块、DEConv模块、重参数化实现 | Defines CAB-DNet architecture, including SWL module, DEConv module, and re-parameterization implementation |
| **data_loader.py** | **data_loader.py** | 数据加载、归一化、训练/测试样本划分、数据增强 | Data loading, normalization, training/testing sample splitting, and data augmentation |
| **main_train.py** | **main_train.py** | 模型训练主程序，配置训练参数并执行训练 | Main training program, configures training parameters and executes training |
| **main_test.py** | **main_test.py** | 模型测试主程序，计算OA、AA、Kappa等评价指标 | Main testing program, computes evaluation metrics including OA, AA, and Kappa |
| **main_os.py** | **main_os.py** | 系统级配置，支持多GPU/多进程训练设置 | System-level configuration, supports multi-GPU/multi-process training setup |
| **pred_all_new1.py** | **pred_all_new1.py** | 对新数据进行预测，生成分类结果图 | Performs prediction on new data and generates classification maps |

---

## 环境要求 | Requirements

| 依赖项 | Dependency | 版本 | Version |
|--------|------------|------|---------|
| Python | Python | 3.8+ | 3.8+ |
| PyTorch | PyTorch | 1.10+ | 1.10+ |
| NumPy | NumPy | 1.19+ | 1.19+ |
| Scikit-learn | Scikit-learn | 0.24+ | 0.24+ |
| Matplotlib | Matplotlib | 3.3+ | 3.3+ |

---

## 训练参数 | Training Parameters

| 参数 | Parameter | 默认值 | Default | 说明 | Description |
|------|-----------|--------|---------|------|-------------|
| **batch_size** | **batch_size** | 16 | 16 | 批次大小 | Batch size |
| **lr_start** | **lr_start** | 1e-2 | 1e-2 | 初始学习率 | Initial learning rate |
| **epochs** | **epochs** | 200 | 200 | 训练轮数 | Number of training epochs |
| **patch_size** | **patch_size** | 13×13 | 13×13 | 输入图像块尺寸 | Input patch size |
| **optimizer** | **optimizer** | adam | adam | 优化器类型 | Optimizer type |
| **weight_decay** | **weight_decay** | 0.0005 | 0.0005 | 权重衰减 | Weight decay |
| **lr_scheduler** | **lr_scheduler** | multisteplr | multisteplr | 学习率调度策略 | Learning rate scheduler |
| **milestones** | **milestones** | [40] | [40] | 学习率衰减轮次 | Epochs for learning rate decay |

---

## 各数据集训练比例 | Training Ratio by Dataset

| 数据集 | Dataset | 训练比例 | Training Ratio | 说明 | Description |
|--------|---------|----------|----------------|------|-------------|
| **PaviaU** | **PaviaU** | 1% | 1% | 每类1%样本用于训练 | 1% samples per class for training |
| **IP (Indian Pines)** | **IP (Indian Pines)** | 10% | 10% | 每类10%样本用于训练 | 10% samples per class for training |
| **Houston** | **Houston** | 4% | 4% | 每类4%样本用于训练 | 4% samples per class for training |
| **Salinas** | **Salinas** | 0.8% | 0.8% | 每类0.8%样本用于训练 | 0.8% samples per class for training |

---

## 快速开始 | Quick Start

### 1. 环境配置 | Environment Setup

```bash
# 克隆代码仓库 | Clone repository
git clone https://gitee.com/mohansir/swd-net.git
cd swd-net

# 安装依赖 | Install dependencies
pip install torch numpy scikit-learn matplotlib
```

### 2. 数据准备 | Data Preparation

将高光谱数据集放置于指定路径，按照数据加载器的格式要求组织数据。

Place the hyperspectral dataset in the specified path and organize the data according to the data loader format requirements.

### 3. 模型训练 | Training

```bash
# 使用默认数据集（PaviaU）| Use default dataset (PaviaU)
python main_train.py

# 指定数据集 | Specify dataset
python main_train.py --dataset IP --train_ratio 0.1

# 手动覆盖训练比例 | Override training ratio
python main_train.py --dataset Houston --train_ratio 0.05
```

### 4. 模型测试 | Testing

```bash
# 加载预训练模型进行测试 | Load pre-trained model for testing
python main_test.py --dataset PaviaU --checkpointsmodelfile ./checkpoints/own/own.pth
```

### 5. 预测新数据 | Prediction

```bash
# 对新数据进行分类预测 | Perform classification prediction on new data
python pred_all_new1.py
```

---

## 数据集 | Datasets

| 数据集 | Dataset | 传感器 | Sensor | 空间尺寸 | Spatial Size | 波段数 | Bands | 类别数 | Classes |
|--------|---------|--------|--------|------------|--------------|--------|-------|--------|---------|
| **Indian Pines** | **Indian Pines** | AVIRIS | AVIRIS | 145×145 | 145×145 | 200 | 200 | 16 | 16 |
| **Pavia University** | **Pavia University** | ROSIS | ROSIS | 610×340 | 610×340 | 103 | 103 | 9 | 9 |
| **Salinas** | **Salinas** | AVIRIS | AVIRIS | 512×217 | 512×217 | 204 | 204 | 16 | 16 |
| **Houston 2013** | **Houston 2013** | ITRES CASI-1500 | ITRES CASI-1500 | 349×1905 | 349×1905 | 144 | 144 | 15 | 15 |

---

## 评价指标 | Evaluation Metrics

| 指标 | Metric | 说明 | Description |
|------|--------|------|-------------|
| **OA** | **Overall Accuracy** | 总体分类精度，正确分类像素占总像素的比例 | Overall classification accuracy, ratio of correctly classified pixels to total pixels |
| **AA** | **Average Accuracy** | 平均分类精度，所有类别分类精度的平均值 | Average classification accuracy, mean of accuracy across all classes |
| **Kappa** | **Kappa Coefficient** | Kappa系数，衡量分类结果与真实标签的一致性 | Kappa coefficient, measures consistency between classification results and ground truth |

---

## 引用 | Citation

如果本工作对您的研究有帮助，请引用：

If this work is helpful to your research, please cite:

```bibtex
@article{CABDNet2025,
  title={CAB-DNet: Category-Aware Band Selection based Detail-Enhanced Network for Hyperspectral Image Classification},
  author={Your Name},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}
}
```

---

## 代码开源 | Code Availability

本方法代码已开源，欢迎访问与使用：

The source code of this method is open source. Welcome to visit and use:

🔗 **https://gitee.com/mohansir/swd-net.git**

---

## 联系方式 | Contact

如有任何问题或建议，请通过GitHub Issues或邮件联系。

If you have any questions or suggestions, please contact us via GitHub Issues or email.

---

## 许可证 | License

本项目采用 MIT 许可证。

This project is licensed under the MIT License.
