# 🏆 国际大赛一等奖 | 基于深度学习的光伏设施智能识别系统（开源）

> 🎯 **摘要**：本文介绍了我在第六届国际高分辨率遥感图像智能解译大赛决赛中获得一等奖的项目——基于深度学习的光伏设施智能识别系统。项目采用 UNet + MiT-B5 架构，mIoU 达到 92.32%，并开发了 Streamlit 可视化对比系统。完整代码已开源，欢迎 Star 和 Fork！

---

## 📖 一、项目背景

### 1.1 比赛介绍

**第六届国际高分辨率遥感图像智能解译大赛**是遥感领域的重要赛事，吸引了来自全球的参赛者。比赛要求参赛者利用深度学习技术，对高分辨率遥感影像中的光伏设施进行精准识别和分割。

### 1.2 比赛挑战

- **数据复杂**：高分二号（GF-2）卫星影像，空间分辨率 1 米
- **目标多样**：光伏设施形态各异，大小不一
- **背景复杂**：包含建筑物、植被、道路等多种地物

### 1.3 我的解决方案

我设计了一个基于 UNet 的语义分割系统，通过对比多种主干网络（ResNet50、VGG、MiT-B5），最终选择 **MiT-B5** 作为主干网络，实现了 **mIoU 92.32%** 的高精度分割。

---

## 🏗️ 二、技术实现

### 2.1 整体架构

```
输入影像 → UNet 编码器 → UNet 解码器 → 分割结果
           ↓
       MiT-B5 主干网络
       （Transformer 架构）
```

### 2.2 主干网络对比

| 主干网络 | 架构 | mIoU | 参数量 | 推理速度 |
|----------|------|------|--------|----------|
| ResNet50 | CNN | 91.83% | 32.5M | 45ms/张 |
| VGG | CNN | 89.5% | 20.1M | 38ms/张 |
| **MiT-B5** | **Transformer** | **92.32%** | 84.3M | 68ms/张 |

### 2.3 为什么选择 MiT-B5？

**MiT-B5** 是 SegFormer 系列中最强的主干网络之一，具有以下优势：

1. **强大的特征提取能力**：基于 Transformer 架构，能够捕捉长距离依赖关系
2. **多尺度特征融合**：有效处理不同大小的光伏设施
3. **高精度**：在遥感影像分割任务中表现优异

### 2.4 关键技术点

#### 2.4.1 数据预处理

```python
# 数据格式：VOC 格式
data/VOCdevkit/VOC2007/
├── JPEGImages/          # 原始卫星影像（.jpg）
├── SegmentationClass/   # 语义分割标签（.png）
└── ImageSets/           # 数据集划分文件
```

#### 2.4.2 模型训练

```python
# ResNet50 基线模型
python src/train.py

# MiT-B5 进阶模型
python src/train_mitb5.py
```

#### 2.4.3 模型预测

```python
# ResNet50 预测
python src/predict.py

# MiT-B5 预测
python src/predict_mitb5.py
```

---

## 📊 三、实验结果

### 3.1 定量结果

| 模型 | 主干网络 | mIoU | 参数量 | 推理速度 |
|------|----------|------|--------|----------|
| UNet-ResNet50 | ResNet50 | 91.83% | 32.5M | 45ms/张 |
| UNet-MiT-B5 | MiT-B5 | **92.32%** | 84.3M | 68ms/张 |

### 3.2 定性结果

**MiT-B5 vs ResNet50 对比展示**（MiT-B5 优势明显的样例）：

<p align="center">
  <img src="https://github.com/2839181579/SolarPV-Segmentation/raw/main/assets/rank_03.png" alt="Rank #3" width="600">
</p>
<p align="center">
  <em>Rank #3：MiT-B5 完美预测（IoU 100%），ResNet50 出现误检（IoU 0%）</em>
</p>

<p align="center">
  <img src="https://github.com/2839181579/SolarPV-Segmentation/raw/main/assets/rank_26.png" alt="Rank #26" width="600">
</p>
<p align="center">
  <em>Rank #26：MiT-B5 高精度预测（IoU 80.93%），ResNet50 完全漏检（IoU 0%）</em>
</p>

### 3.3 关键发现

1. **MiT-B5 在复杂场景下表现更优**：对于背景复杂、目标多样的场景，MiT-B5 能够更准确地识别光伏设施
2. **ResNet50 在简单场景下足够**：对于背景简单、目标明显的场景，ResNet50 已经能够达到较好的效果
3. **模型选择需权衡**：精度和速度需要根据实际需求进行权衡

---

## 🖥️ 四、可视化系统

### 4.1 系统功能

为了方便用户使用和对比，我开发了基于 **Streamlit** 的可视化对比系统，主要功能包括：

1. **多模型对比**：同时加载两个不同模型，实时对比预测结果
2. **批量评测**：一键遍历整个文件夹，生成 CSV 评测报告
3. **批量预测**：带进度条的批量推理工具
4. **结果浏览**：并排查看原图与预测结果

### 4.2 启动方式

```bash
# 启动 Streamlit 界面
streamlit run demo/app.py

# 或使用启动脚本
python demo/launch.py
```

### 4.3 系统截图

<p align="center">
  <img src="https://github.com/2839181579/SolarPV-Segmentation/raw/main/assets/rank_03.png" alt="可视化系统" width="600">
</p>
<p align="center">
  <em>可视化对比系统界面</em>
</p>

---

## 🚀 五、快速开始

### 5.1 环境配置

```bash
# 克隆仓库
git clone https://github.com/2839181579/SolarPV-Segmentation.git
cd SolarPV-Segmentation

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 5.2 数据准备

1. 下载数据集（百度网盘链接见 README）
2. 将数据放入 `data/VOCdevkit/VOC2007/` 目录
3. 运行数据准备脚本：

```bash
python src/prepare_data.py
```

### 5.3 模型训练

```bash
# ResNet50 基线模型（mIoU 91.83%）
python src/train.py

# MiT-B5 进阶模型（mIoU 92.32%）
python src/train_mitb5.py
```

### 5.4 可视化 Demo

```bash
# 启动 Streamlit 界面
streamlit run demo/app.py
```

---

## 📚 六、项目资源

### 6.1 GitHub 仓库

**项目地址**：https://github.com/2839181579/SolarPV-Segmentation

**欢迎 Star 和 Fork！** ⭐

### 6.2 文档资料

- 📄 [技术报告](https://github.com/2839181579/SolarPV-Segmentation/blob/main/docs/technical_report.pdf)
- 🖼️ [项目海报](https://github.com/2839181579/SolarPV-Segmentation/blob/main/docs/poster.pdf)
- 📊 [演示PPT](https://github.com/2839181579/SolarPV-Segmentation/blob/main/docs/slides.pptx)

### 6.3 数据集

- **完整数据集**：百度网盘下载（链接见 README）
- **样例数据**：仓库中提供 3 张样例图片

---

## 🎯 七、总结与展望

### 7.1 项目总结

1. **高精度**：mIoU 达到 92.32%，在比赛中获得一等奖
2. **完整系统**：从数据预处理到可视化 Demo 的完整流程
3. **开源共享**：完整代码和文档开源，方便复现和改进

### 7.2 技术收获

1. **Transformer 在遥感中的应用**：MiT-B5 展示了 Transformer 架构在遥感影像分割中的优势
2. **模型对比的重要性**：通过对比不同主干网络，选择最适合的模型
3. **可视化系统的价值**：Streamlit 可视化系统大大提升了用户体验

### 7.3 未来展望

1. **更多骨干网络**：尝试更多先进的骨干网络，如 Swin Transformer
2. **多任务学习**：结合目标检测、语义分割等多任务
3. **实时推理**：优化模型，实现实时推理

---

## 🙏 致谢

- 感谢 [bubbliiiing](https://github.com/bubbliiiing) 提供的 UNet 基础实现
- 感谢 [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) 提供的 MiT-B5 主干网络
- 感谢大赛组委会提供的数据集和评审

---

## 📧 联系方式

如有问题，欢迎通过以下方式联系：

- **GitHub Issue**：[提交 Issue](https://github.com/2839181579/SolarPV-Segmentation/issues)
- **邮箱**：mashaobo@example.com

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=2839181579/SolarPV-Segmentation&type=Date)](https://star-history.com/#2839181579/SolarPV-Segmentation&Date)

---

**如果觉得有帮助，请给个 ⭐ 支持一下！**