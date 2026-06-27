# 📂 数据获取说明

## 数据集来源

本项目使用的是**第六届国际高分辨率遥感图像智能解译大赛决赛**官方数据集，包含高分二号（GF-2）卫星影像的光伏设施标注。

## 样例数据

为了方便用户了解数据格式，我们提供了 3 张样例图片：

```
data/sample/
├── images/              # 原始卫星影像（.jpg）
│   ├── sample_01.jpg
│   ├── sample_02.jpg
│   └── sample_03.jpg
└── masks/               # 语义分割标签（.png）
    ├── sample_01.png
    ├── sample_02.png
    └── sample_03.png
```

## 数据格式

### 原始影像（JPEGImages）
- **格式**：`.jpg` 或 `.png`
- **来源**：高分二号（GF-2）卫星
- **分辨率**：1米（全色）/ 4米（多光谱）
- **内容**：包含光伏设施的遥感影像

### 分割标签（SegmentationClass）
- **格式**：`.png`（灰度图）
- **像素值**：
  - `0` = 背景
  - `1` = 光伏设施
- **说明**：每个像素的值表示该像素所属的类别

### 数据集划分（ImageSets）
- **格式**：`.txt` 文件
- **内容**：每行一个文件名（不含扩展名）
- **文件**：
  - `train.txt` - 训练集文件名列表
  - `val.txt` - 验证集文件名列表

## 完整数据集获取

### 方式1：百度网盘下载（推荐）

**完整训练数据集**（1.7GB，包含 4,000 张原始影像 + 4,003 张分割标签）：

- **链接**：https://pan.baidu.com/s/17H0SgdH28JeRyhXP0fjHUw
- **提取码**：ygrb

**下载后请将数据放到以下目录**：
```
data/VOCdevkit/VOC2007/
├── JPEGImages/          # 将 0train/images/ 中的文件放在这里
└── SegmentationClass/   # 将 0train/mask/ 中的文件放在这里
```

### 方式2：比赛官方渠道

如需获取完整数据集，请通过以下方式联系：

- **比赛名称**：第六届国际高分辨率遥感图像智能解译大赛
- **联系方式**：请通过比赛官方网站或组委会获取

### 方式3：联系作者

如需获取数据集用于学术研究，请联系项目作者：
- **GitHub Issue**：[提交 Issue](https://github.com/2839181579/SolarPV-Segmentation/issues)
- **邮箱**：mashaobo@example.com

### 方式3：自定义数据集

如果你想使用自己的数据，请按照以下格式准备：

1. **原始影像**：放入 `data/VOCdevkit/VOC2007/JPEGImages/` 目录
   - 格式：`.jpg` 或 `.png`
   - 命名：任意，但建议使用有意义的名称

2. **分割标签**：放入 `data/VOCdevkit/VOC2007/SegmentationClass/` 目录
   - 格式：`.png`（灰度图）
   - 像素值：背景=0，目标=1
   - 命名：与原始影像同名，但扩展名为 `.png`

3. **数据划分**：在 `data/VOCdevkit/VOC2007/ImageSets/Segmentation/` 中创建：
   - `train.txt` - 训练集文件名列表
   - `val.txt` - 验证集文件名列表

## 数据预处理

准备好数据后，运行以下命令生成训练/验证集划分：

```bash
python src/prepare_data.py
```

## 数据统计

| 项目 | 说明 |
|------|------|
| **影像来源** | 高分二号（GF-2）卫星 |
| **空间分辨率** | 1米（全色）/ 4米（多光谱） |
| **标注类别** | 背景、光伏设施 |
| **训练集** | 约 XXX 张（完整数据集） |
| **验证集** | 约 XXX 张（完整数据集） |

## 注意事项

1. **数据集仅供学术研究使用**，请勿用于商业用途
2. **使用数据集时请引用本项目**（见 README.md 的 Citation 部分）
3. **如有数据问题**，请提交 [Issue](https://github.com/2839181579/SolarPV-Segmentation/issues)

## 快速开始

如果你想快速体验项目，可以使用样例数据：

```bash
# 1. 使用样例数据进行预测
python src/predict.py

# 2. 启动可视化 Demo
streamlit run demo/app.py
```

---

**注意**：样例数据仅供格式参考，完整训练请使用完整数据集。