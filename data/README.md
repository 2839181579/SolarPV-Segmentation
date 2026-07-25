# 数据说明

## 数据来源

本项目使用第六届国际高分辨率遥感图像智能解译大赛决赛数据，任务为高分辨率遥感影像中的光伏设施语义分割。

完整比赛数据不纳入本仓库。请通过赛事官方渠道、并按照数据使用要求获取；仓库仅提供 3 组脱敏样例，用于检查目录结构、界面和输入输出格式。

```text
data/sample/
├── images/
│   ├── sample_01.jpg
│   ├── sample_02.jpg
│   └── sample_03.jpg
└── masks/
    ├── sample_01.png
    ├── sample_02.png
    └── sample_03.png
```

## 数据格式

- 原始影像：`.jpg` 或 `.png`。
- 分割标签：单通道 `.png`，像素值 `0` 表示背景、`1` 表示光伏设施。
- 划分文件：`train.txt`、`val.txt` 等文本文件，每行填写一个不含扩展名的样本名称。

如使用自定义数据，请整理为：

```text
data/VOCdevkit/VOC2007/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/Segmentation/
    ├── train.txt
    └── val.txt
```

随后运行：

```bash
python src/prepare_data.py
```

## 项目记录中的数据划分

| 划分 | 数量 |
|---|---:|
| 训练集 | 3,600 |
| 验证集 | 400 |
| 合计 | 4,000 |

以上数字对应本项目归档的 `train.txt`、`val.txt` 与 `trainval.txt`。更换随机种子或自定义划分时，数量可能不同。

## 注意事项

1. 比赛数据请仅在赛事规则和授权范围内使用。
2. 样例数据只用于格式与界面测试，不能复现完整训练指标。
3. 数据相关问题可通过 [GitHub Issues](https://github.com/2839181579/SolarPV-Segmentation/issues) 反馈。
