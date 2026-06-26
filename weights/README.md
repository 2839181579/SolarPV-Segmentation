# ⚖️ 模型权重说明

## 预训练权重

本项目提供两个训练好的模型权重：

| 模型 | 文件名 | mIoU | 大小 | 下载链接 |
|------|--------|------|------|----------|
| UNet-ResNet50 | `ResNet50-UNet-Champion_model_91.83.pth` | 91.83% | 168MB | [百度网盘]() |
| UNet-MiT-B5 | `Mit-B5-UNet-Champion-92-32.pth` | 92.32% | 324MB | [百度网盘]() |

## 使用方式

### 1. 下载权重

从上述链接下载权重文件，放入 `weights/` 目录：

```
weights/
├── ResNet50-UNet-Champion_model_91.83.pth
└── Mit-B5-UNet-Champion-92-32.pth
```

### 2. 修改代码中的路径

在相应的脚本中修改权重路径：

**src/train.py** 或 **src/train_mitb5.py**：
```python
# 修改 model_path 指向你的权重文件
model_path = "weights/ResNet50-UNet-Champion_model_91.83.pth"
```

**src/predict.py** 或 **src/predict_mitb5.py**：
```python
# 修改 model_path 指向你的权重文件
model_path = "weights/Mit-B5-UNet-Champion-92-32.pth"
```

## 自己训练权重

如果你想自己训练模型，请参考 [训练指南](../README.md#3-模型训练)。

## 注意事项

1. 权重文件较大，请确保有足够的磁盘空间
2. 权重仅供学术研究使用，请勿用于商业用途
3. 使用权重时请引用本项目