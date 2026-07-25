# 模型权重说明

训练权重体积较大，不直接纳入 Git 仓库。运行推理或 Streamlit 可视化平台前，请在本地准备权重文件。

| 模型 | 默认文件名 | 项目记录 mIoU | 约占空间 |
|---|---|---:|---:|
| UNet-ResNet50 | `ResNet50-UNet-Champion_model_91.83.pth` | 91.83% | 168 MB |
| UNet-MiT-B5 | `Mit-B5-UNet-Champion-92-32.pth` | 92.32% | 324 MB |

将文件放入：

```text
weights/
├── ResNet50-UNet-Champion_model_91.83.pth
└── Mit-B5-UNet-Champion-92-32.pth
```

`demo/app.py` 默认读取以上相对路径，也可以在界面中选择其他本地权重路径。

如需从头训练，请先准备数据并运行：

```bash
python src/train.py
python src/train_mitb5.py
```

模型结果可能随数据划分、随机种子、硬件与依赖版本略有变化。请遵守比赛数据和模型文件的使用要求。
