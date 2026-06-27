# 🌞 SolarPV-Segmentation: Deep Learning-based Solar PV Facility Intelligent Recognition

<p align="center">
  <img src="assets/banner.png" alt="Solar PV Segmentation" width="800">
</p>

<p align="center">
  <strong>🏆 1st Place - 6th International High-Resolution RS Image Interpretation Contest | 📊 mIoU 92.32% | 🖥️ Streamlit Demo</strong>
</p>

<p align="center">
  <a href="README.md">中文</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📖 Introduction

This project won **1st Place** in the **6th International High-Resolution Remote Sensing Image Interpretation Contest Finals**, implementing a deep learning-based intelligent recognition system for solar PV facilities.

**Key Highlights:**
- 🏆 **International Contest Champion**: 1st Place in 6th International High-Resolution RS Image Interpretation Contest Finals
- 📊 **High Accuracy**: mIoU reaches 92.32% (MiT-B5 backbone)
- 🖥️ **Visualization System**: Interactive comparison and evaluation platform based on Streamlit
- 📄 **Complete Materials**: Technical report, poster, and presentation all open-sourced

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/2839181579/SolarPV-Segmentation.git
cd SolarPV-Segmentation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

1. Download dataset (see [data/README.md](data/README.md))
2. Place data in `data/VOCdevkit/VOC2007/` directory
3. Run data preparation script:

```bash
python src/prepare_data.py
```

### 3. Model Training

```bash
# ResNet50 baseline model (mIoU 91.83%)
python src/train.py

# MiT-B5 advanced model (mIoU 92.32%)
python src/train_mitb5.py
```

### 4. Model Prediction

```bash
# ResNet50 prediction
python src/predict.py

# MiT-B5 prediction
python src/predict_mitb5.py
```

### 5. Visualization Demo

```bash
# Launch Streamlit interface
streamlit run demo/app.py

# Or use launch script
python demo/launch.py
```

## 📊 Results

### Model Comparison

| Model | Backbone | mIoU | Parameters | Inference Speed |
|-------|----------|------|------------|-----------------|
| UNet-ResNet50 | ResNet50 | 91.83% | 32.5M | 45ms/image |
| UNet-MiT-B5 | MiT-B5 | **92.32%** | 84.3M | 68ms/image |

### Visualization Results

**MiT-B5 vs ResNet50 Comparison** (Samples where MiT-B5 has significant advantages):

<p align="center">
  <img src="assets/rank_03.png" alt="Rank #3 - MiT-B5 Perfect Prediction" width="800">
</p>
<p align="center">
  <em>Rank #3: MiT-B5 perfect prediction (IoU 100%), ResNet50 false detection (IoU 0%)</em>
</p>

<p align="center">
  <img src="assets/rank_14.png" alt="Rank #14 - MiT-B5 Perfect Prediction" width="800">
</p>
<p align="center">
  <em>Rank #14: MiT-B5 perfect prediction (IoU 100%), ResNet50 false detection (IoU 0%)</em>
</p>

<p align="center">
  <img src="assets/rank_26.png" alt="Rank #26 - MiT-B5 High Accuracy" width="800">
</p>
<p align="center">
  <em>Rank #26: MiT-B5 high accuracy (IoU 80.93%), ResNet50 complete miss (IoU 0%)</em>
</p>

<p align="center">
  <img src="assets/rank_33.png" alt="Rank #33 - MiT-B5 Clear Advantage" width="800">
</p>
<p align="center">
  <em>Rank #33: MiT-B5 accurate prediction (IoU 77.96%), ResNet50 severe miss (IoU 28.10%)</em>
</p>

<p align="center">
  <img src="assets/rank_36.png" alt="Rank #36 - MiT-B5 Clear Advantage" width="800">
</p>
<p align="center">
  <em>Rank #36: MiT-B5 accurate prediction (IoU 80.63%), ResNet50 large alignment deviation (IoU 47.11%)</em>
</p>

<p align="center">
  <img src="assets/rank_40.png" alt="Rank #40 - MiT-B5 Clear Advantage" width="800">
</p>
<p align="center">
  <em>Rank #40: MiT-B5 comprehensive prediction (IoU 68.78%), ResNet50 partial miss (IoU 43.25%)</em>
</p>

> **Note**: The above shows samples where MiT-B5 model has the most significant advantages over ResNet50 (sorted by IoU difference). Each comparison includes: original satellite image, ground truth, MiT-B5 prediction, ResNet50 prediction.

## 📁 Project Structure

```
SolarPV-Segmentation/
├── src/                    # Source code
│   ├── train.py            # ResNet50 training
│   ├── train_mitb5.py      # MiT-B5 training
│   ├── predict.py          # Prediction script
│   ├── evaluate.py         # Evaluation script
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── demo/                   # Visualization demo
├── data/                   # Data directory
├── weights/                # Model weights
├── docs/                   # Documentation
└── assets/                 # Display materials
```

## 📚 Documentation

- 📄 [Technical Report](docs/technical_report.pdf)
- 🖼️ [Project Poster](docs/poster.pdf)
- 📊 [Presentation Slides](docs/slides.pptx)
- 📝 [Data Acquisition Guide](data/README.md)
- ⚖️ [Model Weights Guide](weights/README.md)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) to learn how to participate.

## 📜 License

This project is open-sourced under the MIT License, see [LICENSE](LICENSE).

**Note**: This project is based on [bubbliiiing/unet-pytorch](https://github.com/bubbliiiing/unet-pytorch) with improvements, retaining the original author's copyright notice.

## 🙏 Acknowledgments

- Thanks to [bubbliiiing](https://github.com/bubbliiiing) for providing the UNet basic implementation
- Thanks to [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) for providing the MiT-B5 backbone
- Thanks to the contest organizing committee for providing the dataset and evaluation

## 📝 Citation

If this project helps your research, please cite:

```bibtex
@misc{solarpvsegmentation2025,
  title={SolarPV-Segmentation: Deep Learning-based Solar PV Facility Intelligent Recognition},
  author={Ma Shaobo},
  year={2025},
  howpublished={\url{https://github.com/2839181579/SolarPV-Segmentation}},
  note={1st Place - 6th International High-Resolution RS Image Interpretation Contest}
}
```

Or cite in your paper:

```
Ma Shaobo. SolarPV-Segmentation: Deep Learning-based Solar PV Facility Intelligent Recognition. 
GitHub Repository, 2025. https://github.com/2839181579/SolarPV-Segmentation
```

## 📧 Contact

If you have any questions, please contact us through:
- Submit [Issue](https://github.com/2839181579/SolarPV-Segmentation/issues)
- Email: mashaobo@example.com

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=2839181579/SolarPV-Segmentation&type=Date)](https://star-history.com/#2839181579/SolarPV-Segmentation&Date)

---

<p align="center">
  If you find this helpful, please give a ⭐ to support!
</p>