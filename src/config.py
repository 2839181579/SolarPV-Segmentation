# 项目配置文件
# 所有路径都使用相对路径，便于跨平台使用

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型权重路径
MODEL_DIR = os.path.join(PROJECT_ROOT, "weights")

# 预训练权重
PRETRAINED_WEIGHTS = {
    "resnet50_unet": os.path.join(MODEL_DIR, "unet_resnet_voc.pth"),
    "mitb5_unet": os.path.join(MODEL_DIR, "Mit-B5-UNet-Champion-92-32.pth"),
    "resnet50_champion": os.path.join(MODEL_DIR, "ResNet50-UNet-Champion_model_91.83.pth"),
}

# 数据集路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VOC_DIR = os.path.join(DATA_DIR, "VOCdevkit", "VOC2007")

# 输入输出路径
INPUT_DIR = os.path.join(VOC_DIR, "JPEGImages")
MASK_DIR = os.path.join(VOC_DIR, "SegmentationClass")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval_result")

# 日志目录
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def get_model_path(model_name):
    """获取模型路径"""
    return PRETRAINED_WEIGHTS.get(model_name, "")

def get_data_path(subdir=""):
    """获取数据路径"""
    if subdir:
        return os.path.join(DATA_DIR, subdir)
    return DATA_DIR