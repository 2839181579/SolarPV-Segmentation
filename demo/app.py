import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog
import subprocess
import sys
import pandas as pd  # 用于数据统计
import math


# ===========================================================================
#   工具函数：调用 Windows 原生选择框
# ===========================================================================
def select_folder_dialog(title="请选择文件夹"):
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        folder_path = filedialog.askdirectory(title=title)
        root.destroy()
        return folder_path
    except Exception as e:
        print(f"Tkinter error: {e}")
        return ""


def select_file_dialog(title="请选择文件", filetypes=[("Model files", "*.pth"), ("All files", "*.*")]):
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return file_path
    except Exception as e:
        print(f"Tkinter error: {e}")
        return ""


# ===========================================================================
#   【核心】回调函数
# ===========================================================================
def on_click_select_file(key_name):
    """点击按钮时触发：选择文件并更新 session_state"""
    f = select_file_dialog()
    if f:
        st.session_state[key_name] = f


def on_click_select_folder(key_name):
    """点击按钮时触发：选择文件夹并更新 session_state"""
    d = select_folder_dialog()
    if d:
        st.session_state[key_name] = d


# ===========================================================================
#   IoU 计算核心逻辑
# ===========================================================================
def calculate_iou(gt_image, pred_image):
    """
    计算两张图片的 IoU (交并比)
    """
    try:
        # 1. 转换为 Numpy 数组
        gt_arr = np.array(gt_image)
        pred_arr = np.array(pred_image)

        # 2. 尺寸对齐
        if gt_arr.shape[:2] != pred_arr.shape[:2]:
            pred_image = pred_image.resize((gt_image.width, gt_image.height), Image.NEAREST)
            pred_arr = np.array(pred_image)

        # 3. 二值化处理
        if len(gt_arr.shape) == 3:
            gt_mask = np.max(gt_arr, axis=2) > 0
        else:
            gt_mask = gt_arr > 0

        if len(pred_arr.shape) == 3:
            pred_mask = np.max(pred_arr, axis=2) > 0
        else:
            pred_mask = pred_arr > 0

        # 4. 计算交集和并集
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0, "空集"

        iou = intersection / union
        return iou, "Success"

    except Exception as e:
        return 0.0, str(e)


# ===========================================================================
#   图片拼接工具 (用于保存对比图和实时预览)
#   【修复】字体加载逻辑，解决中文乱码方框问题
# ===========================================================================
def concat_images_horizontal(images, labels=None, padding=10, bg_color=(255, 255, 255)):
    """将多张图片横向拼接，并添加文字标签和分隔线"""
    widths, heights = zip(*(i.size for i in images))

    # 总宽度 = 所有图片宽度 + (图片数量-1) * 间隔
    total_width = sum(widths) + (len(images) - 1) * padding
    max_height = max(heights)

    # 如果有标签，增加顶部文字区域高度
    text_height = 40 if labels else 0

    new_im = Image.new('RGB', (total_width, max_height + text_height), bg_color)

    x_offset = 0
    draw = ImageDraw.Draw(new_im)

    # 【关键修复】尝试加载支持中文的字体 (Windows常用字体)
    font = None
    try:
        # 尝试加载黑体 (Windows 默认存在)
        font = ImageFont.truetype("simhei.ttf", 24)
    except:
        try:
            # 尝试加载微软雅黑
            font = ImageFont.truetype("msyh.ttf", 24)
        except:
            try:
                # 尝试加载 Arial (不支持中文，但作为保底)
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                # 最后的保底，不支持中文 (会显示方框)
                font = ImageFont.load_default()

    for idx, im in enumerate(images):
        # 粘贴图片
        new_im.paste(im, (x_offset, text_height))

        # 绘制标签
        if labels and idx < len(labels):
            label_text = labels[idx]
            # 计算文字宽高
            if font:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
            else:
                text_w = 0  # Fallback

            # 文本相对于当前图片的居中位置
            text_x = x_offset + (im.width - text_w) // 2

            # 绘制文字 (黑色)
            draw.text((text_x, 5), label_text, fill=(0, 0, 0), font=font)

        # 更新下一个图片的起始位置 (当前图片宽度 + 间隔)
        x_offset += im.width + padding

    return new_im


# ===========================================================================
#   环境检查与导入
# ===========================================================================
try:
    from unet import Unet
except ImportError:
    st.error("❌ 无法导入 'unet' 模块。请确保此文件与 'unet.py' 在同一个文件夹内。")
    st.stop()

# ===========================================================================
#   页面配置
# ===========================================================================
st.set_page_config(
    page_title="光伏分割深度学习模型对比与预测平台",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ 光伏分割深度学习模型对比与预测平台")

# ===========================================================================
#   State 初始化
# ===========================================================================
if 'predicted_image' not in st.session_state:
    st.session_state.predicted_image = None
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'viewer_page' not in st.session_state:
    st.session_state.viewer_page = 1

# ===========================================================================
#   【核心配置】默认路径设置
# ===========================================================================
DEFAULT_MODEL_A = r"D:\006–马少博-基于深度学习的光伏设施智能识别-2025-P01-A\2.源代码\UNet_Demo\model_data\Mit-B5-UNet-Champion-92-32.pth"
DEFAULT_MODEL_B = r"D:\006–马少博-基于深度学习的光伏设施智能识别-2025-P01-A\2.源代码\UNet_Demo\model_data\ResNet50-UNet-Champion_model_91.83.pth"
DEFAULT_INPUT_DIR = r"D:\006–马少博-基于深度学习的光伏设施智能识别-2025-P01-A\2.源代码\UNet_Demo\0train\images"
DEFAULT_MASK_DIR = r"D:\006–马少博-基于深度学习的光伏设施智能识别-2025-P01-A\2.源代码\UNet_Demo\0train\mask"
DEFAULT_OUTPUT_DIR = r"D:\006–马少博-基于深度学习的光伏设施智能识别-2025-P01-A\2.源代码\UNet_Demo\img_out"
DEFAULT_EVAL_DIR = r"D:\006–马少博-基于深度学习的光伏设施智能识别-2025-P01-A\2.源代码\UNet_Demo\eval_result"

# 批量注册 Session State 变量
defaults = {
    'model_a_path': DEFAULT_MODEL_A,
    'model_b_path': DEFAULT_MODEL_B,
    'comparison_mask_dir': DEFAULT_MASK_DIR,
    'path_batch_a': DEFAULT_MODEL_A,
    'path_batch_b': DEFAULT_MODEL_B,
    'in_batch_in': DEFAULT_INPUT_DIR,
    'in_batch_out': DEFAULT_OUTPUT_DIR,
    'eval_out_dir': DEFAULT_EVAL_DIR,
    'path_single': DEFAULT_MODEL_A,
    'single_save_path': os.path.join(DEFAULT_OUTPUT_DIR, "single"),
    'batch_model_path': DEFAULT_MODEL_A,
    'viewer_img_dir': DEFAULT_INPUT_DIR,
    'viewer_res_dir': DEFAULT_OUTPUT_DIR
}

for key, val in defaults.items():
    if key not in st.session_state or not st.session_state[key]:
        st.session_state[key] = val

# ===========================================================================
#   侧边栏：配置
# ===========================================================================
st.sidebar.header("🛠️ 工作模式")

app_mode = st.sidebar.radio(
    "请选择功能",
    ["⚔️ 多模型对比 (单张)", "📊 批量对比评测 (mIoU)", "🖼️ 单张预测", "📂 批量预测", "👀 结果对比浏览"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 通用配置")

mix_type = st.sidebar.selectbox(
    "可视化样式 (Mix Type)",
    options=[0, 1, 2],
    index=1,
    format_func=lambda x: {0: "0: 原图+Mask半透明 (IoU不准)", 1: "1: 仅Mask(纯色) [推荐]", 2: "2: 仅原图 (无法计算IoU)"}[x]
)
if mix_type != 1:
    st.sidebar.info("💡 提示：如需准确计算 IoU，请将可视化样式选择为 '1: 仅Mask'。")

st.sidebar.markdown("---")
if st.sidebar.button("🔴 停止并退出系统", type="primary"):
    st.sidebar.warning("正在关闭后台进程...")
    time.sleep(0.5)
    os._exit(0)

# ===========================================================================
#   【新增】作者信息展示
# ===========================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍🎓 作者信息")
st.sidebar.info("**马少博**\n\n🏫 中山大学 (Sun Yat-sen University)")

# ===========================================================================
#   模型加载函数
# ===========================================================================
SUPPORTED_BACKBONES = ["mit_b5", "resnet50", "vgg", "mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4"]


@st.cache_resource
def load_unet_model(path, mode_type, backbone_name):
    if not path or not os.path.exists(path):
        return None, f"文件不存在: {path}"
    try:
        model = Unet(
            model_path=path,
            backbone=backbone_name,
            num_classes=2,
            input_shape=[512, 512],
            mix_type=mode_type
        )
        return model, "Success"
    except Exception as e:
        return None, str(e)


def try_load_mask(filename, mask_dir):
    """尝试加载同名Mask"""
    if not mask_dir or not os.path.exists(mask_dir):
        return None, "Mask目录未设置"

    exact_path = os.path.join(mask_dir, filename)
    if os.path.exists(exact_path):
        return Image.open(exact_path), "Loaded"

    name_no_ext = os.path.splitext(filename)[0]
    for ext in ['.png', '.jpg', '.bmp', '.tif']:
        test_path = os.path.join(mask_dir, name_no_ext + ext)
        if os.path.exists(test_path):
            return Image.open(test_path), "Loaded"

    return None, "Not Found"


# ===========================================================================
#   功能模块 1: 多模型对比 (单张)
# ===========================================================================
if app_mode == "⚔️ 多模型对比 (单张)":
    st.subheader("⚔️ 多模型异构对比 & IoU 评测 (单张)")

    with st.expander("🛠️ 模型与标签路径配置", expanded=True):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.caption("🟢 模型 A")
            backbone_a = st.selectbox("主干网络 A", SUPPORTED_BACKBONES, index=0, key="bb_a")
            c1, c2 = st.columns([5, 1])
            with c1: st.text_input("权重路径 A", key="model_a_path")
            with c2: st.button("📂", key="btn_sel_a", on_click=on_click_select_file, args=("model_a_path",))

        with col_m2:
            st.caption("🔵 模型 B")
            backbone_b = st.selectbox("主干网络 B", SUPPORTED_BACKBONES, index=1, key="bb_b")
            c3, c4 = st.columns([5, 1])
            with c3: st.text_input("权重路径 B", key="model_b_path")
            with c4: st.button("📂", key="btn_sel_b", on_click=on_click_select_file, args=("model_b_path",))

        st.markdown("---")
        st.caption("🏁 真实标签 (Ground Truth) - 用于计算 IoU")
        cm1, cm2 = st.columns([8, 1])
        with cm1: st.text_input("标签文件夹路径", key="comparison_mask_dir")
        with cm2: st.button("📂", key="btn_sel_mask", on_click=on_click_select_folder, args=("comparison_mask_dir",))

    uploaded_file = st.file_uploader("📂 上传测试图片", type=["jpg", "png", "bmp", "tif"])

    if uploaded_file:
        if st.button("🚀 开始对比预测", type="primary", use_container_width=True):
            if not st.session_state.model_a_path or not st.session_state.model_b_path:
                st.error("请确保两个模型的路径都已填写！")
            else:
                image_orig = Image.open(uploaded_file)
                mask_img, _ = try_load_mask(uploaded_file.name, st.session_state.comparison_mask_dir)

                # Mask Display Logic
                mask_display = None
                if mask_img:
                    mask_arr = np.array(mask_img)
                    if len(mask_arr.shape) == 2 and np.max(mask_arr) <= 1:
                        mask_display = Image.fromarray(mask_arr * 255)
                    else:
                        mask_display = mask_img

                col_res1, col_res2, col_res3, col_res4 = st.columns(4)

                with col_res1:
                    st.image(image_orig, caption="原始图片", use_container_width=True)
                with col_res2:
                    if mask_display:
                        st.image(mask_display, caption="✅ 真实标签 (GT)", use_container_width=True)
                    else:
                        st.warning("无标签")

                # Model A
                with col_res3:
                    with st.spinner(f'{backbone_a}...'):
                        model_a, msg_a = load_unet_model(st.session_state.model_a_path, mix_type, backbone_a)
                    if model_a:
                        res_a = model_a.detect_image(image_orig.copy(), count=False, name_classes=["bg", "pv"])
                        st.image(res_a, caption=f"🟢 模型 A ({backbone_a})", use_container_width=True)
                        if mask_img and mix_type == 1:
                            iou_a, _ = calculate_iou(mask_img, res_a)
                            st.metric("IoU 得分", f"{iou_a * 100:.2f}%")
                    else:
                        st.error("加载失败")

                # Model B
                with col_res4:
                    with st.spinner(f'{backbone_b}...'):
                        model_b, msg_b = load_unet_model(st.session_state.model_b_path, mix_type, backbone_b)
                    if model_b:
                        res_b = model_b.detect_image(image_orig.copy(), count=False, name_classes=["bg", "pv"])
                        st.image(res_b, caption=f"🔵 模型 B ({backbone_b})", use_container_width=True)
                        if mask_img and mix_type == 1:
                            iou_b, _ = calculate_iou(mask_img, res_b)
                            delta = 0
                            if 'iou_a' in locals(): delta = iou_b - iou_a
                            st.metric("IoU 得分", f"{iou_b * 100:.2f}%", delta=f"{delta * 100:.2f}%")
                    else:
                        st.error("加载失败")

# ===========================================================================
#   功能模块 2: 批量对比评测 (mIoU)
# ===========================================================================
elif app_mode == "📊 批量对比评测 (mIoU)":
    st.subheader("📊 全局批量评测 (mIoU Report)")
    st.markdown("遍历图片文件夹，对比两个模型性能，生成 **CSV 评测报告** 和 **可视化对比图**。")

    col_conf1, col_conf2 = st.columns(2)

    with col_conf1:
        st.info("🟢 模型 A 配置")
        bb_batch_a = st.selectbox("主干网络 A", SUPPORTED_BACKBONES, index=0, key="bb_batch_a")
        ca1, ca2 = st.columns([5, 1])
        with ca1: st.text_input("权重路径 A", key="path_batch_a")
        with ca2: st.button("📂", key="btn_path_a", on_click=on_click_select_file, args=("path_batch_a",))

    with col_conf2:
        st.info("🔵 模型 B 配置")
        bb_batch_b = st.selectbox("主干网络 B", SUPPORTED_BACKBONES, index=1, key="bb_batch_b")
        cb1, cb2 = st.columns([5, 1])
        with cb1: st.text_input("权重路径 B", key="path_batch_b")
        with cb2: st.button("📂", key="btn_path_b", on_click=on_click_select_file, args=("path_batch_b",))

    st.markdown("#### 📂 输入与输出")

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.text_input("输入图片文件夹", key="in_batch_in")
        st.button("📂 选择图片目录", key="btn_b_in", on_click=on_click_select_folder, args=("in_batch_in",))
    with c_d2:
        st.text_input("真实标签 (Mask) 文件夹", key="comparison_mask_dir")
        st.button("📂 选择标签目录", key="btn_b_mask", on_click=on_click_select_folder, args=("comparison_mask_dir",))

    st.markdown("---")
    c_out1, c_out2 = st.columns([5, 1])
    with c_out1:
        st.text_input("📄 评测结果输出目录 (存放 .csv 报告和对比图)", key="eval_out_dir")
    with c_out2:
        st.write("")
        st.write("")
        st.button("📂", key="btn_eval_out", on_click=on_click_select_folder, args=("eval_out_dir",))

    # === 新增功能：实验模式与常规保存 ===
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        save_vis = st.checkbox("💾 常规保存：保存所有图片的可视化对比图 (速度较慢)", value=False)
    with col_opt2:
        # 实验模式：只保存 Top-K
        st.markdown("#### 🧪 实验模式 (双向优势筛选)")
        save_top_k = st.checkbox("启用：寻找 [模型A > 模型B] 和 [模型B > 模型A] 的 Top-K 样本", value=False)
        if save_top_k:
            top_k_num = st.number_input("选择每个方向保存前多少名 (Top-K)", min_value=1, value=50, step=10)

    st.markdown("---")

    if st.button("🚀 开始全量评测", type="primary", use_container_width=True):
        img_dir = st.session_state.in_batch_in
        mask_dir = st.session_state.comparison_mask_dir
        out_dir = st.session_state.eval_out_dir
        path_a = st.session_state.path_batch_a
        path_b = st.session_state.path_batch_b

        # 1. 基础检查
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            st.error("输入文件夹不存在，请检查！")
            st.stop()
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            st.error("模型文件路径错误，请检查！")
            st.stop()

        if not os.path.exists(out_dir): os.makedirs(out_dir)

        # 常规保存路径
        vis_out_dir = os.path.join(out_dir, "comparison_images_all")
        if save_vis and not os.path.exists(vis_out_dir): os.makedirs(vis_out_dir)

        # 2. 加载模型
        with st.status("正在初始化...", expanded=True) as status:
            st.write(f"正在加载模型 A: {bb_batch_a}...")
            model_a, msg_a = load_unet_model(path_a, 1, bb_batch_a)
            if not model_a:
                st.error(f"模型 A 失败: {msg_a}")
                st.stop()

            st.write(f"正在加载模型 B: {bb_batch_b}...")
            model_b, msg_b = load_unet_model(path_b, 1, bb_batch_b)
            if not model_b:
                st.error(f"模型 B 失败: {msg_b}")
                st.stop()

            status.update(label="模型加载完成，开始处理数据...", state="running")

            files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif'))]
            total = len(files)

            if total == 0:
                st.warning("图片文件夹为空！")
                st.stop()

            results = []
            progress_bar = st.progress(0)

            st.markdown("### 🎬 实时处理预览")
            preview_placeholder = st.empty()

            for i, f in enumerate(files):
                img_path = os.path.join(img_dir, f)

                try:
                    img = Image.open(img_path)
                    mask_gt, _ = try_load_mask(f, mask_dir)

                    if mask_gt is None: continue

                    res_a = model_a.detect_image(img.copy(), count=False, name_classes=["bg", "pv"])
                    iou_a, _ = calculate_iou(mask_gt, res_a)

                    res_b = model_b.detect_image(img.copy(), count=False, name_classes=["bg", "pv"])
                    iou_b, _ = calculate_iou(mask_gt, res_b)

                    # 记录数据 (diff = A - B，因为我们要找 A 更好的)
                    diff_val = iou_a - iou_b
                    results.append({
                        "Image": f,
                        "IoU_A": round(iou_a, 4),
                        "IoU_B": round(iou_b, 4),
                        "Diff(A-B)": round(diff_val, 4)
                    })

                    # 准备可视化素材 (预览或常规保存用)
                    mask_arr = np.array(mask_gt)
                    if len(mask_arr.shape) == 2 and np.max(mask_arr) <= 1:
                        mask_vis = Image.fromarray(mask_arr * 255)
                    else:
                        mask_vis = mask_gt.convert("L")

                    # 实时预览 (降低分辨率提高速度)
                    preview_size = (300, 300)
                    imgs_preview = [
                        img.resize(preview_size),
                        mask_vis.resize(preview_size).convert("RGB"),
                        res_a.resize(preview_size),
                        res_b.resize(preview_size)
                    ]
                    labels_prev = ["Orig", "GT", f"A:{iou_a:.1%}", f"B:{iou_b:.1%}"]
                    # 增加间隔
                    preview_vis = concat_images_horizontal(imgs_preview, labels_prev, padding=20)
                    preview_placeholder.image(preview_vis, caption=f"Processing: {f}", use_container_width=True)

                    # 常规保存 (如果勾选)
                    if save_vis:
                        save_size = (512, 512)
                        imgs_save = [
                            img.resize(save_size),
                            mask_vis.resize(save_size).convert("RGB"),
                            res_a.resize(save_size),
                            res_b.resize(save_size)
                        ]
                        labels_save = ["Original", "GT", f"Model A ({iou_a:.2%})", f"Model B ({iou_b:.2%})"]
                        save_vis_img = concat_images_horizontal(imgs_save, labels_save, padding=20)
                        save_vis_img.save(os.path.join(vis_out_dir, f))

                except Exception as e:
                    print(f"Error {f}: {e}")

                progress_bar.progress((i + 1) / total)

            status.update(label="基础评测完成，正在整理数据...", state="complete")

        if len(results) > 0:
            df = pd.DataFrame(results)
            csv_path = os.path.join(out_dir, "evaluation_report.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            miou_a = df["IoU_A"].mean()
            miou_b = df["IoU_B"].mean()

            st.divider()
            st.success(f"✅ 评测已完成！")

            # --- 实验模式：后处理筛选 (A Wins & B Wins) ---
            if save_top_k:
                st.subheader("🧪 实验模式：生成双向优势样本")

                # 1. 筛选 A 优于 B (Diff > 0)
                df_a_wins = df[df['Diff(A-B)'] > 0].sort_values(by='Diff(A-B)', ascending=False).head(top_k_num)

                # 2. 筛选 B 优于 A (Diff < 0)
                df_b_wins = df[df['Diff(A-B)'] < 0].sort_values(by='Diff(A-B)', ascending=True).head(
                    top_k_num)  # 升序，因为是负数


                # 辅助函数：生成并保存图片
                def save_top_k_images(df_subset, subdir_name, title_prefix):
                    if len(df_subset) == 0:
                        st.warning(f"⚠️ 未找到 {title_prefix} 的样本。")
                        return

                    target_dir = os.path.join(out_dir, subdir_name)
                    if not os.path.exists(target_dir): os.makedirs(target_dir)

                    st.write(f"正在保存 {title_prefix} 的图片到: {subdir_name} ...")
                    k_bar = st.progress(0)

                    for idx, (index, row) in enumerate(df_subset.iterrows()):
                        f_name = row['Image']
                        img_path = os.path.join(img_dir, f_name)

                        try:
                            img = Image.open(img_path)
                            mask_gt, _ = try_load_mask(f_name, mask_dir)
                            res_a = model_a.detect_image(img.copy(), count=False, name_classes=["bg", "pv"])
                            res_b = model_b.detect_image(img.copy(), count=False, name_classes=["bg", "pv"])

                            mask_arr = np.array(mask_gt)
                            if len(mask_arr.shape) == 2 and np.max(mask_arr) <= 1:
                                mask_vis = Image.fromarray(mask_arr * 255)
                            else:
                                mask_vis = mask_gt.convert("L")

                            target_size = (512, 512)
                            imgs_concat = [
                                img.resize(target_size),
                                mask_vis.resize(target_size).convert("RGB"),
                                res_a.resize(target_size),
                                res_b.resize(target_size)
                            ]
                            labels = [
                                f"Rank #{idx + 1}",
                                "Ground Truth",
                                f"Model A (IoU {row['IoU_A']:.2%})",
                                f"Model B (IoU {row['IoU_B']:.2%})"
                            ]
                            final_img = concat_images_horizontal(imgs_concat, labels, padding=20)

                            # 格式化文件名：排名_差异值_原文件名
                            diff_percent = row['Diff(A-B)'] * 100
                            save_name = f"rank_{idx + 1:02d}_diff_{diff_percent:+.1f}%_{f_name}"
                            final_img.save(os.path.join(target_dir, save_name))

                        except Exception as e:
                            print(f"Error filtering {f_name}: {e}")

                        k_bar.progress((idx + 1) / len(df_subset))
                    st.success(f"✅ {title_prefix} 保存完成！")


                # 执行保存
                save_top_k_images(df_a_wins, "model_a_wins_top_k", "模型 A 优于 B")
                save_top_k_images(df_b_wins, "model_b_wins_top_k", "模型 B 优于 A")

                # 展示数据表
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🟢 A 优于 B (Top-K)")
                    st.dataframe(df_a_wins)
                with c2:
                    st.markdown("#### 🔵 B 优于 A (Top-K)")
                    st.dataframe(df_b_wins)

            # --- 最终报告 ---
            st.subheader("🏆 最终成绩单")
            c1, c2, c3 = st.columns(3)
            c1.metric("样本数量", len(df))
            c2.metric(f"模型 A ({bb_batch_a}) mIoU", f"{miou_a:.2%}")
            c3.metric(f"模型 B ({bb_batch_b}) mIoU", f"{miou_b:.2%}", delta=f"{miou_b - miou_a:.2%}")

            st.line_chart(df[["IoU_A", "IoU_B"]])

            if st.button("📂 打开结果保存目录"):
                if os.name == 'nt': os.startfile(out_dir)

# ===========================================================================
#   功能模块 3: 单张预测
# ===========================================================================
elif app_mode == "🖼️ 单张预测":
    st.subheader("🖼️ 单张快速预测")

    col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 4, 1])
    with col_cfg1:
        single_backbone = st.selectbox("选择主干网络", SUPPORTED_BACKBONES, index=0)
    with col_cfg2:
        path_single = st.text_input("模型权重路径", key="path_single")
    with col_cfg3:
        st.button("📂", key="btn_sel_single", on_click=on_click_select_file, args=("path_single",))

    if st.session_state.path_single:
        unet_model, status = load_unet_model(st.session_state.path_single, mix_type, single_backbone)
        if unet_model is None:
            st.error(f"❌ 模型加载失败: {status}")
            st.stop()

    uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "bmp", "tif"])

    if uploaded_file:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="原始输入", use_container_width=True)
            if st.button("🚀 开始预测", type="primary", use_container_width=True):
                st.session_state.predicted_image = unet_model.detect_image(image, count=False,
                                                                           name_classes=["_bg_", "PV"])
                st.session_state.prediction_done = True
                st.rerun()

        if st.session_state.prediction_done and st.session_state.predicted_image:
            with col2:
                st.image(st.session_state.predicted_image, caption=f"预测结果 ({single_backbone})",
                         use_container_width=True)
                if st.session_state.comparison_mask_dir and mix_type == 1:
                    mask_img, _ = try_load_mask(uploaded_file.name, st.session_state.comparison_mask_dir)
                    if mask_img:
                        iou, _ = calculate_iou(mask_img, st.session_state.predicted_image)
                        st.success(f"📈 IoU Score: {iou * 100:.2f}%")

            with st.expander("💾 保存结果"):
                sc1, sc2 = st.columns([4, 1])
                with sc1:
                    st.text_input("保存目录", key="single_save_path")
                with sc2:
                    st.button("📂", key="btn_save_sel", on_click=on_click_select_folder, args=("single_save_path",))

                if st.button("确认保存"):
                    save_dir = st.session_state.single_save_path
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    save_p = os.path.join(save_dir, uploaded_file.name)
                    st.session_state.predicted_image.save(save_p)
                    st.success(f"已保存: {save_p}")
                    if os.name == 'nt': subprocess.Popen(f'explorer /select,"{save_p}"')

# ===========================================================================
#   功能模块 4: 批量预测 (原批量处理)
# ===========================================================================
elif app_mode == "📂 批量预测":
    st.subheader("📂 文件夹批量预测")

    # 修复：添加文件选择按钮
    c_b1, c_b2 = st.columns([2, 5])
    with c_b1:
        batch_backbone = st.selectbox("选择主干网络", SUPPORTED_BACKBONES, index=0, key="bb_batch")
    with c_b2:
        col_p, col_btn = st.columns([5, 1])
        with col_p:
            st.text_input("模型权重路径", key="batch_model_path")
        with col_btn:
            st.button("📂", key="btn_batch_mod", on_click=on_click_select_file, args=("batch_model_path",))

    path_batch = st.session_state.batch_model_path
    if path_batch:
        unet_model, status = load_unet_model(path_batch, mix_type, batch_backbone)
        if not unet_model:
            st.error(f"模型无效: {status}")
            st.stop()

    c1, c2 = st.columns([6, 1])
    with c1:
        st.text_input("输入文件夹", key="in_batch_in")
    with c2:
        st.button("📂", key="b_in", on_click=on_click_select_folder, args=("in_batch_in",))

    c3, c4 = st.columns([6, 1])
    with c3:
        st.text_input("输出文件夹", key="in_batch_out")
    with c4:
        st.button("📂", key="b_out", on_click=on_click_select_folder, args=("in_batch_out",))

    if st.button("🚀 开始批量任务", type="primary"):
        input_dir = st.session_state.in_batch_in
        output_dir = st.session_state.in_batch_out

        if not os.path.exists(input_dir):
            st.error("输入文件夹不存在")
        else:
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif'))]

            bar = st.progress(0)
            status = st.empty()

            # 【新增】实时预览区域
            st.markdown("### 🎬 实时处理预览")
            preview_placeholder = st.empty()

            for i, f in enumerate(files):
                try:
                    img = Image.open(os.path.join(input_dir, f))
                    res = unet_model.detect_image(img.copy(), count=False, name_classes=["bg", "pv"])  # copy以防万一
                    res.save(os.path.join(output_dir, f))

                    # 生成预览图：原图 | 预测结果
                    target_size = (300, 300)
                    imgs_to_concat = [
                        img.resize(target_size),
                        res.resize(target_size)
                    ]
                    labels = ["Original", "Prediction"]
                    final_vis = concat_images_horizontal(imgs_to_concat, labels, padding=20)

                    preview_placeholder.image(final_vis, caption=f"正在处理: {f}", use_container_width=True)

                except Exception as e:
                    print(f"Error {f}: {e}")

                bar.progress((i + 1) / len(files))
                status.text(f"Processing: {i + 1}/{len(files)}")

            st.success("完成！")
            if st.button("打开输出目录"): os.startfile(output_dir)

# ===========================================================================
#   功能模块 5: 结果对比浏览 (新功能)
# ===========================================================================
elif app_mode == "👀 结果对比浏览":
    st.subheader("👀 预测结果对比浏览器")
    st.markdown("选择原图文件夹和预测结果文件夹，并排查看对比效果。")

    # 1. 文件夹选择
    c_img, c_res = st.columns(2)
    with c_img:
        st.text_input("📂 原图文件夹 (Original)", key="viewer_img_dir")
        st.button("选择原图目录", key="btn_v_img", on_click=on_click_select_folder, args=("viewer_img_dir",))
    with c_res:
        st.text_input("📂 预测结果文件夹 (Prediction)", key="viewer_res_dir")
        st.button("选择结果目录", key="btn_v_res", on_click=on_click_select_folder, args=("viewer_res_dir",))

    st.markdown("---")

    img_dir = st.session_state.viewer_img_dir
    res_dir = st.session_state.viewer_res_dir

    if os.path.exists(img_dir) and os.path.exists(res_dir):
        try:
            img_files = set([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif'))])
            res_files = set([f for f in os.listdir(res_dir) if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif'))])

            common_files = sorted(list(img_files & res_files))
            total_files = len(common_files)

            if total_files == 0:
                st.warning("⚠️ 两个文件夹中没有找到同名图片，请检查文件名是否一致。")
            else:
                # 3. 分页控制
                ITEMS_PER_PAGE = 20
                total_pages = math.ceil(total_files / ITEMS_PER_PAGE)


                # 回调函数
                def prev_page():
                    if st.session_state.viewer_page > 1:
                        st.session_state.viewer_page -= 1


                def next_page():
                    if st.session_state.viewer_page < total_pages:
                        st.session_state.viewer_page += 1


                # 分页导航栏
                c_p1, c_p2, c_p3 = st.columns([1, 3, 1])
                with c_p1:
                    st.button("⬅️ 上一页", on_click=prev_page)
                with c_p2:
                    # 页码显示 (绑定 Session State)
                    st.number_input(
                        f"页码 (共 {total_pages} 页, {total_files} 张图片)",
                        min_value=1, max_value=total_pages,
                        key="viewer_page"  # 直接绑定 Key
                    )
                with c_p3:
                    st.button("下一页 ➡️", on_click=next_page)

                # 4. 显示网格
                start_idx = (st.session_state.viewer_page - 1) * ITEMS_PER_PAGE
                end_idx = min(start_idx + ITEMS_PER_PAGE, total_files)
                current_batch = common_files[start_idx:end_idx]

                st.markdown("---")

                # 使用 Grid 布局展示
                cols = st.columns(2)

                for i, filename in enumerate(current_batch):
                    col_idx = i % 2
                    with cols[col_idx]:
                        # 加载图片
                        f_img = os.path.join(img_dir, filename)
                        f_res = os.path.join(res_dir, filename)

                        try:
                            im_orig = Image.open(f_img)
                            im_pred = Image.open(f_res)

                            # 拼接显示
                            target_h = 300
                            w_orig = int(im_orig.width * (target_h / im_orig.height))
                            w_pred = int(im_pred.width * (target_h / im_pred.height))

                            im_orig_r = im_orig.resize((w_orig, target_h))
                            im_pred_r = im_pred.resize((w_pred, target_h))

                            if im_pred_r.mode != 'RGB':
                                im_pred_r = im_pred_r.convert('RGB')

                            combined = concat_images_horizontal([im_orig_r, im_pred_r], labels=["原图", "预测"], padding=5)

                            st.image(combined, caption=filename, use_container_width=True)
                            st.markdown("---")

                        except Exception as e:
                            st.error(f"Error loading {filename}: {e}")

        except Exception as e:
            st.error(f"读取文件夹出错: {e}")
    else:
        st.info("👈 请先在上方选择两个文件夹以开始浏览。")