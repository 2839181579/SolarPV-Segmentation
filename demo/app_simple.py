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
# ===========================================================================
def concat_images_horizontal(images, labels=None):
    """将多张图片横向拼接，并添加文字标签"""
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    # 如果有标签，增加顶部文字区域高度
    text_height = 40 if labels else 0

    new_im = Image.new('RGB', (total_width, max_height + text_height), (255, 255, 255))

    x_offset = 0
    draw = ImageDraw.Draw(new_im)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    for idx, im in enumerate(images):
        new_im.paste(im, (x_offset, text_height))

        if labels and idx < len(labels):
            label_text = labels[idx]
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = x_offset + (im.width - text_w) // 2
            draw.text((text_x, 5), label_text, fill=(0, 0, 0), font=font)

        x_offset += im.width

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

# ===========================================================================
#   【核心配置】默认路径设置
# ===========================================================================
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. 默认模型路径
DEFAULT_MODEL_A = os.path.join(PROJECT_ROOT, "weights", "Mit-B5-UNet-Champion-92-32.pth")
DEFAULT_MODEL_B = os.path.join(PROJECT_ROOT, "weights", "ResNet50-UNet-Champion_model_91.83.pth")

# 2. 默认文件夹路径
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2007", "JPEGImages")
DEFAULT_MASK_DIR = os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2007", "SegmentationClass")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_EVAL_DIR = os.path.join(PROJECT_ROOT, "eval_result")

# 批量注册 Session State 变量
defaults = {
    # 多模型对比模式
    'model_a_path': DEFAULT_MODEL_A,
    'model_b_path': DEFAULT_MODEL_B,
    'comparison_mask_dir': DEFAULT_MASK_DIR,

    # 批量评测模式
    'path_batch_a': DEFAULT_MODEL_A,
    'path_batch_b': DEFAULT_MODEL_B,
    'in_batch_in': DEFAULT_INPUT_DIR,  # 批量预测/评测输入
    'in_batch_out': DEFAULT_OUTPUT_DIR,  # 批量预测输出
    'eval_out_dir': DEFAULT_EVAL_DIR,  # 评测报告输出

    # 单张预测模式
    'path_single': DEFAULT_MODEL_A,
    'single_save_path': os.path.join(DEFAULT_OUTPUT_DIR, "single"),

    # 批量预测模式 (独立变量，但复用默认值)
    'batch_model_path': DEFAULT_MODEL_A,
}

for key, val in defaults.items():
    # 逻辑修改：如果key不存在 或者 key对应的值为空字符串，则应用默认值
    if key not in st.session_state or not st.session_state[key]:
        st.session_state[key] = val

# ===========================================================================
#   侧边栏：配置
# ===========================================================================
st.sidebar.header("🛠️ 工作模式")

# 修改菜单名称："批量处理" -> "批量预测"
app_mode = st.sidebar.radio(
    "请选择功能",
    ["⚔️ 多模型对比 (单张)", "📊 批量对比评测 (mIoU)", "🖼️ 单张预测", "📂 批量预测"]
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

    save_vis = st.checkbox("💾 同时保存可视化对比图 (会生成四合一拼图，处理速度会变慢)", value=False)

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

        vis_out_dir = os.path.join(out_dir, "comparison_images")
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

                    results.append({
                        "Image": f,
                        "IoU_A": round(iou_a, 4),
                        "IoU_B": round(iou_b, 4),
                        "Diff": round(iou_b - iou_a, 4)
                    })

                    mask_arr = np.array(mask_gt)
                    if len(mask_arr.shape) == 2 and np.max(mask_arr) <= 1:
                        mask_vis = Image.fromarray(mask_arr * 255)
                    else:
                        mask_vis = mask_gt.convert("L")

                    target_size = (300, 300)
                    imgs_to_concat = [
                        img.resize(target_size),
                        mask_vis.resize(target_size).convert("RGB"),
                        res_a.resize(target_size),
                        res_b.resize(target_size)
                    ]
                    labels = [
                        "Original",
                        "Ground Truth",
                        f"A: {iou_a:.1%}",
                        f"B: {iou_b:.1%}"
                    ]
                    final_vis = concat_images_horizontal(imgs_to_concat, labels)

                    preview_placeholder.image(final_vis, caption=f"正在处理: {f}", use_container_width=True)

                    if save_vis:
                        save_size = (512, 512)
                        imgs_save = [
                            img.resize(save_size),
                            mask_vis.resize(save_size).convert("RGB"),
                            res_a.resize(save_size),
                            res_b.resize(save_size)
                        ]
                        save_vis_img = concat_images_horizontal(imgs_save, labels)
                        save_vis_img.save(os.path.join(vis_out_dir, f))

                except Exception as e:
                    print(f"Error {f}: {e}")

                progress_bar.progress((i + 1) / total)

            status.update(label="评测完成！", state="complete")

        if len(results) > 0:
            df = pd.DataFrame(results)
            csv_path = os.path.join(out_dir, "evaluation_report.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            miou_a = df["IoU_A"].mean()
            miou_b = df["IoU_B"].mean()

            st.divider()
            st.success(f"✅ 评测已完成！")
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
                    final_vis = concat_images_horizontal(imgs_to_concat, labels)

                    preview_placeholder.image(final_vis, caption=f"正在处理: {f}", use_container_width=True)

                except Exception as e:
                    print(f"Error {f}: {e}")

                bar.progress((i + 1) / len(files))
                status.text(f"Processing: {i + 1}/{len(files)}")

            st.success("完成！")
            if st.button("打开输出目录"): os.startfile(output_dir)