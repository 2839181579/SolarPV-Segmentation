# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image
from unet import Unet
import os
from tqdm import tqdm
import shutil

if __name__ == "__main__":
    # ===========================================================================
    #   1. 【核心配置区域】 请在这里修改路径
    # ===========================================================================

    # 模式选择
    mode = "dir_predict"

    # 【重要】模型权重路径
    # 请将这里修改为你刚才验证过的那个 Epoch 200 的 .pth 文件路径
    # 举例: r"logs\loss_2025_11_30_xxxx\ep200-loss0.222-val_loss0.217.pth"
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "Mit-B5-UNet-Champion-92-32.pth")

    # 【重要】输入图片文件夹
    dir_origin_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "VOCdevkit", "VOC2007", "JPEGImages")

    # 【重要】结果保存文件夹 (会自动创建)
    # 建议加上 _MiT_B5 后缀以区分
    dir_save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "predictions_mitb5")

    # ===========================================================================
    #   2. 模型加载
    # ===========================================================================
    print(f"\n{'=' * 50}")
    print(f"🚀 正在初始化 MiT-B5-UNet 模型...")
    print(f"📂 加载权重: {model_path}")
    print(f"{'=' * 50}\n")

    # -------------------------------------------------------------------------#
    #   实例化 Unet 类
    #   这里通过参数覆盖 unet.py 里的默认设置，确保使用 B5 和 200轮的权重
    # -------------------------------------------------------------------------#
    unet = Unet(
        model_path=model_path,  # 强制指定权重路径
        backbone="mit_b5",  # 【核心】修改主干为 mit_b5
        num_classes=2,  # 类别数
        input_shape=[512, 512],  # 输入大小
        mix_type=1  # 0=原图+mask混合, 1=仅mask(纯色), 2=仅原图(没啥用)                           # 修改
    )

    # ===========================================================================
    #   3. 执行预测逻辑
    # ===========================================================================
    count = False
    name_classes = ["_background_", "PV"]  # 对应 class 0 和 class 1

    # ----------------------------------------------------------------------------------------------------------#
    #   dir_predict 模式：遍历文件夹进行预测
    # ----------------------------------------------------------------------------------------------------------#
    if mode == "dir_predict":
        import os
        from tqdm import tqdm
        import shutil

        img_names = os.listdir(dir_origin_path)

        # 路径检查与创建
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
            print(f"✅ 已创建输出目录: {dir_save_path}")

        # 创建一个用于存放原图备份的文件夹 (根据你之前的逻辑保留)
        # 逻辑是: 在输出目录的上级目录，建一个和输入目录同名的文件夹
        new_save_folder = os.path.join(os.path.dirname(dir_save_path), os.path.basename(dir_origin_path) + "_backup")
        if not os.path.exists(new_save_folder):
            os.makedirs(new_save_folder)

        print(f"\n📌 任务信息:")
        print(f"   输入目录: {dir_origin_path}")
        print(f"   输出目录: {dir_save_path}")
        print(f"   图片总数: {len(img_names)}")
        print(f"   开始处理...\n")

        processed_count = 0
        skipped_count = 0

        for img_name in tqdm(img_names, desc="Predicting"):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)

                try:
                    image = Image.open(image_path)

                    # 进行预测
                    r_image = unet.detect_image(image, count=count, name_classes=name_classes)

                    # ---------------------------------------------------
                    # 检查是否为纯黑 (没有预测出目标)
                    # ---------------------------------------------------
                    r_image_array = np.array(r_image)
                    if np.all(r_image_array == 0):
                        skipped_count += 1
                        # print(f"Skipping {img_name} (No PV detected)")
                        continue

                    # 保存预测结果图
                    save_name = os.path.join(dir_save_path, img_name)
                    # 注意：如果你的文件名需要特殊前缀，可以在这里加，例如 "res_" + img_name
                    r_image.save(save_name)
                    processed_count += 1

                    # ---------------------------------------------------
                    # 复制相关文件 (.pgw 地理信息文件)
                    # ---------------------------------------------------
                    # 1. 备份原图
                    shutil.copy(image_path, os.path.join(new_save_folder, img_name))

                    # 2. 复制 .pgw 文件 (如果存在)
                    pwg_file_name = os.path.splitext(img_name)[0] + '.pgw'
                    pwg_file_path = os.path.join(dir_origin_path, pwg_file_name)

                    if os.path.exists(pwg_file_path):
                        shutil.copy(pwg_file_path, os.path.join(dir_save_path, pwg_file_name))

                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
                    continue

        print(f"\n{'=' * 50}")
        print(f"✅ 处理完成!")
        print(f"   共检测到含目标图片: {processed_count} 张")
        print(f"   跳过纯背景图片: {skipped_count} 张")
        print(f"   结果保存在: {dir_save_path}")
        print(f"{'=' * 50}")

    # ----------------------------------------------------------------------------------------------------------#
    #   predict 模式：单张图片预测
    # ----------------------------------------------------------------------------------------------------------#
    elif mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    # ----------------------------------------------------------------------------------------------------------#
    #   video 模式：视频预测
    # ----------------------------------------------------------------------------------------------------------#
    elif mode == "video":
        video_path = 0
        video_save_path = ""
        video_fps = 25.0
        test_interval = 100

        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(unet.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval=100)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify=True, onnx_save_path="model_data/models.onnx")