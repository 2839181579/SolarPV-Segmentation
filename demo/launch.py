import os
import sys
import subprocess

# 1. 确定你的 Streamlit 界面文件名
# ui_filename = "可视化预测界面.py"
ui_filename = "可视化对比.py"

# 2. 获取当前文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(current_dir, ui_filename)

# 3. 检查文件是否存在
if not os.path.exists(script_path):
    print(f"❌ 错误：找不到文件 '{ui_filename}'")
    print(f"   请确保此脚本和 '{ui_filename}' 在同一个文件夹下。")
    input("按回车键退出...")
    sys.exit(1)

print(f"🚀 正在启动网页界面...")
print(f"📂 目标文件: {script_path}")
print("-" * 50)

# 4. 使用 subprocess 启动 (更稳定，解决中文路径和乱码问题)
# sys.executable: 自动获取当前 Python 解释器的路径
# cwd=current_dir: 强制将运行目录设置为当前文件夹，防止找不到 unet.py
try:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", ui_filename],
        cwd=current_dir,  # 【关键】确保运行目录正确
        check=True
    )
except KeyboardInterrupt:
    print("\n👋 程序已关闭")
except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    input("按回车键退出...")
