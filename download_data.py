"""
数据集自动下载脚本
==================
自动从 GitHub 克隆公开的水果检测数据集（无需任何 API key）。

数据集: lightly-ai/dataset_fruits_detection
  - 6 类: Apple, Banana, Grape, Orange, Pineapple, Watermelon
  - 8479 张图 (train 7108 / valid 914 / test 457)
  - 已是 YOLO 格式, 640x640, 自带 data.yaml
  - 来源: Kaggle (Lakshay Tyagi), CC 许可

运行: python download_data.py

完成后数据位于 data/ 目录，data.yaml 路径已自动配好，
可直接 python train.py。
"""

import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/lightly-ai/dataset_fruits_detection.git"
TMP_DIR = "_tmp_fruit_dataset"
DATA_DIR = "data"


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    # 已经下载过就跳过
    if os.path.exists(os.path.join(DATA_DIR, "train", "images")):
        print("数据集似乎已存在 (data/train/images)。如需重下，请先删除 data/ 下的 train/valid/test。")
        return

    # 1. 浅克隆到临时目录
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    print("正在克隆数据集（约 8000+ 张图，请耐心等待）...")
    run(["git", "clone", "--depth", "1", REPO_URL, TMP_DIR])

    # 2. 把 train/valid/test 搬到 data/ 下
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in ("train", "valid", "test"):
        src = os.path.join(TMP_DIR, split)
        dst = os.path.join(DATA_DIR, split)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
        print(f"  已移动 {split}/ -> {dst}")

    # 3. 写入与本项目结构匹配的 data.yaml
    #    （路径相对于 data.yaml 所在的 data/ 目录）
    data_yaml = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 6\n"
        "names: ['Apple', 'Banana', 'Grape', 'Orange', 'Pineapple', 'Watermelon']\n"
    )
    with open(os.path.join(DATA_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(data_yaml)
    print(f"  已写入 {DATA_DIR}/data.yaml")

    # 4. 清理临时目录
    shutil.rmtree(TMP_DIR)

    # 5. 汇总
    def count(split):
        p = os.path.join(DATA_DIR, split, "images")
        return len(os.listdir(p)) if os.path.exists(p) else 0

    print("\n========== 下载完成 ==========")
    print(f"train: {count('train')} 张")
    print(f"valid: {count('valid')} 张")
    print(f"test : {count('test')} 张")
    print("类别 : Apple, Banana, Grape, Orange, Pineapple, Watermelon")
    print("\n现在可以运行: python train.py")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError:
        print("\n克隆失败。请检查网络，或手动执行:")
        print(f"  git clone --depth 1 {REPO_URL}")
        sys.exit(1)
