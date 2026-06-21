"""
水果检测模型训练脚本
====================
使用 Ultralytics YOLO11 在自定义水果数据集上训练目标检测模型。
本地 Apple Silicon (MPS) 运行。

运行: python train.py
"""

from ultralytics import YOLO


def main():
    # 加载预训练模型
    #   yolo11n.pt = nano，最快、最省内存，适合上手
    #   yolo11s.pt = small，精度更高，训练稍慢（跑通后可换这个对比）
    model = YOLO("yolo11n.pt")

    # 开始训练
    model.train(
        data="data/data.yaml",   # 数据集配置（Roboflow导出）
        epochs=50,               # 训练轮数；早停会在合适时提前结束
        imgsz=640,               # 输入图像尺寸
        batch=16,                # 批大小；显存/内存不够就调小到 8
        device="mps",            # Apple Silicon GPU；没有则改为 "cpu"
        project="runs",          # 输出根目录
        name="fruit_v1",         # 本次实验名
        patience=15,             # 早停：15轮无提升则停止
        mosaic=1.0,              # mosaic数据增强，对密集/小目标友好
        verbose=True,
    )

    # 训练结束后在验证集上评估
    metrics = model.val()
    print("\n========== 验证集结果 ==========")
    print(f"mAP50    : {metrics.box.map50:.3f}")
    print(f"mAP50-95 : {metrics.box.map:.3f}")
    print(f"权重保存在: runs/fruit_v1/weights/best.pt")


if __name__ == "__main__":
    main()
