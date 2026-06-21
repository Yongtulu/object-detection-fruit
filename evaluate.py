"""
模型评估 + conf/iou 调参脚本
============================
1. 在测试集上算 mAP 指标
2. 对测试图做可视化预测（带框结果存盘）
3. 做一个 conf/iou 网格扫描，观察检测数量怎么变化
   —— 这是水果计数任务的核心练手点

运行: python evaluate.py
"""

from ultralytics import YOLO

WEIGTHS = "runs/detect/runs/fruit_v1/weights/best.pt"


def basic_eval():
    model = YOLO(WEIGTHS)

    # 测试集指标
    metrics = model.val(data="data/data.yaml", split="test")
    print("\n========== 测试集指标 ==========")
    print(f"mAP50    : {metrics.box.map50:.3f}")
    print(f"mAP50-95 : {metrics.box.map:.3f}")

    # 可视化预测，结果存到 runs/.../predict
    model.predict(
        source="data/test/images",
        save=True,
        conf=0.25,
        iou=0.45,
    )
    print("可视化结果已保存到 runs/ 下的 predict 目录")


def conf_iou_sweep(sample_image: str):
    """
    对单张图扫描不同 conf / iou，打印检测到的目标总数。
    用于直观理解：
      - conf 越高 -> 越严格 -> 检测数越少（可能漏检）
      - iou 越低  -> NMS越激进 -> 重叠目标更易被合并（数量减少）
    """
    model = YOLO(WEIGTHS)
    print(f"\n========== conf/iou 扫描: {sample_image} ==========")
    print(f"{'conf':>6} | {'iou':>5} | {'检测数':>6}")
    print("-" * 26)
    for conf in (0.10, 0.25, 0.40, 0.60):
        for iou in (0.30, 0.45, 0.60):
            r = model.predict(sample_image, conf=conf, iou=iou, verbose=False)[0]
            n = len(r.boxes)
            print(f"{conf:>6.2f} | {iou:>5.2f} | {n:>6d}")


if __name__ == "__main__":
    basic_eval()

    # 换成你测试集里任意一张图的真实路径
    # conf_iou_sweep("data/test/images/your_image.jpg")
