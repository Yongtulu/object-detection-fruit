"""
推理 + 计数逻辑
==============
封装两个函数：
  - detect_and_count : 返回计数与检测细节（JSON 用）
  - detect_to_image  : 返回画好框的图片（base64，网页显示用）

被 app.py 调用。
"""

import base64
from collections import Counter
from io import BytesIO

from PIL import Image
from ultralytics import YOLO

# 你本机训练好的权重路径（resume 后 Ultralytics 多套了一层 detect/）
WEIGHTS = "runs/detect/runs/fruit_v1/weights/best.pt"

# 模型只加载一次（模块级），避免每次请求重复加载
_model = YOLO(WEIGHTS)


def detect_and_count(image_path: str, conf: float = 0.25, iou: float = 0.45) -> dict:
    """
    对一张图做检测并计数，同时返回画好框的图片(base64)。

    返回:
        dict: total / by_class / detections / image_base64
    """
    results = _model.predict(image_path, conf=conf, iou=iou, verbose=False)[0]
    names = _model.names

    counts = Counter()
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        confidence = float(box.conf[0])
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]

        counts[label] += 1
        detections.append({
            "label": label,
            "conf": round(confidence, 3),
            "bbox": xyxy,
        })

    # 生成带框可视化图：results.plot() 返回 BGR 的 numpy 数组
    plotted_bgr = results.plot()
    # BGR -> RGB
    plotted_rgb = plotted_bgr[:, :, ::-1]
    img = Image.fromarray(plotted_rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "total": sum(counts.values()),
        "by_class": dict(counts),
        "detections": detections,
        "image_base64": image_base64,
    }


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) > 1:
        out = detect_and_count(sys.argv[1])
        # 命令行下不打印超长的 base64
        out_print = {k: v for k, v in out.items() if k != "image_base64"}
        print(json.dumps(out_print, indent=2, ensure_ascii=False))
    else:
        print("用法: python inference.py <图片路径>")
