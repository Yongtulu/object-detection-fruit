"""
本地 FastAPI 推理服务 + 网页界面
==============================
启动:
    uvicorn app:app --reload

打开:
    http://localhost:8000/          网页界面（上传图片即检测）
    http://localhost:8000/docs      Swagger API 文档

接口:
    GET  /            网页界面
    GET  /health      健康检查
    GET  /info        模型信息（类别、权重）
    POST /detect      上传图片 -> 检测 + 计数 + 带框图
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse

from inference import detect_and_count, _model

app = FastAPI(title="Fruit Detection & Counting", version="2.0")


@app.get("/")
def index():
    """网页界面首页"""
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    """返回模型类别等信息，供前端展示"""
    return {
        "classes": list(_model.names.values()),
        "num_classes": len(_model.names),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
):
    """上传图片，返回检测计数 + 带框图(base64)"""
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = detect_and_count(tmp_path, conf=conf, iou=iou)
    finally:
        os.remove(tmp_path)
    return result
