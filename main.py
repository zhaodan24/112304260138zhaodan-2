from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image

from model import DigitCNN


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pth"
IMAGE_SIZE = 28
MNIST_MEAN = 0.131015
MNIST_STD = 0.308540
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CanvasPayload(BaseModel):
    data_url: str


app = FastAPI(title="CNN Handwritten Digit Recognizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model() -> DigitCNN:
    model = DigitCNN().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = load_model()


def shift_on_black(image: Image.Image, shift_x: int, shift_y: int) -> Image.Image:
    shifted = Image.new("L", image.size, 0)
    width, height = image.size
    src_left = max(0, -shift_x)
    src_top = max(0, -shift_y)
    src_right = min(width, width - shift_x)
    src_bottom = min(height, height - shift_y)
    dst_left = max(0, shift_x)
    dst_top = max(0, shift_y)
    if src_right > src_left and src_bottom > src_top:
        crop = image.crop((src_left, src_top, src_right, src_bottom))
        shifted.paste(crop, (dst_left, dst_top))
    return shifted


def preprocess_image(image: Image.Image) -> Image.Image:
    pil = image.convert("RGBA")
    white_bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
    pil = Image.alpha_composite(white_bg, pil).convert("L")

    arr = np.asarray(pil).astype(np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr[arr < 20] = 0
    ys, xs = np.where(arr > 20)
    if len(xs) == 0 or len(ys) == 0:
        raise HTTPException(status_code=400, detail="没有检测到有效笔迹，请重新输入。")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    digit = Image.fromarray(np.clip(arr[y0 : y1 + 1, x0 : x1 + 1], 0, 255).astype(np.uint8))

    w, h = digit.size
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = digit.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    left = (IMAGE_SIZE - new_w) // 2
    top = (IMAGE_SIZE - new_h) // 2
    canvas.paste(digit, (left, top))

    centered = np.asarray(canvas).astype(np.float32)
    ys, xs = np.where(centered > 20)
    if len(xs) and len(ys):
        weights = centered[ys, xs]
        cx = float((xs * weights).sum() / weights.sum())
        cy = float((ys * weights).sum() / weights.sum())
        canvas = shift_on_black(canvas, int(round(IMAGE_SIZE / 2 - cx)), int(round(IMAGE_SIZE / 2 - cy)))
    return canvas


def predict_image(image: Image.Image) -> dict[str, object]:
    processed = preprocess_image(image)
    arr = np.asarray(processed).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).view(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    tensor = ((tensor - MNIST_MEAN) / MNIST_STD).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:3]
    prediction = int(top_indices[0])
    return {
        "prediction": prediction,
        "confidence": float(probs[prediction]),
        "top3": [{"digit": int(i), "probability": float(probs[i])} for i in top_indices],
        "probabilities": [{"digit": i, "probability": float(probs[i])} for i in range(10)],
    }


def image_from_data_url(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
        return Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="画板图片解析失败。") from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/api/predict-upload")
async def predict_upload(file: UploadFile = File(...)) -> dict[str, object]:
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="上传文件不是有效图片。") from exc
    return predict_image(image)


@app.post("/api/predict-canvas")
async def predict_canvas(payload: CanvasPayload) -> dict[str, object]:
    return predict_image(image_from_data_url(payload.data_url))


HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>手写数字识别系统</title>
  <style>
    :root {
      --ink: #1f2530;
      --muted: #647084;
      --paper: rgba(255, 255, 255, .94);
      --line: #e5e7ec;
      --primary: #e85d3f;
      --secondary: #315f86;
      --panel: #33384e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 12%, rgba(255,255,255,.30), transparent 28%),
        linear-gradient(128deg, #f6f0d8 0%, #efc37c 37%, #d9684e 67%, #363a55 100%);
    }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 24px; }
    .hero {
      padding: 28px;
      color: white;
      background: rgba(24, 29, 48, .25);
      border: 1px solid rgba(255, 255, 255, .20);
      border-radius: 20px;
      margin-bottom: 18px;
      backdrop-filter: blur(8px);
    }
    h1 { margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }
    .hero p { margin: 0; color: rgba(255,255,255,.86); line-height: 1.7; }
    .grid { display: grid; grid-template-columns: 1.55fr 1fr; gap: 18px; align-items: start; }
    .card {
      background: var(--paper);
      border: 1px solid rgba(255,255,255,.72);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 20px 46px rgba(31,37,48,.13);
    }
    .tabs { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
    .tab {
      border: 0;
      padding: 10px 14px;
      border-radius: 999px;
      background: #eef0f3;
      color: var(--ink);
      cursor: pointer;
      font-weight: 700;
    }
    .tab.active { background: var(--panel); color: #fff; }
    .panel { display: none; }
    .panel.active { display: block; }
    .actions { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
    button.primary, button.secondary {
      border: 0;
      padding: 11px 15px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 700;
      color: #fff;
    }
    button.primary { background: var(--primary); }
    button.secondary { background: var(--secondary); }
    input[type="file"] {
      display: block;
      width: 100%;
      padding: 14px;
      background: #fff;
      border: 1px dashed #bdc4cf;
      border-radius: 8px;
    }
    #canvas {
      width: 280px;
      height: 280px;
      background: #fff;
      border-radius: 8px;
      border: 2px solid #cfd5df;
      touch-action: none;
      display: block;
    }
    #preview {
      display: none;
      width: 220px;
      height: 220px;
      object-fit: contain;
      margin-top: 12px;
      border-radius: 8px;
      border: 1px solid #d8dde6;
      background: #fff;
    }
    .metric { font-size: 42px; font-weight: 850; margin: 2px 0 4px; }
    .muted { color: var(--muted); }
    .bars { display: grid; gap: 8px; margin-top: 14px; }
    .bar-row { display: grid; grid-template-columns: 32px 1fr 64px; gap: 10px; align-items: center; }
    .bar-track { width: 100%; height: 12px; background: #edf1f5; border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; background: linear-gradient(90deg, var(--secondary), var(--primary)); }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: .95rem; }
    th, td { padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: #434b5b; }
    .history-title { margin: 20px 0 0; font-size: 18px; }
    .error { color: #b32318; margin-top: 10px; min-height: 22px; }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      #canvas { width: 100%; max-width: 280px; }
      h1 { font-size: 28px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>手写数字识别系统</h1>
      <p>基于 CNN 的在线识别页面，支持上传图片、网页手写板、Top-3 预测、概率分布和连续识别历史。</p>
    </section>
    <main class="grid">
      <section class="card">
        <div class="tabs">
          <button class="tab active" data-target="upload-panel">上传图片识别</button>
          <button class="tab" data-target="draw-panel">在线手写板识别</button>
        </div>
        <div id="upload-panel" class="panel active">
          <input type="file" id="fileInput" accept="image/*" />
          <img id="preview" alt="预览图片" />
          <div class="actions">
            <button class="primary" id="predictUploadBtn">识别上传图片</button>
          </div>
        </div>
        <div id="draw-panel" class="panel">
          <p class="muted">请在白底画板上用黑色笔迹手写数字 0-9。</p>
          <canvas id="canvas" width="280" height="280"></canvas>
          <div class="actions">
            <button class="primary" id="predictCanvasBtn">识别当前手写内容</button>
            <button class="secondary" id="clearCanvasBtn">清空画板</button>
          </div>
        </div>
        <div class="error" id="errorBox"></div>
      </section>
      <aside class="card">
        <div class="metric" id="predictionValue">-</div>
        <div id="confidenceValue" class="muted">置信度: -</div>
        <table id="top3Table">
          <thead><tr><th>Rank</th><th>Digit</th><th>Probability</th></tr></thead>
          <tbody></tbody>
        </table>
        <div class="bars" id="probabilityBars"></div>
        <h3 class="history-title">连续识别历史</h3>
        <table id="historyTable">
          <thead><tr><th>Mode</th><th>Prediction</th><th>Confidence</th><th>Top-3</th></tr></thead>
          <tbody></tbody>
        </table>
      </aside>
    </main>
  </div>
  <script>
    const tabs = document.querySelectorAll(".tab");
    const panels = document.querySelectorAll(".panel");
    const history = [];
    const errorBox = document.getElementById("errorBox");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const preview = document.getElementById("preview");
    const fileInput = document.getElementById("fileInput");

    function resetCanvas() {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 18;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#000000";
    }
    resetCanvas();

    let drawing = false;
    function pointFromEvent(event) {
      const rect = canvas.getBoundingClientRect();
      const touch = event.touches ? event.touches[0] : event;
      return {
        x: (touch.clientX - rect.left) * canvas.width / rect.width,
        y: (touch.clientY - rect.top) * canvas.height / rect.height
      };
    }
    function startDraw(event) {
      drawing = true;
      const p = pointFromEvent(event);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      event.preventDefault();
    }
    function draw(event) {
      if (!drawing) return;
      const p = pointFromEvent(event);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
      event.preventDefault();
    }
    function endDraw() { drawing = false; }
    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", endDraw);
    canvas.addEventListener("touchstart", startDraw, {passive: false});
    canvas.addEventListener("touchmove", draw, {passive: false});
    window.addEventListener("touchend", endDraw);

    tabs.forEach(tab => tab.addEventListener("click", () => {
      tabs.forEach(x => x.classList.remove("active"));
      panels.forEach(x => x.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.target).classList.add("active");
      errorBox.textContent = "";
    }));

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) {
        preview.style.display = "none";
        return;
      }
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
    });

    function renderHistory() {
      const body = document.querySelector("#historyTable tbody");
      body.innerHTML = "";
      history.forEach(item => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>${item.mode}</td><td>${item.prediction}</td><td>${item.confidence}</td><td>${item.top3}</td>`;
        body.appendChild(row);
      });
    }

    function renderResult(data, mode) {
      errorBox.textContent = "";
      document.getElementById("predictionValue").textContent = data.prediction;
      document.getElementById("confidenceValue").textContent = `置信度: ${(data.confidence * 100).toFixed(2)}%`;

      const top3Body = document.querySelector("#top3Table tbody");
      top3Body.innerHTML = "";
      data.top3.forEach((item, index) => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>Top ${index + 1}</td><td>${item.digit}</td><td>${(item.probability * 100).toFixed(2)}%</td>`;
        top3Body.appendChild(row);
      });

      const bars = document.getElementById("probabilityBars");
      bars.innerHTML = "";
      data.probabilities.forEach(item => {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `<div>${item.digit}</div><div class="bar-track"><div class="bar-fill" style="width:${(item.probability * 100).toFixed(2)}%"></div></div><div>${(item.probability * 100).toFixed(2)}%</div>`;
        bars.appendChild(row);
      });

      history.unshift({
        mode,
        prediction: data.prediction,
        confidence: `${(data.confidence * 100).toFixed(2)}%`,
        top3: data.top3.map(x => `${x.digit}:${(x.probability * 100).toFixed(1)}%`).join(", ")
      });
      if (history.length > 10) history.pop();
      renderHistory();
    }

    async function checkedJson(response) {
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "预测失败，请重试。");
      return data;
    }

    async function predictUpload() {
      const file = fileInput.files[0];
      if (!file) {
        errorBox.textContent = "请先上传图片。";
        return;
      }
      const formData = new FormData();
      formData.append("file", file);
      try {
        renderResult(await checkedJson(await fetch("/api/predict-upload", {method: "POST", body: formData})), "Upload");
      } catch (err) {
        errorBox.textContent = err.message;
      }
    }

    async function predictCanvas() {
      try {
        const dataUrl = canvas.toDataURL("image/png");
        renderResult(await checkedJson(await fetch("/api/predict-canvas", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({data_url: dataUrl})
        })), "Canvas");
      } catch (err) {
        errorBox.textContent = err.message;
      }
    }

    document.getElementById("predictUploadBtn").addEventListener("click", predictUpload);
    document.getElementById("predictCanvasBtn").addEventListener("click", predictCanvas);
    document.getElementById("clearCanvasBtn").addEventListener("click", () => {
      resetCanvas();
      errorBox.textContent = "";
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
