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
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>112304260138 Zhao Dan | CNN Digit Recognition Console</title>
  <style>
    :root { --ink:#17202a; --muted:#667085; --paper:rgba(255,255,255,.94); --line:#d9e0e8; --steel:#26384f; --teal:#0f8b8d; --coral:#e45f3c; --shadow:0 18px 55px rgba(25,39,52,.16); }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; font-family:"Segoe UI","Microsoft YaHei",sans-serif; color:var(--ink); background:linear-gradient(rgba(255,255,255,.34) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.24) 1px,transparent 1px),linear-gradient(132deg,#f4efe5 0%,#d8e9e7 30%,#f3cf8a 63%,#40516a 100%); background-size:34px 34px,34px 34px,cover; }
    .shell { max-width:1220px; margin:0 auto; padding:24px; }
    .topbar { display:grid; grid-template-columns:1fr auto; gap:18px; align-items:end; margin-bottom:18px; }
    .brand-kicker { color:#2f5868; font-size:13px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    h1 { margin:6px 0 8px; font-size:36px; line-height:1.14; letter-spacing:0; }
    .subtitle { margin:0; color:#4e5d6d; line-height:1.75; max-width:780px; }
    .status-strip { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
    .chip { padding:8px 10px; border:1px solid rgba(255,255,255,.65); border-radius:999px; background:rgba(255,255,255,.72); color:#26384f; font-size:13px; font-weight:700; box-shadow:0 8px 24px rgba(25,39,52,.08); white-space:nowrap; }
    .workspace { display:grid; grid-template-columns:minmax(0,1.46fr) minmax(360px,.9fr); gap:18px; align-items:start; }
    .surface { border:1px solid rgba(255,255,255,.72); background:var(--paper); border-radius:10px; box-shadow:var(--shadow); overflow:hidden; }
    .surface-head { display:flex; justify-content:space-between; gap:12px; align-items:center; padding:16px 18px; border-bottom:1px solid var(--line); background:linear-gradient(90deg,rgba(255,255,255,.82),rgba(244,247,248,.72)); }
    .surface-title { margin:0; font-size:18px; font-weight:850; }
    .surface-note { margin:4px 0 0; color:var(--muted); font-size:13px; }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; }
    .tab { border:1px solid #d2d9e2; background:#f8fafc; color:#344054; padding:9px 12px; border-radius:8px; cursor:pointer; font-weight:800; }
    .tab.active { background:var(--steel); color:#fff; border-color:var(--steel); }
    .panel { display:none; padding:18px; min-height:406px; }
    .panel.active { display:block; }
    .upload-zone { min-height:236px; border:1.5px dashed #a9b6c4; border-radius:10px; background:linear-gradient(180deg,#fff,#f7fafb); display:grid; place-items:center; padding:18px; text-align:center; }
    input[type="file"] { width:min(520px,100%); padding:13px; background:#fff; border:1px solid #d5dde6; border-radius:8px; }
    #preview { display:none; width:220px; height:220px; object-fit:contain; margin:14px auto 0; border-radius:10px; border:1px solid #d8dde6; background:#fff; }
    .draw-layout { display:grid; grid-template-columns:300px 1fr; gap:18px; align-items:start; }
    #canvas { width:280px; height:280px; background:#fff; border-radius:10px; border:2px solid #bfc9d5; display:block; touch-action:none; box-shadow:inset 0 0 0 1px rgba(23,32,42,.03); }
    .tips { margin:0; color:var(--muted); line-height:1.75; }
    .actions { display:flex; gap:10px; margin-top:14px; flex-wrap:wrap; }
    button.primary,button.secondary { border:0; border-radius:8px; cursor:pointer; font-weight:850; padding:11px 15px; color:#fff; }
    button.primary { background:linear-gradient(135deg,var(--coral),#c9412a); }
    button.secondary { background:linear-gradient(135deg,var(--teal),#126d75); }
    button.primary:disabled,button.secondary:disabled { opacity:.65; cursor:wait; }
    .error { min-height:24px; margin-top:10px; color:#b42318; font-weight:700; }
    .result-body { padding:18px; }
    .result-stage { display:grid; grid-template-columns:140px 1fr; gap:16px; align-items:center; padding:16px; border:1px solid #e2e8f0; border-radius:10px; background:linear-gradient(145deg,#fff,#f7fbfb); }
    .confidence-ring { width:132px; height:132px; border-radius:50%; background:conic-gradient(var(--teal) calc(var(--score,0) * 1%),#e8edf2 0); display:grid; place-items:center; position:relative; }
    .confidence-ring::after { content:""; position:absolute; width:102px; height:102px; background:#fff; border-radius:50%; box-shadow:inset 0 0 0 1px #e7edf3; }
    .metric { position:relative; z-index:1; font-size:54px; font-weight:900; line-height:1; }
    .result-caption { margin:0 0 4px; color:var(--muted); font-weight:700; }
    .confidence-text { margin:0; font-size:26px; font-weight:900; color:var(--steel); }
    .mini-note { margin:8px 0 0; color:var(--muted); line-height:1.6; font-size:13px; }
    table { width:100%; border-collapse:collapse; margin-top:14px; font-size:14px; }
    th,td { padding:10px 8px; border-bottom:1px solid var(--line); text-align:left; }
    th { color:#475467; font-weight:850; }
    .bars { display:grid; gap:8px; margin-top:14px; }
    .bar-row { display:grid; grid-template-columns:28px 1fr 58px; gap:10px; align-items:center; }
    .bar-track { height:13px; background:#edf2f7; border-radius:999px; overflow:hidden; }
    .bar-fill { height:100%; background:linear-gradient(90deg,var(--teal),var(--coral)); border-radius:999px; transition:width .25s ease; }
    .section-label { margin:18px 0 0; font-size:15px; color:#24364b; font-weight:900; }
    .history-wrap { max-height:230px; overflow:auto; }
    @media (max-width:920px) { .topbar,.workspace,.draw-layout,.result-stage { grid-template-columns:1fr; } .status-strip { justify-content:flex-start; } #canvas { width:100%; max-width:280px; } h1 { font-size:30px; } }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div>
        <div class="brand-kicker">112304260138 | Zhao Dan</div>
        <h1>CNN Digit Recognition Console</h1>
        <p class="subtitle">A PyTorch CNN web system for handwritten digit recognition. Upload an image or draw directly on the canvas, then inspect Top-3 predictions, probability distribution, and recognition history.</p>
      </div>
      <div class="status-strip"><span class="chip" id="serviceChip">Checking service</span><span class="chip">Kaggle Score 0.99639</span><span class="chip">Render Ready</span></div>
    </header>
    <main class="workspace">
      <section class="surface">
        <div class="surface-head"><div><h2 class="surface-title">Input Workspace</h2><p class="surface-note">Choose image upload or draw a digit online.</p></div><div class="tabs"><button class="tab active" data-target="upload-panel">Upload Image</button><button class="tab" data-target="draw-panel">Draw Digit</button></div></div>
        <div id="upload-panel" class="panel active"><div class="upload-zone"><div><strong>Select a handwritten digit image</strong><p class="surface-note">White background / black stroke images work best. The system crops, centers, and normalizes automatically.</p><input type="file" id="fileInput" accept="image/*" /><img id="preview" alt="Preview" /></div></div><div class="actions"><button class="primary" id="predictUploadBtn">Predict Uploaded Image</button></div></div>
        <div id="draw-panel" class="panel"><div class="draw-layout"><canvas id="canvas" width="280" height="280"></canvas><div><p class="tips">Write a single digit from 0 to 9 on the white canvas. The stroke is converted into a 28 x 28 model input before CNN inference.</p><div class="actions"><button class="primary" id="predictCanvasBtn">Predict Canvas Digit</button><button class="secondary" id="clearCanvasBtn">Clear Canvas</button></div></div></div></div>
        <div class="error" id="errorBox"></div>
      </section>
      <aside class="surface">
        <div class="surface-head"><div><h2 class="surface-title">Recognition Result</h2><p class="surface-note">Prediction, Top-3 classes, and full probability distribution.</p></div></div>
        <div class="result-body"><div class="result-stage"><div class="confidence-ring" id="confidenceRing" style="--score:0"><div class="metric" id="predictionValue">-</div></div><div><p class="result-caption">Current Prediction</p><p class="confidence-text" id="confidenceValue">Confidence: -</p><p class="mini-note">After inference, the chart below shows how strongly the model supports each digit class.</p></div></div><div class="section-label">Top-3 Predictions</div><table id="top3Table"><thead><tr><th>Rank</th><th>Digit</th><th>Probability</th></tr></thead><tbody></tbody></table><div class="section-label">Probability Distribution</div><div class="bars" id="probabilityBars"></div><div class="section-label">Recognition History</div><div class="history-wrap"><table id="historyTable"><thead><tr><th>Mode</th><th>Prediction</th><th>Confidence</th><th>Top-3</th></tr></thead><tbody></tbody></table></div></div>
      </aside>
    </main>
  </div>
  <script>
    const tabs=document.querySelectorAll(".tab"),panels=document.querySelectorAll(".panel"),history=[]; const errorBox=document.getElementById("errorBox"),canvas=document.getElementById("canvas"),ctx=canvas.getContext("2d"),preview=document.getElementById("preview"),fileInput=document.getElementById("fileInput"),uploadBtn=document.getElementById("predictUploadBtn"),canvasBtn=document.getElementById("predictCanvasBtn");
    function resetCanvas(){ctx.fillStyle="#ffffff";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.lineWidth=18;ctx.lineCap="round";ctx.lineJoin="round";ctx.strokeStyle="#000000";} resetCanvas();
    let drawing=false; function pointFromEvent(event){const rect=canvas.getBoundingClientRect();const touch=event.touches?event.touches[0]:event;return{x:(touch.clientX-rect.left)*canvas.width/rect.width,y:(touch.clientY-rect.top)*canvas.height/rect.height};} function startDraw(event){drawing=true;const p=pointFromEvent(event);ctx.beginPath();ctx.moveTo(p.x,p.y);event.preventDefault();} function draw(event){if(!drawing)return;const p=pointFromEvent(event);ctx.lineTo(p.x,p.y);ctx.stroke();event.preventDefault();} function endDraw(){drawing=false;} canvas.addEventListener("mousedown",startDraw);canvas.addEventListener("mousemove",draw);window.addEventListener("mouseup",endDraw);canvas.addEventListener("touchstart",startDraw,{passive:false});canvas.addEventListener("touchmove",draw,{passive:false});window.addEventListener("touchend",endDraw);
    tabs.forEach(tab=>tab.addEventListener("click",()=>{tabs.forEach(x=>x.classList.remove("active"));panels.forEach(x=>x.classList.remove("active"));tab.classList.add("active");document.getElementById(tab.dataset.target).classList.add("active");errorBox.textContent="";}));
    fileInput.addEventListener("change",()=>{const file=fileInput.files[0];if(!file){preview.style.display="none";return;}preview.src=URL.createObjectURL(file);preview.style.display="block";});
    async function refreshHealth(){try{const response=await fetch("/health");const data=await response.json();document.getElementById("serviceChip").textContent=`Model online | ${data.device}`;}catch{document.getElementById("serviceChip").textContent="Model status pending";}} refreshHealth();
    function renderHistory(){const body=document.querySelector("#historyTable tbody");body.innerHTML="";history.forEach(item=>{const row=document.createElement("tr");row.innerHTML=`<td>${item.mode}</td><td>${item.prediction}</td><td>${item.confidence}</td><td>${item.top3}</td>`;body.appendChild(row);});}
    function renderResult(data,mode){errorBox.textContent="";const confidence=data.confidence*100;document.getElementById("predictionValue").textContent=data.prediction;document.getElementById("confidenceValue").textContent=`Confidence: ${confidence.toFixed(2)}%`;document.getElementById("confidenceRing").style.setProperty("--score",confidence.toFixed(2));const top3Body=document.querySelector("#top3Table tbody");top3Body.innerHTML="";data.top3.forEach((item,index)=>{const row=document.createElement("tr");row.innerHTML=`<td>Top ${index+1}</td><td>${item.digit}</td><td>${(item.probability*100).toFixed(2)}%</td>`;top3Body.appendChild(row);});const bars=document.getElementById("probabilityBars");bars.innerHTML="";data.probabilities.forEach(item=>{const row=document.createElement("div");row.className="bar-row";row.innerHTML=`<div>${item.digit}</div><div class="bar-track"><div class="bar-fill" style="width:${(item.probability*100).toFixed(2)}%"></div></div><div>${(item.probability*100).toFixed(2)}%</div>`;bars.appendChild(row);});history.unshift({mode,prediction:data.prediction,confidence:`${confidence.toFixed(2)}%`,top3:data.top3.map(x=>`${x.digit}:${(x.probability*100).toFixed(1)}%`).join(", ")});if(history.length>10)history.pop();renderHistory();}
    async function checkedJson(response){const data=await response.json();if(!response.ok)throw new Error(data.detail||"Prediction failed. Please try again.");return data;}
    async function predictUpload(){const file=fileInput.files[0];if(!file){errorBox.textContent="Please upload an image first.";return;}const formData=new FormData();formData.append("file",file);uploadBtn.disabled=true;try{renderResult(await checkedJson(await fetch("/api/predict-upload",{method:"POST",body:formData})),"Upload");}catch(err){errorBox.textContent=err.message;}finally{uploadBtn.disabled=false;}}
    async function predictCanvas(){canvasBtn.disabled=true;try{const dataUrl=canvas.toDataURL("image/png");renderResult(await checkedJson(await fetch("/api/predict-canvas",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({data_url:dataUrl})})),"Canvas");}catch(err){errorBox.textContent=err.message;}finally{canvasBtn.disabled=false;}}
    uploadBtn.addEventListener("click",predictUpload);canvasBtn.addEventListener("click",predictCanvas);document.getElementById("clearCanvasBtn").addEventListener("click",()=>{resetCanvas();errorBox.textContent="";});
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
