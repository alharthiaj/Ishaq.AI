import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
import uvicorn

app = FastAPI(title="إسحاق AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج (سيحمّله Render تلقائياً من الإنترنت عند أول تشغيل)
model = YOLO('yolov8n.pt')

DAMAGE_AR = {"dent": "طعجة", "scratch": "خدش", "crack": "كسر", "car": "سيارة"}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    results = model(img, conf=0.3)
    
    damages = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        damages.append({
            "type": DAMAGE_AR.get(name, name),
            "confidence": round(float(box.conf[0]) * 100, 1),
            "bbox": box.xyxy[0].tolist()
        })
    
    return {"status": "success", "damages": damages}

if __name__ == "__main__":
    # Render يحدد المنفذ تلقائياً عبر متغير البيئة PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
