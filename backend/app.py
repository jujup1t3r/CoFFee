import sys
from unittest.mock import MagicMock

sys.modules["optree"] = MagicMock()
sys.modules["optree._C"] = MagicMock()

from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="Coffee Bean Classification API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve and load the trained model weights from train-5
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "backend" / "runs" / "classify" / "train-5" / "weights" / "best.pt"

if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "runs" / "classify" / "train-5" / "weights" / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))


@app.get("/")
def read_root():
    return {"status": "online", "message": "Coffee Bean Classifier API is running"}


@app.post("/api/predict")
async def predict_coffee(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image (.jpg, .png).")

    try:
        # Read uploaded image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image file.")

        # Run inference
        results = model.predict(source=img, verbose=False)
        result = results[0]

        top1_idx = result.probs.top1
        label = result.names[top1_idx]
        confidence = float(result.probs.top1conf.item()) * 100

        # Extract Top-3 predictions
        top3_indices = result.probs.top5[:3]
        top3_predictions = [
            {
                "class": result.names[idx],
                "confidence": round(float(result.probs.data[idx].item()) * 100, 2)
            }
            for idx in top3_indices
        ]

        # Determine defect status
        defect_keywords = ["damage", "fungus", "broken", "black", "sour", "withered", "floater", "shell"]
        is_defective = any(keyword in label.lower() for keyword in defect_keywords)

        return {
            "status": "success",
            "prediction": {
                "label": label,
                "confidence": round(confidence, 2),
                "is_defective": is_defective
            },
            "top3": top3_predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
from collections import defaultdict
from contextlib import asynccontextmanager
import json
from pathlib import Path
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ==================== DATA STORAGE & STATE ====================
bean_stats = {
    "total": 0,
    "good": 0,
    "defects": 0,
    "classes": defaultdict(int),
    "history": [],
}

active_websockets: list[WebSocket] = []

# ==================== MQTT CLIENT SETUP ====================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_EVENT = "coffee/conveyor/bean_detected"
TOPIC_SUMMARY = "coffee/conveyor/summary"


def on_connect(client, userdata, flags, rc, properties=None):
  if rc == 0:
    print("[MQTT] Connected to broker successfully.")
    client.subscribe([(TOPIC_EVENT, 0), (TOPIC_SUMMARY, 0)])
  else:
    print(f"[MQTT] Connection failed with code {rc}")


def on_message(client, userdata, msg):
  global bean_stats
  try:
    payload = json.loads(msg.payload.decode())

    if msg.topic == TOPIC_EVENT:
      defect_type = payload.get("defect_type", "unknown")
      is_defect = payload.get("is_defect", False)

      bean_stats["total"] += 1
      if is_defect:
        bean_stats["defects"] += 1
      else:
        bean_stats["good"] += 1

      bean_stats["classes"][defect_type] += 1
      bean_stats["history"].append(payload)
      if len(bean_stats["history"]) > 50:
        bean_stats["history"].pop(0)

      # Broadcast to connected frontend clients
      message_data = json.dumps({"type": "bean_detected", "data": payload, "stats": bean_stats})
      for ws in active_websockets:
        import asyncio
        asyncio.run_coroutine_threadsafe(ws.send_text(message_data), loop)

    elif msg.topic == TOPIC_SUMMARY:
      print("[MQTT] Summary received:", payload)
      summary_data = json.dumps({"type": "summary", "data": payload})
      for ws in active_websockets:
        import asyncio
        asyncio.run_coroutine_threadsafe(ws.send_text(summary_data), loop)

  except Exception as e:
    print(f"[MQTT] Error processing message: {e}")


try:
  mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
  mqtt_client = mqtt.Client()

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message


# ==================== APPLICATION LIFECYCLE ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
  global loop
  import asyncio
  loop = asyncio.get_running_loop()

  # Start MQTT client
  try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()
    print("[Lifecycle] MQTT loop started.")
  except Exception as e:
    print(f"[Lifecycle] Failed to start MQTT: {e}")

  yield

  # Cleanup on shutdown
  mqtt_client.loop_stop()
  mqtt_client.disconnect()
  print("[Lifecycle] MQTT loop stopped.")


app = FastAPI(title="Coffee Bean Inspection API", lifespan=lifespan)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve and load model weights
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "backend" / "runs" / "classify" / "train-5" / "weights" / "best.pt"

if not MODEL_PATH.exists():
  MODEL_PATH = BASE_DIR / "runs" / "classify" / "train-5" / "weights" / "best.pt"

if not MODEL_PATH.exists():
  raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print(f"Loading classifier from: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))


# ==================== ROUTES & WEBSOCKETS ====================
@app.get("/")
def read_root():
  return {"status": "online", "message": "Coffee Bean Inspection API is running"}


@app.get("/api/stats")
def get_stats():
  return {
      "total": bean_stats["total"],
      "good": bean_stats["good"],
      "defects": bean_stats["defects"],
      "classes": dict(bean_stats["classes"]),
      "history": bean_stats["history"][-10:],
  }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
  await websocket.accept()
  active_websockets.append(websocket)
  try:
    # Send current stats immediately upon connecting
    await websocket.send_text(json.dumps({"type": "init", "stats": bean_stats}))
    while True:
      await websocket.receive_text()
  except WebSocketDisconnect:
    active_websockets.remove(websocket)


@app.post("/api/predict")
async def predict_coffee(file: UploadFile = File(...)):
  if not file.content_type.startswith("image/"):
    raise HTTPException(
        status_code=400,
        detail="Invalid file type. Please upload an image (.jpg, .png).",
    )

  try:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
      raise HTTPException(status_code=400, detail="Failed to decode image file.")

    results = model.predict(source=img, verbose=False)
    result = results[0]

    top1_idx = result.probs.top1
    label = result.names[top1_idx]
    confidence = float(result.probs.top1conf.item()) * 100

    top3_indices = result.probs.top5[:3]
    top3_predictions = [
        {
            "class": result.names[idx],
            "confidence": round(float(result.probs.data[idx].item()) * 100, 2),
        }
        for idx in top3_indices
    ]

    defect_keywords = [
        "damage",
        "fungus",
        "broken",
        "black",
        "sour",
        "withered",
        "floater",
        "shell",
    ]
    is_defective = any(keyword in label.lower() for keyword in defect_keywords)

    return {
        "status": "success",
        "prediction": {
            "label": label,
            "confidence": round(confidence, 2),
            "is_defective": is_defective,
        },
        "top3": top3_predictions,
    }

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  import uvicorn
  uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
