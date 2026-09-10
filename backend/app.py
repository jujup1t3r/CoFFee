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