from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import random

app = FastAPI()

FACE_SHAPES = ["OVAL", "ROUND", "SQUARE", "HEART", "DIAMOND"]

@app.post("/analyze-face")
async def analyze_face(image: UploadFile = File(...)):
    # For now ignore the real image and return a random shape
    face_shape = random.choice(FACE_SHAPES)
    confidence = round(random.uniform(0.8, 0.98), 2)

    return JSONResponse(
        {
            "faceShape": face_shape,
            "confidence": confidence
        }
    )
