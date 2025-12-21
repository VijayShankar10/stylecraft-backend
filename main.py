from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/analyze-face")
async def analyze_face(image: UploadFile = File(...)):
    # For now ignore the real image and return a fixed/dummy result
    # You can plug real ML logic here later.
    return JSONResponse(
        {
            "faceShape": "OVAL",
            "confidence": 0.9
        }
    )
