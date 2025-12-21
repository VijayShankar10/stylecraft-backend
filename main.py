from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import random

app = FastAPI()

def analyze_face_shape_simple(image: Image.Image):
    """
    Simple face shape heuristic based on image aspect ratio.
    This is a placeholder until you add real ML logic.
    """
    width, height = image.size
    
    # Avoid division by zero
    if width == 0 or height == 0:
        return "OVAL", 0.5
    
    aspect_ratio = height / width
    
    # Simple heuristic classification
    # In production, replace with real face landmark detection
    if aspect_ratio > 1.4:
        # Tall and narrow
        shapes = ["OVAL", "HEART"]
        shape = random.choice(shapes)
        confidence = round(random.uniform(0.82, 0.92), 2)
    elif aspect_ratio < 1.15:
        # Wide face
        shapes = ["ROUND", "SQUARE"]
        shape = random.choice(shapes)
        confidence = round(random.uniform(0.80, 0.90), 2)
    else:
        # Medium proportions
        shapes = ["OVAL", "DIAMOND", "SQUARE"]
        shape = random.choice(shapes)
        confidence = round(random.uniform(0.83, 0.93), 2)
    
    return shape, confidence

@app.post("/analyze-face")
async def analyze_face(image: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Analyze face shape
        face_shape, confidence = analyze_face_shape_simple(pil_image)
        
        return JSONResponse(
            {
                "faceShape": face_shape,
                "confidence": confidence
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )
