from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def analyze_face_shape_from_landmarks(landmarks, image_width, image_height):
    """
    Simple heuristic-based face shape classification using landmark ratios.
    """
    
    # Key landmark indices (MediaPipe 468-point model)
    # Forehead top: 10
    # Chin bottom: 152
    # Left cheekbone: 234
    # Right cheekbone: 454
    # Left jaw: 172
    # Right jaw: 397
    
    def get_point(idx):
        lm = landmarks[idx]
        return (lm.x * image_width, lm.y * image_height)
    
    forehead = get_point(10)
    chin = get_point(152)
    left_cheek = get_point(234)
    right_cheek = get_point(454)
    left_jaw = get_point(172)
    right_jaw = get_point(397)
    
    # Calculate dimensions
    face_height = abs(chin[1] - forehead[1])
    face_width = abs(right_cheek[0] - left_cheek[0])
    jaw_width = abs(right_jaw[0] - left_jaw[0])
    
    # Avoid division by zero
    if face_height == 0 or face_width == 0 or jaw_width == 0:
        return "OVAL", 0.5
    
    # Calculate ratios
    aspect_ratio = face_height / face_width
    jaw_to_cheek_ratio = jaw_width / face_width
    
    # Classification logic (simple heuristic)
    confidence = 0.85
    
    if aspect_ratio > 1.5:
        # Tall and narrow
        if jaw_to_cheek_ratio < 0.75:
            return "HEART", confidence
        else:
            return "OVAL", confidence
    elif aspect_ratio < 1.2:
        # Wide face
        if jaw_to_cheek_ratio > 0.9:
            return "SQUARE", confidence
        else:
            return "ROUND", confidence
    else:
        # Medium proportions
        if jaw_to_cheek_ratio < 0.7:
            return "HEART", confidence
        elif jaw_to_cheek_ratio > 0.88:
            return "SQUARE", confidence
        else:
            return "OVAL", confidence

@app.post("/analyze-face")
async def analyze_face(image: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB numpy array
        image_np = np.array(pil_image.convert('RGB'))
        
        # MediaPipe expects RGB
        results = face_mesh.process(image_np)
        
        if not results.multi_face_landmarks:
            return JSONResponse(
                status_code=400,
                content={"error": "No face detected in the image"}
            )
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        
        height, width, _ = image_np.shape
        face_shape, confidence = analyze_face_shape_from_landmarks(
            face_landmarks.landmark,
            width,
            height
        )
        
        return JSONResponse(
            {
                "faceShape": face_shape,
                "confidence": round(confidence, 2)
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )
