from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from PIL import Image
from PIL import ExifTags
import io
import base64
import numpy as np

app = FastAPI()

# Initialize MediaPipe on first use
_hair_segmenter = None

def get_hair_segmenter():
    """Lazy-load MediaPipe Image Segmenter for hair detection."""
    global _hair_segmenter
    if _hair_segmenter is None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Use the multiclass selfie segmenter which includes hair class
        base_options = python.BaseOptions(
            model_asset_path='selfie_multiclass_256x256.tflite'
        )
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        _hair_segmenter = vision.ImageSegmenter.create_from_options(options)
    return _hair_segmenter


def download_model_if_needed():
    """Download the MediaPipe hair segmentation model if not present."""
    import os
    import urllib.request
    
    model_path = 'selfie_multiclass_256x256.tflite'
    if not os.path.exists(model_path):
        print("Downloading hair segmentation model...")
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully!")
    return model_path


def correct_image_orientation(image: Image.Image):
    """
    Rotates image based on EXIF orientation tag.
    CameraX stores rotation in EXIF rather than rotating pixels.
    """
    try:
        exif = image._getexif()
        if exif is None:
            return image
        
        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == 'Orientation':
                orientation_key = key
                break
        
        if orientation_key is None:
            return image
        
        orientation = exif.get(orientation_key)
        
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
            
    except (AttributeError, KeyError, IndexError):
        # No EXIF data or orientation tag
        pass
    
    return image


def analyze_face_shape(image: Image.Image):
    """
    Analyze face shape based on image proportions.
    """
    width, height = image.size
    
    if width == 0 or height == 0:
        return "OVAL", 0.75
    
    ratio = width / height
    
    if height > width:
        ratio = width / height
    else:
        ratio = height / width
    
    if ratio < 0.60:
        shape = "OBLONG"
        confidence = 0.88
    elif ratio < 0.68:
        shape = "OBLONG"
        confidence = 0.85
    elif ratio < 0.76:
        shape = "OVAL"
        confidence = 0.90
    elif ratio < 0.82:
        shape = "HEART"
        confidence = 0.82
    elif ratio < 0.88:
        shape = "SQUARE"
        confidence = 0.85
    else:
        shape = "ROUND"
        confidence = 0.87
    
    return shape, confidence


def segment_hair_multiclass(pil_image: Image.Image):
    """
    Use MediaPipe Multiclass Selfie Segmentation for precise hair detection.
    
    The multiclass model outputs these classes:
    - 0: Background
    - 1: Hair
    - 2: Body/Skin
    - 3: Face/Skin
    - 4: Clothes
    - 5: Others/Accessories
    """
    import mediapipe as mp
    import cv2
    
    # Ensure model is downloaded
    download_model_if_needed()
    
    # Convert PIL to numpy RGB
    img_rgb = np.array(pil_image.convert('RGB'))
    original_height, original_width = img_rgb.shape[:2]
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Get segmenter
    segmenter = get_hair_segmenter()
    
    # Segment the image
    result = segmenter.segment(mp_image)
    
    if result.category_mask is None:
        return None
    
    # Get the category mask
    category_mask = result.category_mask.numpy_view()
    
    # Hair is class 1
    hair_mask = (category_mask == 1).astype(np.uint8) * 255
    
    # Resize back to original size if needed
    if hair_mask.shape[:2] != (original_height, original_width):
        hair_mask = cv2.resize(hair_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
    
    # Gaussian blur for smooth edges
    hair_mask = cv2.GaussianBlur(hair_mask, (7, 7), 0)
    
    return hair_mask


def apply_color_to_hair(pil_image: Image.Image, hair_mask: np.ndarray, color_hex: str, intensity: float = 0.6):
    """
    Apply a color tint to the hair region only using HSV blending.
    Preserves hair texture while changing color.
    """
    import cv2
    
    # Parse hex color
    color_hex = color_hex.lstrip('#')
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    
    # Convert PIL to numpy RGB
    img_rgb = np.array(pil_image.convert('RGB'))
    
    # Normalize hair mask to 0-1
    mask_normalized = hair_mask.astype(np.float32) / 255.0
    
    # Apply intensity factor
    mask_with_intensity = mask_normalized * intensity
    
    # Convert to HSV for better color blending
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Create target color in HSV
    target_rgb = np.array([[[r, g, b]]], dtype=np.uint8)
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]
    
    # Blend Hue and Saturation only, preserve Value (brightness/texture)
    result_hsv = img_hsv.copy()
    
    # Hue blending
    result_hsv[:, :, 0] = (
        img_hsv[:, :, 0] * (1 - mask_with_intensity) + 
        target_hsv[0] * mask_with_intensity
    )
    
    # Saturation blending - blend towards target saturation
    result_hsv[:, :, 1] = (
        img_hsv[:, :, 1] * (1 - mask_with_intensity * 0.7) + 
        target_hsv[1] * mask_with_intensity * 0.7
    )
    
    # Value (brightness) - keep original to preserve texture
    # result_hsv[:, :, 2] stays the same
    
    # Clamp values
    result_hsv[:, :, 0] = np.clip(result_hsv[:, :, 0], 0, 179)
    result_hsv[:, :, 1] = np.clip(result_hsv[:, :, 1], 0, 255)
    result_hsv[:, :, 2] = np.clip(result_hsv[:, :, 2], 0, 255)
    
    # Convert back to RGB
    result_hsv = result_hsv.astype(np.uint8)
    result_rgb = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(result_rgb)


def apply_adjustments_to_hair(
    pil_image: Image.Image, 
    hair_mask: np.ndarray,
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    highlight: float = 0,
    shadow: float = 0
):
    """
    Apply brightness, contrast, saturation, highlight, shadow adjustments
    ONLY to the hair region, leaving the rest of the image unchanged.
    
    All values range from -30 to 30 where 0 is neutral.
    """
    import cv2
    
    # Convert PIL to numpy RGB
    img_rgb = np.array(pil_image.convert('RGB')).astype(np.float32)
    original_img = img_rgb.copy()
    
    # Normalize mask
    mask = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask] * 3, axis=-1)
    
    # Apply brightness (-30 to 30 maps to -76.5 to 76.5)
    brightness_delta = brightness * 2.55
    img_rgb = img_rgb + brightness_delta
    
    # Apply contrast (scale factor 0.7 to 1.3)
    contrast_factor = 1 + (contrast / 100)
    img_rgb = (img_rgb - 127.5) * contrast_factor + 127.5
    
    # Apply saturation in HSV space
    if saturation != 0:
        img_hsv = cv2.cvtColor(np.clip(img_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_factor = 1 + (saturation / 50)
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * sat_factor
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
    # Apply highlights (affects bright pixels)
    if highlight != 0:
        highlight_factor = highlight / 100
        brightness_map = np.mean(img_rgb, axis=2) / 255.0
        highlight_mask = np.clip(brightness_map - 0.5, 0, 0.5) * 2  # Only bright pixels
        highlight_mask_3ch = np.stack([highlight_mask] * 3, axis=-1)
        img_rgb = img_rgb + (highlight_factor * 100 * highlight_mask_3ch)
    
    # Apply shadows (affects dark pixels)
    if shadow != 0:
        shadow_factor = shadow / 100
        brightness_map = np.mean(img_rgb, axis=2) / 255.0
        shadow_mask = np.clip(0.5 - brightness_map, 0, 0.5) * 2  # Only dark pixels
        shadow_mask_3ch = np.stack([shadow_mask] * 3, axis=-1)
        img_rgb = img_rgb + (shadow_factor * 100 * shadow_mask_3ch)
    
    # Clamp to valid range
    img_rgb = np.clip(img_rgb, 0, 255)
    
    # Blend: apply adjustments only where hair mask is present
    result = original_img * (1 - mask_3ch) + img_rgb * mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


@app.post("/analyze-face")
async def analyze_face_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        pil_image = correct_image_orientation(pil_image)
        face_shape, confidence = analyze_face_shape(pil_image)
        
        return JSONResponse({
            "faceShape": face_shape,
            "confidence": confidence
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )


@app.post("/segment-hair")
async def segment_hair_endpoint(image: UploadFile = File(...)):
    """
    Segment hair from an image and return the mask as base64 PNG.
    """
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        pil_image = correct_image_orientation(pil_image)
        
        hair_mask = segment_hair_multiclass(pil_image)
        
        if hair_mask is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not detect hair in the image"}
            )
        
        mask_image = Image.fromarray(hair_mask)
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return JSONResponse({
            "mask": mask_base64,
            "width": mask_image.width,
            "height": mask_image.height
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Segmentation failed: {str(e)}"}
        )


@app.post("/apply-hair-color")
async def apply_hair_color_endpoint(
    image: UploadFile = File(...),
    color: str = Form(...),
    intensity: float = Form(0.6)
):
    """
    Apply a hair color to the image and return the result as base64 JPEG.
    Only the hair region is affected; face and background remain unchanged.
    """
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        pil_image = correct_image_orientation(pil_image)
        
        # Get hair mask using multiclass segmentation
        hair_mask = segment_hair_multiclass(pil_image)
        
        if hair_mask is None or np.max(hair_mask) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not detect hair in the image. Please use a clearer photo with visible hair."}
            )
        
        intensity = max(0.0, min(1.0, intensity))
        result_image = apply_color_to_hair(pil_image, hair_mask, color, intensity)
        
        buffer = io.BytesIO()
        result_image.save(buffer, format="JPEG", quality=90)
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return JSONResponse({
            "image": result_base64,
            "width": result_image.width,
            "height": result_image.height
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Color application failed: {str(e)}"}
        )


@app.post("/apply-hair-adjustments")
async def apply_hair_adjustments_endpoint(
    image: UploadFile = File(...),
    brightness: float = Form(0),
    contrast: float = Form(0),
    saturation: float = Form(0),
    highlight: float = Form(0),
    shadow: float = Form(0),
    color: str = Form(None),
    colorIntensity: float = Form(0.6)
):
    """
    Apply adjustments (brightness, contrast, saturation, highlight, shadow)
    AND optionally a color tint ONLY to the hair region.
    
    All adjustment values range from -30 to 30 where 0 is neutral.
    """
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        pil_image = correct_image_orientation(pil_image)
        
        # Get hair mask
        hair_mask = segment_hair_multiclass(pil_image)
        
        if hair_mask is None or np.max(hair_mask) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not detect hair in the image. Please use a clearer photo with visible hair."}
            )
        
        # Apply color first if specified
        if color and color.strip():
            pil_image = apply_color_to_hair(pil_image, hair_mask, color, colorIntensity)
        
        # Apply adjustments to hair only
        result_image = apply_adjustments_to_hair(
            pil_image, hair_mask,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            highlight=highlight,
            shadow=shadow
        )
        
        buffer = io.BytesIO()
        result_image.save(buffer, format="JPEG", quality=90)
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return JSONResponse({
            "image": result_base64,
            "width": result_image.width,
            "height": result_image.height
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Adjustment failed: {str(e)}"}
        )
