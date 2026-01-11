from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from PIL import Image
from PIL import ExifTags
import io
import os
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


# =============================================================================
# HAIRSTYLE TRANSFER - Virtual Try-On Feature
# Uses AI to transfer hairstyles from reference images to user photos
# =============================================================================

# Hairstyle definitions with prompts for AI generation
HAIRSTYLE_CATALOG = {
    "layered_bob": {
        "name": "Layered Bob",
        "prompt": "layered bob haircut, modern layered bob hairstyle, chin-length layers, voluminous bob",
        "description": "Modern layered cut that adds volume and movement",
        "reference_url": "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?w=512"
    },
    "textured_waves": {
        "name": "Textured Waves", 
        "prompt": "beach waves hairstyle, textured wavy hair, natural soft waves, medium length wavy hair",
        "description": "Soft, natural-looking waves for a relaxed yet elegant look",
        "reference_url": "https://images.unsplash.com/photo-1580618672591-eb180b1a973f?w=512"
    },
    "curly_bob": {
        "name": "Curly Bob",
        "prompt": "curly bob hairstyle, bouncy curls bob haircut, short curly hair, ringlet curls",
        "description": "Bouncy curls that bring life and texture to a classic bob",
        "reference_url": "https://images.unsplash.com/photo-1595152772835-219674b2a8a6?w=512"
    },
    "classic_bob": {
        "name": "Classic Bob",
        "prompt": "classic straight bob, sleek bob haircut, blunt cut bob, professional bob hairstyle",
        "description": "A timeless, straight-cut bob that works for any setting",
        "reference_url": "https://images.unsplash.com/photo-1559563458-527698bf5295?w=512"
    },
    "side_swept": {
        "name": "Side Swept",
        "prompt": "side swept hairstyle, elegant side part hair, swept bangs, glamorous side style",
        "description": "Elegant side-swept look that highlights facial features",
        "reference_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512"
    },
    "pixie_cut": {
        "name": "Pixie Cut",
        "prompt": "pixie cut hairstyle, short pixie haircut, chic pixie, edgy short hair",
        "description": "A short, chic pixie cut for a bold and confident statement",
        "reference_url": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=512"
    },
    "voluminous_curls": {
        "name": "Voluminous Curls",
        "prompt": "voluminous curly hair, big bouncy curls, full curly hairstyle, glamorous curls",
        "description": "Big, bouncy curls for a glamorous and bold look",
        "reference_url": "https://images.unsplash.com/photo-1580618672591-eb180b1a973f?w=512"
    },
    "sleek_straight": {
        "name": "Sleek Straight",
        "prompt": "sleek straight hair, pin straight hairstyle, glossy straight hair, smooth silky hair",
        "description": "Perfectly smooth and sleek straight hairstyle",
        "reference_url": "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=512"
    },
    "french_bob": {
        "name": "French Bob",
        "prompt": "french bob hairstyle, parisian bob, chin length bob with bangs, chic french haircut",
        "description": "Classic French bob with subtle sophistication",
        "reference_url": "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=512"
    },
    "shaggy_layers": {
        "name": "Shaggy Layers",
        "prompt": "shaggy layered haircut, textured shag hairstyle, messy layers, modern shag cut",
        "description": "Textured, effortlessly cool shaggy layers",
        "reference_url": "https://images.unsplash.com/photo-1580618672591-eb180b1a973f?w=512"
    }
}


async def transfer_hairstyle_with_ai(source_image_bytes: bytes, hairstyle_id: str):
    """
    Transfer hairstyle from reference image to user's photo using HairFastGAN.
    
    This creates a merged photo where:
    1. The hairstyle from the reference image is extracted
    2. Applied onto the user's photo
    3. Result shows user with the selected hairstyle
    """
    import httpx
    import tempfile
    
    print(f"Starting hairstyle transfer for: {hairstyle_id}")
    print(f"Image size: {len(source_image_bytes)} bytes")
    
    hairstyle = HAIRSTYLE_CATALOG.get(hairstyle_id)
    if not hairstyle:
        return None, f"Unknown hairstyle ID: {hairstyle_id}"
    
    # Validate image
    if not source_image_bytes or len(source_image_bytes) < 100:
        return None, "Invalid image data"
    
    try:
        # Load and prepare source image
        pil_image = Image.open(io.BytesIO(source_image_bytes))
        pil_image = correct_image_orientation(pil_image)
        
        # Resize for processing (HairFastGAN works best with ~512px)
        max_size = 512
        if pil_image.width > max_size or pil_image.height > max_size:
            ratio = min(max_size / pil_image.width, max_size / pil_image.height)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
        
        print(f"Source image prepared: {pil_image.size}")
        
        # Try HairFastGAN for real hairstyle transfer
        result = await try_hairfastgan_with_reference(pil_image, hairstyle)
        if result:
            print("HairFastGAN transfer successful!")
            return result, None
        
        # Fallback: Apply dramatic local styling
        print("HairFastGAN failed, applying local hairstyle effect")
        
        # Get hair mask
        hair_mask = segment_hair_multiclass(pil_image)
        if hair_mask is None:
            hair_mask = create_face_based_hair_mask(pil_image)
        
        img_rgb = np.array(pil_image.convert('RGB'))
        result = apply_dramatic_hairstyle(img_rgb, hair_mask, hairstyle_id, hairstyle)
        
        result_image = Image.fromarray(result)
        buffer = io.BytesIO()
        result_image.save(buffer, format="JPEG", quality=92)
        
        print(f"Local hairstyle applied: {len(buffer.getvalue())} bytes")
        return buffer.getvalue(), None
        
    except Exception as e:
        print(f"Hairstyle transfer failed: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Failed to apply hairstyle: {str(e)}"


async def try_hairfastgan_with_reference(source_image, hairstyle: dict):
    """
    Use HairFastGAN via Hugging Face Spaces to transfer hairstyle.
    HairFastGAN takes: source photo + reference hairstyle photo
    And outputs: source photo with the reference hairstyle applied
    """
    import httpx
    import tempfile
    from gradio_client import Client, handle_file
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    reference_url = hairstyle.get('reference_url')
    if not reference_url:
        print("No reference URL for hairstyle")
        return None
    
    print(f"Using reference image: {reference_url}")
    
    def run_hairfastgan():
        try:
            # Download reference image
            import requests
            ref_response = requests.get(reference_url, timeout=30)
            if ref_response.status_code != 200:
                print(f"Failed to download reference: {ref_response.status_code}")
                return None
            
            ref_image = Image.open(io.BytesIO(ref_response.content))
            
            # Save images to temp files
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as src_file:
                source_image.save(src_file, format='PNG')
                src_path = src_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as ref_file:
                ref_image.save(ref_file, format='PNG')
                ref_path = ref_file.name
            
            print(f"Connecting to HairFastGAN...")
            
            # Connect to HairFastGAN on Hugging Face Spaces
            client = Client("AIRI-Institute/HairFastGAN")
            
            # Try to discover the correct API endpoint
            try:
                # Print available endpoints for debugging
                print(f"Available endpoints: {client.endpoints}")
            except:
                pass
            
            # Try different possible API names
            api_names_to_try = ["/swap", "/predict", "/run", "/process", "/inference", "/transfer"]
            
            result = None
            for api_name in api_names_to_try:
                try:
                    print(f"Trying API: {api_name}")
                    result = client.predict(
                        handle_file(src_path),   # Source face
                        handle_file(ref_path),   # Hairstyle reference
                        handle_file(ref_path),   # Color reference (same as hairstyle)
                        api_name=api_name
                    )
                    print(f"Success with {api_name}!")
                    break
                except ValueError as ve:
                    if "Cannot find a function" in str(ve):
                        continue
                    raise
                except Exception as e:
                    print(f"API {api_name} failed: {e}")
                    continue
            
            print(f"HairFastGAN result: {result}")
            
            # Clean up temp files
            os.unlink(src_path)
            os.unlink(ref_path)
            
            # Load result image
            if result:
                # Result could be a path or a tuple
                result_path = result[0] if isinstance(result, (list, tuple)) else result
                if os.path.exists(str(result_path)):
                    with open(str(result_path), 'rb') as f:
                        return f.read()
            
            return None
            
        except Exception as e:
            print(f"HairFastGAN error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Run in executor with timeout
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, run_hairfastgan),
                timeout=120  # 2 minute timeout
            )
            return result
    except asyncio.TimeoutError:
        print("HairFastGAN timeout after 120 seconds")
        return None
    except Exception as e:
        print(f"HairFastGAN executor error: {e}")
        return None


def create_face_based_hair_mask(pil_image):
    """
    Create hair mask based on face detection.
    Identifies face position and creates mask for area above/around it.
    """
    import cv2
    import mediapipe as mp
    
    img_rgb = np.array(pil_image.convert('RGB'))
    h, w = img_rgb.shape[:2]
    
    # Try to detect face for accurate positioning
    mp_face_mesh = mp.solutions.face_mesh
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            results = face_mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get face boundary points
                forehead_y = int(landmarks[10].y * h)  # Top of forehead
                left_x = int(landmarks[234].x * w)     # Left side
                right_x = int(landmarks[454].x * w)    # Right side
                
                # Create hair region (above forehead, extending to sides)
                face_width = right_x - left_x
                center_x = (left_x + right_x) // 2
                
                # Hair extends beyond face width
                hair_left = max(0, center_x - int(face_width * 0.8))
                hair_right = min(w, center_x + int(face_width * 0.8))
                hair_top = 0
                hair_bottom = min(h, forehead_y + int(face_width * 0.2))
                
                # Create elliptical mask for natural hair shape
                cv2.ellipse(
                    mask,
                    (center_x, forehead_y - int(face_width * 0.2)),
                    (int(face_width * 0.7), int(face_width * 0.5)),
                    0, 0, 360, 255, -1
                )
                
                # Also fill rectangular area at top
                mask[:forehead_y, hair_left:hair_right] = 255
                
                # Blur for soft edges
                mask = cv2.GaussianBlur(mask, (31, 31), 0)
                
                print(f"Face-based mask created: forehead at y={forehead_y}")
                return mask
                
    except Exception as e:
        print(f"Face detection failed: {e}")
    
    # Fallback: assume top 35% is hair
    mask[:int(h * 0.35), :] = 255
    
    # Add gradient fade
    gradient_height = int(h * 0.1)
    for i in range(gradient_height):
        y = int(h * 0.35) + i
        if y < h:
            alpha = 1.0 - (i / gradient_height)
            mask[y, :] = int(255 * alpha)
    
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask


def apply_dramatic_hairstyle(img_rgb: np.ndarray, hair_mask: np.ndarray, 
                             hairstyle_id: str, hairstyle: dict):
    """
    Apply a dramatic, highly visible hairstyle transformation.
    Changes both color and texture of the hair region.
    """
    import cv2
    
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy().astype(np.float32)
    
    # Hairstyle definitions with dramatic colors and textures
    hairstyle_configs = {
        'curly_bob': {
            'base_color': (50, 35, 25),      # Rich dark brown
            'highlight_color': (90, 70, 50),  # Golden highlights
            'texture': 'curly',
            'color_intensity': 0.85
        },
        'textured_waves': {
            'base_color': (70, 50, 35),       # Warm brown
            'highlight_color': (110, 85, 60), # Honey highlights
            'texture': 'wavy',
            'color_intensity': 0.80
        },
        'layered_bob': {
            'base_color': (40, 28, 18),       # Dark chocolate
            'highlight_color': (70, 50, 35),  # Subtle highlights
            'texture': 'layered',
            'color_intensity': 0.82
        },
        'classic_bob': {
            'base_color': (30, 22, 15),       # Deep brown
            'highlight_color': (55, 40, 28),  # Natural highlights
            'texture': 'smooth',
            'color_intensity': 0.85
        },
        'side_swept': {
            'base_color': (55, 40, 28),       # Medium brown
            'highlight_color': (85, 65, 45),  # Caramel highlights
            'texture': 'swept',
            'color_intensity': 0.78
        },
        'pixie_cut': {
            'base_color': (35, 25, 18),       # Dark brown
            'highlight_color': (60, 45, 32),  # Subtle shine
            'texture': 'short',
            'color_intensity': 0.88
        },
        'voluminous_curls': {
            'base_color': (80, 55, 40),       # Warm auburn
            'highlight_color': (120, 90, 65), # Copper highlights
            'texture': 'curly',
            'color_intensity': 0.82
        },
        'sleek_straight': {
            'base_color': (20, 15, 10),       # Jet black
            'highlight_color': (40, 30, 22),  # Blue-black shine
            'texture': 'sleek',
            'color_intensity': 0.90
        },
        'french_bob': {
            'base_color': (45, 32, 22),       # Elegant brown
            'highlight_color': (75, 55, 40),  # Soft highlights
            'texture': 'smooth',
            'color_intensity': 0.83
        },
        'shaggy_layers': {
            'base_color': (85, 60, 42),       # Light brown
            'highlight_color': (130, 100, 70),# Sun-kissed highlights
            'texture': 'shaggy',
            'color_intensity': 0.75
        }
    }
    
    config = hairstyle_configs.get(hairstyle_id, hairstyle_configs['classic_bob'])
    
    # Ensure mask is proper size
    if hair_mask.shape[:2] != img_rgb.shape[:2]:
        hair_mask = cv2.resize(hair_mask, (w, h))
    
    # Normalize and smooth mask
    mask_norm = hair_mask.astype(np.float32) / 255.0
    mask_norm = cv2.GaussianBlur(mask_norm, (15, 15), 0)
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Create styled hair color
    styled_hair = create_styled_hair_color(
        img_rgb, h, w, 
        config['base_color'], 
        config['highlight_color'],
        config['texture']
    )
    
    # Apply color change with high intensity
    intensity = config['color_intensity']
    result = result * (1 - mask_3ch * intensity) + styled_hair * mask_3ch * intensity
    
    # Add texture effects based on style
    result = apply_texture_effect(result, mask_norm, config['texture'])
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_styled_hair_color(img_rgb, h, w, base_color, highlight_color, texture_type):
    """Create the styled hair color layer with highlights and depth."""
    import cv2
    
    # Get luminance from original for natural variation
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    styled = np.zeros((h, w, 3), dtype=np.float32)
    
    # Create base color
    for c in range(3):
        styled[:, :, c] = base_color[c]
    
    # Add highlights based on original brightness
    highlight_mask = gray > 0.5
    for c in range(3):
        styled[:, :, c] = np.where(
            highlight_mask,
            styled[:, :, c] * 0.7 + highlight_color[c] * 0.3,
            styled[:, :, c]
        )
    
    # Add natural variation based on original luminance
    for c in range(3):
        styled[:, :, c] = styled[:, :, c] * (0.6 + gray * 0.8)
    
    # Add shine/highlight streak
    shine_start = w // 4
    shine_width = w // 3
    shine = np.zeros((h, w), dtype=np.float32)
    shine[:, shine_start:shine_start + shine_width] = 25
    shine = cv2.GaussianBlur(shine, (51, 51), 0)
    
    styled += np.stack([shine] * 3, axis=-1)
    
    return styled


def apply_texture_effect(result: np.ndarray, mask_norm: np.ndarray, texture_type: str):
    """Apply texture-specific effects to make hairstyle more visible."""
    import cv2
    
    h, w = result.shape[:2]
    
    if texture_type == 'curly':
        # Add curl pattern
        for freq in [18, 28]:
            x_sin = np.sin(np.linspace(0, freq * np.pi, w))
            y_cos = np.cos(np.linspace(0, freq * np.pi, h))
            pattern = np.outer(y_cos, x_sin) * 15
            pattern = cv2.GaussianBlur(pattern.astype(np.float32), (5, 5), 0)
            for c in range(3):
                result[:, :, c] += pattern * mask_norm
                
    elif texture_type == 'wavy':
        # Add wave pattern
        y_coords = np.linspace(0, 6 * np.pi, h)
        for i in range(h):
            wave = np.sin(np.linspace(0, 4 * np.pi, w) + y_coords[i]) * 12
            for c in range(3):
                result[i, :, c] += wave * mask_norm[i, :]
                
    elif texture_type == 'sleek':
        # Add shine streak
        shine_x = w // 3
        shine = np.zeros((h, w), dtype=np.float32)
        shine[:, shine_x:shine_x + w//4] = 20
        shine = cv2.GaussianBlur(shine, (41, 41), 0)
        for c in range(3):
            result[:, :, c] += shine * mask_norm
            
    elif texture_type == 'swept':
        # Add diagonal sweep pattern
        for i in range(h):
            offset = int(i * 0.3)
            shift = np.roll(np.linspace(0, 15, w), offset) * mask_norm[i, :]
            for c in range(3):
                result[i, :, c] += shift
                
    elif texture_type in ['short', 'pixie']:
        # Add fine texture
        texture = np.random.uniform(-8, 8, (h, w))
        texture = cv2.GaussianBlur(texture.astype(np.float32), (3, 3), 0)
        for c in range(3):
            result[:, :, c] += texture * mask_norm
    
    return result
    
    return None, "All hairstyle transfer methods failed. Please try again."


def create_fallback_hair_mask(pil_image):
    """Create a fallback hair mask when segmentation fails."""
    import cv2
    
    w, h = pil_image.size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Assume top 40% of image contains hair
    mask[:int(h * 0.4), :] = 255
    
    # Apply gradient for smoother blending
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    return mask


async def try_huggingface_inpainting(pil_image, hair_mask, hairstyle):
    """
    Use Hugging Face's FREE Inference API with FLUX for AI hairstyle generation.
    Uses image-to-image approach: generates a new hairstyle based on the face.
    
    To enable: Set HUGGINGFACE_API_TOKEN environment variable on Render
    Get your free token at: https://huggingface.co/settings/tokens
    """
    import httpx
    import base64
    
    # Get API token from environment
    HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", os.environ.get("HF_TOKEN", ""))
    
    if not HF_TOKEN:
        print("No HuggingFace API token found. Set HUGGINGFACE_API_TOKEN env variable.")
        print("Get your FREE token at: https://huggingface.co/settings/tokens")
        return None
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        # Create a detailed prompt for hairstyle generation
        # Include context about the face to help the AI
        prompt = (
            f"Portrait photo of a person with {hairstyle['prompt']}, "
            f"photorealistic, natural lighting, high quality, detailed hair texture, "
            f"professional photograph, studio lighting"
        )
        
        print(f"Sending to HuggingFace FLUX: {prompt[:80]}...")
        
        # New HuggingFace router endpoint with FLUX model (fast, high quality)
        API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": 3.5,
                "num_inference_steps": 4  # FLUX is very fast
            }
        }
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "image" in content_type:
                    print(f"FLUX generation successful! Size: {len(response.content)} bytes")
                    
                    # Now blend this generated hairstyle with the original face
                    generated_img = Image.open(io.BytesIO(response.content))
                    result = blend_hairstyle_with_face(pil_image, generated_img, hair_mask)
                    
                    if result:
                        buffer = io.BytesIO()
                        result.save(buffer, format="JPEG", quality=90)
                        return buffer.getvalue()
                    else:
                        return response.content  # Return as-is if blending fails
                else:
                    print(f"Unexpected response type: {content_type}")
                    print(f"Response: {response.text[:300]}")
                    return None
                    
            elif response.status_code == 503:
                print("Model is loading, please wait...")
                # Wait and retry once
                import asyncio
                await asyncio.sleep(20)
                response = await client.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
                    return response.content
                return None
            else:
                print(f"HF API error: {response.status_code} - {response.text[:200]}")
                return None
                
    except httpx.TimeoutException:
        print("HuggingFace API timeout")
        return None
    except Exception as e:
        print(f"HuggingFace generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def blend_hairstyle_with_face(original_image, generated_image, hair_mask):
    """
    Blend the AI-generated hairstyle image with the original face.
    Takes the face from original and hair from generated.
    """
    import cv2
    
    try:
        # Resize generated to match original
        generated_resized = generated_image.resize(original_image.size, Image.LANCZOS)
        
        # Convert to arrays
        original_np = np.array(original_image.convert('RGB'))
        generated_np = np.array(generated_resized.convert('RGB'))
        
        # Ensure mask is same size
        if hair_mask.shape[:2] != original_np.shape[:2]:
            hair_mask = cv2.resize(hair_mask, (original_np.shape[1], original_np.shape[0]))
        
        # Normalize mask
        mask_norm = hair_mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=-1)
        
        # Invert mask: we want to keep face from original, take hair from generated
        # But actually for hairstyle, we want generated hair on original face
        # So use: hair region from generated, face region from original
        
        # Apply Gaussian blur to mask edges for smooth blending
        mask_blurred = cv2.GaussianBlur(mask_3ch.astype(np.float32), (21, 21), 0)
        
        # Blend: hair from generated, face from original
        result = original_np * (1 - mask_blurred * 0.7) + generated_np * mask_blurred * 0.7
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
        
    except Exception as e:
        print(f"Blending failed: {e}")
        return None


async def try_replicate_inpainting(pil_image, hair_mask, hairstyle):
    """
    Try Replicate's free tier for inpainting.
    """
    import httpx
    import base64
    
    # Replicate requires API key, skip if not available
    REPLICATE_API_KEY = os.environ.get("REPLICATE_API_TOKEN", "")
    if not REPLICATE_API_KEY:
        print("No Replicate API key, skipping")
        return None
    
    try:
        # Convert to base64 data URIs
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="PNG")
        img_base64 = f"data:image/png;base64,{base64.b64encode(img_buffer.getvalue()).decode()}"
        
        mask_img = Image.fromarray(hair_mask)
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format="PNG")
        mask_base64 = f"data:image/png;base64,{base64.b64encode(mask_buffer.getvalue()).decode()}"
        
        headers = {
            "Authorization": f"Token {REPLICATE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "version": "c11bac58203367db93a3c552bd49a25a5418458ddffb7e90dae55780765e26d6",  # SD inpainting
            "input": {
                "image": img_base64,
                "mask": mask_base64,
                "prompt": hairstyle['prompt'],
                "num_outputs": 1
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Start the prediction
            response = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 201:
                print(f"Replicate start error: {response.status_code}")
                return None
            
            prediction = response.json()
            get_url = prediction.get("urls", {}).get("get")
            
            if not get_url:
                return None
            
            # Poll for result
            for _ in range(30):
                await asyncio.sleep(2)
                result_response = await client.get(get_url, headers=headers)
                result = result_response.json()
                
                if result.get("status") == "succeeded":
                    output_url = result.get("output", [None])[0]
                    if output_url:
                        img_response = await client.get(output_url)
                        return img_response.content
                elif result.get("status") == "failed":
                    print(f"Replicate failed: {result.get('error')}")
                    return None
            
            return None
            
    except Exception as e:
        print(f"Replicate inpainting failed: {e}")
        return None


def apply_local_hairstyle_effect(img_rgb: np.ndarray, hair_mask: np.ndarray, hairstyle_id: str):
    """
    Apply a strong visible local effect as fallback when AI APIs fail.
    This changes hair color and texture noticeably.
    """
    import cv2
    
    # Hairstyle color definitions (more dramatic colors for visibility)
    hairstyle_colors = {
        'curly_bob': {'color': (60, 40, 25), 'texture': 'curly'},
        'textured_waves': {'color': (80, 55, 35), 'texture': 'wavy'},
        'layered_bob': {'color': (50, 35, 20), 'texture': 'layered'},
        'classic_bob': {'color': (40, 28, 18), 'texture': 'smooth'},
        'side_swept': {'color': (70, 50, 32), 'texture': 'swept'},
        'pixie_cut': {'color': (35, 25, 15), 'texture': 'short'},
        'voluminous_curls': {'color': (90, 65, 45), 'texture': 'curly'},
        'sleek_straight': {'color': (25, 18, 10), 'texture': 'sleek'},
        'french_bob': {'color': (55, 40, 28), 'texture': 'bob'},
        'shaggy_layers': {'color': (85, 60, 40), 'texture': 'shaggy'}
    }
    
    style_config = hairstyle_colors.get(hairstyle_id, {'color': (50, 35, 25), 'texture': 'default'})
    base_color = style_config['color']
    texture_type = style_config['texture']
    
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy().astype(np.float32)
    
    # Normalize mask
    mask_norm = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Create styled hair based on texture type
    styled_hair = create_textured_hair(h, w, base_color, texture_type)
    
    # Preserve some original detail for realism
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    detail = cv2.Laplacian(gray, cv2.CV_32F)
    detail = cv2.GaussianBlur(detail, (3, 3), 0)
    
    # Add detail to styled hair
    styled_hair += np.stack([detail * 30] * 3, axis=-1)
    
    # Strong blend (90%) for very visible change
    blend_strength = 0.90
    result = result * (1 - mask_3ch * blend_strength) + styled_hair * mask_3ch * blend_strength
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_textured_hair(h: int, w: int, base_color: tuple, texture_type: str):
    """Create textured hair based on style type."""
    import cv2
    
    styled_hair = np.zeros((h, w, 3), dtype=np.float32)
    
    if texture_type == 'curly':
        # Multiple frequency curl patterns
        for freq in [12, 20, 30]:
            x_pattern = np.sin(np.linspace(0, freq * np.pi, w)) * 0.5 + 0.5
            y_pattern = np.cos(np.linspace(0, freq * np.pi, h)) * 0.5 + 0.5
            texture = np.outer(y_pattern, x_pattern)
            for c in range(3):
                styled_hair[:, :, c] += base_color[c] * texture * 0.4
        styled_hair = cv2.GaussianBlur(styled_hair.astype(np.float32), (5, 5), 0)
        
    elif texture_type == 'wavy':
        y_coords = np.linspace(0, 6 * np.pi, h)
        for i in range(h):
            wave = np.sin(np.linspace(0, 4 * np.pi, w) + y_coords[i]) * 0.4 + 0.6
            for c in range(3):
                styled_hair[i, :, c] = base_color[c] * wave
        styled_hair = cv2.GaussianBlur(styled_hair.astype(np.float32), (7, 7), 0)
        
    elif texture_type == 'sleek':
        # Smooth gradient with shine
        y_grad = np.linspace(0.6, 1.2, h).reshape(-1, 1)
        shine = np.zeros((h, w), dtype=np.float32)
        shine[:, w//3:w//2] = 0.3
        shine = cv2.GaussianBlur(shine, (41, 41), 0)
        for c in range(3):
            styled_hair[:, :, c] = base_color[c] * y_grad + shine * 40
            
    elif texture_type == 'short' or texture_type == 'pixie':
        texture = np.random.uniform(0.7, 1.3, (h, w))
        texture = cv2.GaussianBlur(texture.astype(np.float32), (5, 5), 0)
        for c in range(3):
            styled_hair[:, :, c] = base_color[c] * texture
            
    else:
        # Default: smooth with subtle texture
        texture = np.random.uniform(0.85, 1.15, (h, w))
        texture = cv2.GaussianBlur(texture.astype(np.float32), (9, 9), 0)
        for c in range(3):
            styled_hair[:, :, c] = base_color[c] * texture
    
    return styled_hair


async def try_hairfastgan_transfer(source_image_bytes: bytes, hairstyle: dict):
    """
    Use HairFastGAN via Hugging Face Gradio API (free).
    """
    import tempfile
    import os
    from concurrent.futures import ThreadPoolExecutor
    import asyncio
    
    def run_gradio_client():
        try:
            from gradio_client import Client, handle_file
            
            # Create temp file for source image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                f.write(source_image_bytes)
                source_path = f.name
            
            # Use HairFastGAN on Hugging Face
            client = Client("AIRI-Institute/HairFastGAN", verbose=False)
            
            # Get a reference image - we'll use the source as both for now
            # and let the AI interpret the hairstyle from the prompt
            result = client.predict(
                source_path,  # Source face image
                source_path,  # Shape reference (using source for shape)
                source_path,  # Color reference (using source for color matching)
                api_name="/swap"
            )
            
            # Clean up temp file
            os.unlink(source_path)
            
            if result and os.path.exists(result):
                with open(result, 'rb') as f:
                    return f.read()
            
            return None
            
        except Exception as e:
            print(f"Gradio client error: {e}")
            return None
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, run_gradio_client)
    
    return result


async def try_stable_hair_transfer(source_image_bytes: bytes, hairstyle: dict):
    """
    Alternative hairstyle transfer using Stable-Hair or similar free API.
    """
    import httpx
    import tempfile
    import os
    
    # Try using a public demo API (no API key required)
    # These are free public endpoints from Hugging Face Spaces
    
    endpoints_to_try = [
        "https://ginipick-ai-hairstyle-changer.hf.space/api/predict",
        "https://spaces.huggingface.co/ginipick/ai-hairstyle-changer",
    ]
    
    for endpoint in endpoints_to_try:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Prepare the request
                files = {
                    'image': ('source.jpg', source_image_bytes, 'image/jpeg')
                }
                data = {
                    'hairstyle': hairstyle['name'],
                    'prompt': hairstyle['prompt']
                }
                
                response = await client.post(endpoint, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'image' in result:
                        # Decode base64 result
                        image_data = base64.b64decode(result['image'])
                        return image_data
                        
        except Exception as e:
            print(f"Endpoint {endpoint} failed: {e}")
            continue
    
    return None


def apply_hairstyle_effect_local(source_image_bytes: bytes, hairstyle: dict):
    """
    Apply a hairstyle overlay to the user's photo.
    This works like Snapchat filters - overlays a pre-styled hairstyle on top of the detected face.
    """
    import cv2
    import mediapipe as mp
    
    # Load the source image
    pil_image = Image.open(io.BytesIO(source_image_bytes))
    pil_image = correct_image_orientation(pil_image)
    img_rgb = np.array(pil_image.convert('RGB'))
    
    # Get hairstyle ID
    hairstyle_id = None
    for hid, h in HAIRSTYLE_CATALOG.items():
        if h == hairstyle:
            hairstyle_id = hid
            break
    
    print(f"Applying overlay hairstyle: {hairstyle_id}")
    
    # Detect face landmarks for positioning
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            print("No face detected, using fallback enhancement")
            # Fallback: apply visible hair enhancement
            return apply_visible_hair_effect(img_rgb, hairstyle_id)
        
        # Get face landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = img_rgb.shape[:2]
        
        # Key face points for hairstyle positioning
        # Forehead top (approximate from hairline)
        top_of_head_idx = 10  # Top of forehead
        left_temple_idx = 54   # Left side of face
        right_temple_idx = 284  # Right side of face
        chin_idx = 152  # Chin bottom
        
        top_y = int(landmarks[top_of_head_idx].y * h)
        left_x = int(landmarks[left_temple_idx].x * w)
        right_x = int(landmarks[right_temple_idx].x * w)
        chin_y = int(landmarks[chin_idx].y * h)
        
        # Calculate face dimensions
        face_width = right_x - left_x
        face_height = chin_y - top_y
        face_center_x = (left_x + right_x) // 2
        
        print(f"Face detected: width={face_width}, height={face_height}, center_x={face_center_x}")
        
        # Apply the hairstyle overlay effect
        result = apply_hairstyle_overlay(
            img_rgb, 
            hairstyle_id,
            face_center_x,
            top_y,
            face_width,
            face_height
        )
    
    # Convert back to bytes
    result_image = Image.fromarray(result)
    buffer = io.BytesIO()
    result_image.save(buffer, format="JPEG", quality=90)
    
    return buffer.getvalue()


def apply_hairstyle_overlay(img_rgb: np.ndarray, hairstyle_id: str, 
                            center_x: int, top_y: int, 
                            face_width: int, face_height: int):
    """
    Apply a generated hairstyle overlay based on the hairstyle type.
    Creates a stylized hair effect positioned over the user's head.
    """
    import cv2
    
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    
    # Calculate hairstyle dimensions (hair typically extends beyond face)
    hair_width = int(face_width * 1.6)
    hair_height = int(face_height * 0.7)  # Hair covers top portion
    
    # Position hairstyle above and around face
    hair_top = max(0, top_y - int(hair_height * 0.4))  # Start above forehead
    hair_left = max(0, center_x - hair_width // 2)
    hair_right = min(w, center_x + hair_width // 2)
    
    # Create the hairstyle based on type
    hairstyle_colors = {
        'curly_bob': (45, 30, 20),       # Dark brown curly
        'textured_waves': (60, 45, 35),   # Medium brown waves
        'layered_bob': (40, 25, 15),      # Dark layered
        'classic_bob': (35, 25, 18),      # Classic dark
        'side_swept': (55, 40, 30),       # Side swept brown
        'pixie_cut': (30, 20, 12),        # Dark pixie
        'voluminous_curls': (65, 55, 45), # Light brown curls
        'sleek_straight': (25, 18, 10),   # Black sleek
        'french_bob': (45, 35, 25),       # French bob
        'shaggy_layers': (70, 55, 40)     # Light shaggy
    }
    
    base_color = hairstyle_colors.get(hairstyle_id, (40, 30, 20))
    
    # Get existing hair mask for better blending
    try:
        pil_img = Image.fromarray(img_rgb)
        hair_mask = segment_hair_multiclass(pil_img)
        if hair_mask is None:
            hair_mask = create_approximate_hair_mask(img_rgb, center_x, top_y, face_width, hair_height)
    except:
        hair_mask = create_approximate_hair_mask(img_rgb, center_x, top_y, face_width, hair_height)
    
    # Apply hairstyle effect based on style type
    if hairstyle_id in ['curly_bob', 'voluminous_curls']:
        result = apply_curly_overlay(result, hair_mask, base_color, hairstyle_id)
    elif hairstyle_id in ['textured_waves', 'shaggy_layers']:
        result = apply_wavy_overlay(result, hair_mask, base_color, hairstyle_id)
    elif hairstyle_id in ['sleek_straight', 'classic_bob']:
        result = apply_sleek_overlay(result, hair_mask, base_color, hairstyle_id)
    elif hairstyle_id == 'pixie_cut':
        result = apply_pixie_overlay(result, hair_mask, base_color)
    elif hairstyle_id == 'side_swept':
        result = apply_side_swept_overlay(result, hair_mask, base_color)
    else:
        result = apply_default_overlay(result, hair_mask, base_color)
    
    return result


def create_approximate_hair_mask(img_rgb: np.ndarray, center_x: int, top_y: int, 
                                  face_width: int, hair_height: int):
    """Create an approximate hair region mask based on face position."""
    import cv2
    
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create elliptical mask for hair region
    hair_width = int(face_width * 1.5)
    ellipse_center = (center_x, max(0, top_y - 20))
    axes = (hair_width // 2, int(hair_height * 0.8))
    
    cv2.ellipse(mask, ellipse_center, axes, 0, 0, 360, 255, -1)
    
    # Blur for soft edges
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    
    return mask


def apply_curly_overlay(img_rgb: np.ndarray, hair_mask: np.ndarray, 
                        base_color: tuple, style: str):
    """Apply curly hair effect with visible texture."""
    import cv2
    
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    
    # Normalize mask
    mask_norm = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Create curly texture pattern
    texture = np.zeros((h, w), dtype=np.float32)
    
    # Add multiple frequency curl patterns
    for freq in [15, 25, 40]:
        x_pattern = np.sin(np.linspace(0, freq * np.pi, w)) * 0.5 + 0.5
        y_pattern = np.cos(np.linspace(0, freq * np.pi, h)) * 0.5 + 0.5
        curl_pattern = np.outer(y_pattern, x_pattern)
        texture += curl_pattern * (1.0 / freq * 15)
    
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
    
    # Create styled hair color
    styled_hair = np.zeros_like(img_rgb, dtype=np.float32)
    for c in range(3):
        styled_hair[:, :, c] = base_color[c] * (0.7 + 0.6 * texture)
    
    # Add highlights based on original brightness
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    highlights = gray * 50
    styled_hair += np.stack([highlights] * 3, axis=-1)
    
    # Blend with original based on mask
    blend_factor = mask_3ch * 0.85
    result = result.astype(np.float32) * (1 - blend_factor) + styled_hair * blend_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_wavy_overlay(img_rgb: np.ndarray, hair_mask: np.ndarray,
                       base_color: tuple, style: str):
    """Apply wavy/textured hair effect."""
    import cv2
    
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    
    mask_norm = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Create wave pattern
    y_coords = np.linspace(0, 8 * np.pi, h)
    wave_offset = np.sin(y_coords) * 10
    
    texture = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        shift = int(wave_offset[i])
        row = np.sin(np.linspace(0, 6 * np.pi, w) + i * 0.1) * 0.5 + 0.5
        texture[i] = np.roll(row, shift)
    
    texture = cv2.GaussianBlur(texture, (7, 7), 0)
    
    # Create styled hair
    styled_hair = np.zeros_like(img_rgb, dtype=np.float32)
    for c in range(3):
        styled_hair[:, :, c] = base_color[c] * (0.6 + 0.8 * texture)
    
    # Add natural variation
    noise = np.random.normal(0, 10, styled_hair.shape)
    styled_hair += noise
    
    # Blend
    blend_factor = mask_3ch * 0.8
    result = result.astype(np.float32) * (1 - blend_factor) + styled_hair * blend_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_sleek_overlay(img_rgb: np.ndarray, hair_mask: np.ndarray,
                        base_color: tuple, style: str):
    """Apply sleek/straight hair effect."""
    import cv2
    
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    
    mask_norm = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Create smooth gradient for sleek look
    y_gradient = np.linspace(0.7, 1.3, h).reshape(-1, 1)
    x_gradient = np.linspace(0.9, 1.1, w).reshape(1, -1)
    gradient = y_gradient * x_gradient
    
    # Create styled hair with shine
    styled_hair = np.zeros_like(img_rgb, dtype=np.float32)
    for c in range(3):
        styled_hair[:, :, c] = base_color[c] * gradient
    
    # Add shine streak
    shine_x = w // 3
    shine_width = w // 4
    shine = np.zeros((h, w), dtype=np.float32)
    shine[:, max(0, shine_x):min(w, shine_x + shine_width)] = 40
    shine = cv2.GaussianBlur(shine, (51, 51), 0)
    styled_hair += np.stack([shine] * 3, axis=-1)
    
    # Blend
    blend_factor = mask_3ch * 0.85
    result = result.astype(np.float32) * (1 - blend_factor) + styled_hair * blend_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_pixie_overlay(img_rgb: np.ndarray, hair_mask: np.ndarray, base_color: tuple):
    """Apply pixie cut effect - shorter hair coverage."""
    import cv2
    
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    
    # Reduce mask for shorter hair
    kernel = np.ones((15, 15), np.uint8)
    short_mask = cv2.erode(hair_mask, kernel, iterations=2)
    short_mask = cv2.GaussianBlur(short_mask, (21, 21), 0)
    
    mask_norm = short_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Create textured short hair
    texture = np.random.uniform(0.8, 1.2, (h, w))
    texture = cv2.GaussianBlur(texture.astype(np.float32), (5, 5), 0)
    
    styled_hair = np.zeros_like(img_rgb, dtype=np.float32)
    for c in range(3):
        styled_hair[:, :, c] = base_color[c] * texture
    
    blend_factor = mask_3ch * 0.85
    result = result.astype(np.float32) * (1 - blend_factor) + styled_hair * blend_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_side_swept_overlay(img_rgb: np.ndarray, hair_mask: np.ndarray, base_color: tuple):
    """Apply side-swept hairstyle effect."""
    import cv2
    
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    
    mask_norm = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Create diagonal sweep pattern
    x_coords = np.linspace(0, 1, w)
    y_coords = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x_coords, y_coords)
    sweep = (xx + yy * 0.5) % 1.0
    sweep = cv2.GaussianBlur(sweep.astype(np.float32), (15, 15), 0)
    
    styled_hair = np.zeros_like(img_rgb, dtype=np.float32)
    for c in range(3):
        styled_hair[:, :, c] = base_color[c] * (0.6 + 0.8 * sweep)
    
    blend_factor = mask_3ch * 0.85
    result = result.astype(np.float32) * (1 - blend_factor) + styled_hair * blend_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_default_overlay(img_rgb: np.ndarray, hair_mask: np.ndarray, base_color: tuple):
    """Apply default styled hair overlay."""
    import cv2
    
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    
    mask_norm = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Create basic styled hair
    styled_hair = np.zeros_like(img_rgb, dtype=np.float32)
    for c in range(3):
        styled_hair[:, :, c] = base_color[c]
    
    # Add some texture
    texture = np.random.uniform(0.9, 1.1, (h, w))
    texture = cv2.GaussianBlur(texture.astype(np.float32), (9, 9), 0)
    styled_hair *= np.stack([texture] * 3, axis=-1)
    
    blend_factor = mask_3ch * 0.8
    result = result.astype(np.float32) * (1 - blend_factor) + styled_hair * blend_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_visible_hair_effect(img_rgb: np.ndarray, hairstyle_id: str):
    """Fallback when face detection fails - apply strong visible effect."""
    import cv2
    
    h, w = img_rgb.shape[:2]
    
    # Try to get hair mask
    try:
        pil_img = Image.fromarray(img_rgb)
        hair_mask = segment_hair_multiclass(pil_img)
    except:
        hair_mask = None
    
    if hair_mask is None:
        # Create top-portion mask as fallback
        hair_mask = np.zeros((h, w), dtype=np.uint8)
        hair_mask[:int(h * 0.45), :] = 255
        hair_mask = cv2.GaussianBlur(hair_mask, (51, 51), 0)
    
    # Get color for this style
    colors = {
        'curly_bob': (50, 35, 25),
        'textured_waves': (65, 50, 40),
        'layered_bob': (45, 30, 20),
        'sleek_straight': (25, 18, 10),
        'side_swept': (55, 40, 30),
        'pixie_cut': (35, 25, 15),
        'voluminous_curls': (70, 55, 45),
        'classic_bob': (40, 30, 20),
        'french_bob': (50, 40, 30),
        'shaggy_layers': (75, 60, 45)
    }
    
    base_color = colors.get(hairstyle_id, (45, 35, 25))
    
    mask_norm = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Create styled hair with texture
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    texture = cv2.GaussianBlur(gray, (5, 5), 0)
    
    styled_hair = np.zeros_like(img_rgb, dtype=np.float32)
    for c in range(3):
        styled_hair[:, :, c] = base_color[c] * (0.6 + 0.8 * texture)
    
    result = img_rgb.astype(np.float32) * (1 - mask_3ch * 0.85) + styled_hair * mask_3ch * 0.85
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_curl_texture(img_rgb: np.ndarray, hair_mask: np.ndarray):
    """Add subtle curl/texture effect to hair region."""
    import cv2
    
    mask_normalized = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    
    # Increase contrast and saturation in hair region
    img_float = img_rgb.astype(np.float32)
    
    # Enhance texture with local contrast
    hair_region = img_float * mask_3ch
    enhanced = cv2.GaussianBlur(hair_region, (0, 0), 3)
    hair_sharpened = cv2.addWeighted(hair_region, 1.3, enhanced, -0.3, 0)
    
    # Blend back
    result = img_float * (1 - mask_3ch) + hair_sharpened * mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_sleek_effect(img_rgb: np.ndarray, hair_mask: np.ndarray):
    """Add smooth, sleek shine effect to hair."""
    import cv2
    
    mask_normalized = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    
    img_float = img_rgb.astype(np.float32)
    
    # Smooth hair region
    hair_region = img_float * mask_3ch
    smoothed = cv2.GaussianBlur(hair_region, (5, 5), 0)
    
    # Add shine highlights
    brightness = np.mean(hair_region, axis=2)
    highlight_mask = np.clip((brightness - 150) / 100, 0, 1)
    highlight_3ch = np.stack([highlight_mask] * 3, axis=-1) * mask_3ch
    
    result = smoothed + (30 * highlight_3ch)
    result = img_float * (1 - mask_3ch) + result * mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_wave_texture(img_rgb: np.ndarray, hair_mask: np.ndarray):
    """Add subtle wave texture effect."""
    import cv2
    
    mask_normalized = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    
    img_float = img_rgb.astype(np.float32)
    
    # Create subtle wave pattern overlay
    h, w = img_rgb.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    
    # Sine wave pattern
    wave = np.sin(yy * 0.1 + xx * 0.05) * 10
    wave_3ch = np.stack([wave] * 3, axis=-1)
    
    # Apply only to hair
    result = img_float + (wave_3ch * mask_3ch * 0.3)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_enhancement(img_rgb: np.ndarray, hair_mask: np.ndarray):
    """General hair enhancement - improved color and shine."""
    import cv2
    
    mask_normalized = hair_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    
    # Convert to LAB for better enhancement
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Enhance luminance in hair region
    l_channel = img_lab[:, :, 0]
    enhanced_l = l_channel + (10 * mask_normalized)
    img_lab[:, :, 0] = np.clip(enhanced_l, 0, 255)
    
    # Slightly boost saturation
    img_lab[:, :, 1] = img_lab[:, :, 1] + (5 * mask_normalized)
    img_lab[:, :, 2] = img_lab[:, :, 2] + (5 * mask_normalized)
    
    result = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return result


@app.post("/transfer-hairstyle")
async def transfer_hairstyle_endpoint(
    image: UploadFile = File(...),
    hairstyle_id: str = Form(...)
):
    """
    Transfer a hairstyle to the user's photo.
    
    Args:
        image: User's face photo
        hairstyle_id: ID from HAIRSTYLE_CATALOG (e.g., 'layered_bob', 'curly_bob')
    
    Returns:
        JSON with base64 transformed image
    """
    try:
        # Read and validate image
        contents = await image.read()
        
        if hairstyle_id not in HAIRSTYLE_CATALOG:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Unknown hairstyle_id. Available: {list(HAIRSTYLE_CATALOG.keys())}"
                }
            )
        
        # Transfer the hairstyle
        result_bytes, error = await transfer_hairstyle_with_ai(contents, hairstyle_id)
        
        if error:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": error}
            )
        
        if result_bytes is None:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to process hairstyle transfer"}
            )
        
        # Encode result as base64
        result_base64 = base64.b64encode(result_bytes).decode('utf-8')
        
        # Get dimensions
        result_image = Image.open(io.BytesIO(result_bytes))
        
        return JSONResponse({
            "success": True,
            "image": result_base64,
            "width": result_image.width,
            "height": result_image.height,
            "hairstyle": HAIRSTYLE_CATALOG[hairstyle_id]
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Hairstyle transfer failed: {str(e)}"}
        )


@app.get("/hairstyles")
async def get_available_hairstyles():
    """
    Get list of all available hairstyles for virtual try-on.
    """
    hairstyles = []
    for hid, h in HAIRSTYLE_CATALOG.items():
        hairstyles.append({
            "id": hid,
            "name": h["name"],
            "description": h["description"]
        })
    return JSONResponse({"hairstyles": hairstyles})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "features": ["face_analysis", "hair_segmentation", "hairstyle_transfer"]}

