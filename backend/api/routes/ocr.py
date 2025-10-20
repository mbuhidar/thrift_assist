"""
OCR phrase detection endpoints.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List
import json
import os
import io
import base64
import gc

from backend.services.ocr_service import ocr_service
from backend.services.cache_service import cache_service
from backend.services.image_service import image_service
from backend.utils.image_utils import resize_image_for_display, calculate_optimal_jpeg_quality
from backend.api.models.requests import PhraseDetectionRequest
from backend.api.models.responses import PhraseDetectionResponse
from backend.core.config import settings

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.post("/upload")
async def upload_and_detect(
    file: UploadFile = File(...),
    search_phrases: str = Form(...),
    threshold: int = Form(settings.DEFAULT_THRESHOLD),
    text_scale: int = Form(settings.DEFAULT_TEXT_SCALE),
    max_image_width: int = Form(2560)  # New parameter with default
):
    """
    Upload an image file and detect phrases.
    
    This endpoint supports caching to allow fast threshold updates without re-running OCR.
    Device-optimized image resizing ensures optimal performance across mobile, tablet, and desktop.
    """
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Parse search phrases
        try:
            phrases_list = json.loads(search_phrases)
            if not isinstance(phrases_list, list):
                raise ValueError("search_phrases must be a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid search_phrases format: {str(e)}")
        
        # Log received parameters
        print(f"📥 Received request: max_image_width={max_image_width}, text_scale={text_scale}")
        
        # Read and validate image data
        image_data = await file.read()
        
        if not image_service.validate_image_data(image_data, settings.MAX_UPLOAD_SIZE_MB):
            raise HTTPException(status_code=400, detail="Invalid or oversized image")
        
        # Generate cache key
        image_hash = cache_service.get_image_hash(image_data, text_scale)
        
        # Check cache
        cached_result = cache_service.get_cached_result(image_hash)
        
        if cached_result and cached_result.get('image_path'):
            # Use cached image path
            temp_image_path = cached_result['image_path']
            if os.path.exists(temp_image_path):
                print(f"🚀 Using cached image for hash: {image_hash[:8]}...")
            else:
                cached_result = None
        
        if not cached_result:
            # Process image
            import numpy as np
            import cv2
            
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Save to temporary file
            temp_image_path = image_service.save_temp_image(image)
            if not temp_image_path:
                raise HTTPException(status_code=500, detail="Failed to save image")
            
            # Clean up immediately
            del image_array
            del image
            gc.collect()
            
            # Cache the image info
            cache_service.cache_result(image_hash, {
                'image_path': temp_image_path,
                'image_dimensions': None,
                'filename': file.filename,
                'text_scale': text_scale
            })
        
        # Clear uploaded data from memory
        del image_data
        gc.collect()
        
        # IMPORTANT: Resize image BEFORE OCR detection if needed
        # This ensures annotations are drawn at the correct scale
        import cv2
        original_image = cv2.imread(temp_image_path)
        original_height, original_width = original_image.shape[:2]
        
        # Determine if we need to resize for processing
        if original_width > max_image_width:
            scale_factor = max_image_width / original_width
            new_width = max_image_width
            new_height = int(original_height * scale_factor)
            
            print(f"📐 Pre-processing resize: {original_width}×{original_height} → {new_width}×{new_height}")
            
            # Resize with OpenCV
            resized_image = cv2.resize(
                original_image,
                (new_width, new_height),
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Save resized image temporarily
            import tempfile
            resized_path = temp_image_path.replace('.jpg', '_resized.jpg')
            cv2.imwrite(resized_path, resized_image)
            
            # Use resized image for OCR
            processing_image_path = resized_path
            
            # Clean up
            del original_image
            del resized_image
            gc.collect()
        else:
            processing_image_path = temp_image_path
            print(f"⏭️ No pre-processing resize needed (image ≤ {max_image_width}px)")
        
        # Run OCR detection on the appropriately-sized image
        results = ocr_service.detect_phrases(
            image_path=processing_image_path,
            search_phrases=phrases_list,
            threshold=threshold,
            text_scale=text_scale,
            show_plot=False
        )
        
        # Clean up resized temp file if it was created
        if processing_image_path != temp_image_path and os.path.exists(processing_image_path):
            os.unlink(processing_image_path)
        
        if not results:
            return JSONResponse({
                "success": False,
                "total_matches": 0,
                "matches": {},
                "processing_time_ms": 0,
                "error_message": "Failed to process image or no text detected"
            })
        
        # Format response
        serializable_matches = ocr_service.format_matches_for_api(results)
        
        annotated_image_base64 = None
        if 'annotated_image' in results:
            from PIL import Image
            import cv2
            
            # Image is already at the correct size with annotations
            annotated_cv2 = results['annotated_image']
            current_height, current_width = annotated_cv2.shape[:2]
            
            print(f"📐 Annotated image size: {current_width}×{current_height}")
            
            # Convert to PIL
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_cv2, cv2.COLOR_BGR2RGB))
            
            # Clear OpenCV image
            del annotated_cv2
            gc.collect()
            
            # Calculate optimal JPEG quality
            jpeg_quality = calculate_optimal_jpeg_quality(annotated_pil.width)
            
            # Encode to base64 with optimization
            buffered = io.BytesIO()
            annotated_pil.save(buffered, format="JPEG", quality=jpeg_quality, optimize=True)
            annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            print(f"💾 Final encoded image: {annotated_pil.width}×{annotated_pil.height}, quality={jpeg_quality}")
            
            # Clean up everything immediately
            del annotated_pil
            del buffered
            gc.collect()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get image dimensions (use original dimensions for reference)
        if cached_result and 'image_dimensions' in cached_result and cached_result['image_dimensions']:
            image_dims = cached_result['image_dimensions']
        else:
            image_dims = [original_width, original_height]
        
        # Calculate total matches correctly
        total_matches = sum(len(matches) for matches in serializable_matches.values())
        
        # Clear results from memory
        del results
        gc.collect()
        
        return JSONResponse({
            "success": True,
            "total_matches": total_matches,
            "matches": serializable_matches,
            "processing_time_ms": processing_time,
            "image_dimensions": image_dims,
            "annotated_image_base64": annotated_image_base64,
            "all_detected_text": "",
            "filename": file.filename,
            "cached": cached_result is not None
        })
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "total_matches": 0,
            "matches": {},
            "processing_time_ms": processing_time,
            "error_message": str(e)
        })
    finally:
        # Always clean up at the end
        gc.collect()


@router.post("/detect", response_model=PhraseDetectionResponse)
async def detect_phrases(request: PhraseDetectionRequest):
    """
    Detect phrases in base64-encoded image.
    
    This endpoint accepts a base64-encoded image and searches for specified phrases.
    """
    start_time = datetime.now()
    
    try:
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="image_base64 is required")
        
        # Convert base64 to image
        image = image_service.base64_to_array(request.image_base64)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Save to temporary file
        temp_image_path = image_service.save_temp_image(image)
        if not temp_image_path:
            raise HTTPException(status_code=500, detail="Failed to save image")
        
        # Clear image from memory immediately after saving
        original_dims = [image.shape[1], image.shape[0]]
        del image
        gc.collect()
        
        try:
            # Run OCR detection
            results = ocr_service.detect_phrases(
                image_path=temp_image_path,
                search_phrases=request.search_phrases,
                threshold=request.threshold,
                text_scale=request.text_scale,
                show_plot=False
            )
            
            if not results:
                return PhraseDetectionResponse(
                    success=False,
                    total_matches=0,
                    matches={},
                    processing_time_ms=0,
                    error_message="Failed to process image or no text detected"
                )
            
            # Format matches for API response
            serializable_matches = ocr_service.format_matches_for_api(results)
            
            # Convert annotated image to base64 with device-optimized resizing
            annotated_image_base64 = None
            if 'annotated_image' in results:
                from PIL import Image
                import cv2
                
                annotated_cv2 = results['annotated_image']
                original_height, original_width = annotated_cv2.shape[:2]
                
                # Get max_image_width from request
                max_width = request.max_image_width
                
                # MEMORY OPTIMIZATION: Resize with OpenCV BEFORE PIL conversion
                if original_width > max_width:
                    scale_factor = max_width / original_width
                    new_width = max_width
                    new_height = int(original_height * scale_factor)
                    
                    annotated_cv2 = cv2.resize(
                        annotated_cv2,
                        (new_width, new_height),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                    
                    # Clear original
                    del results['annotated_image']
                    gc.collect()
                
                # Convert smaller image to PIL
                annotated_pil = Image.fromarray(cv2.cvtColor(annotated_cv2, cv2.COLOR_BGR2RGB))
                
                # Clear OpenCV image
                del annotated_cv2
                gc.collect()
                
                # Calculate optimal JPEG quality
                jpeg_quality = calculate_optimal_jpeg_quality(annotated_pil.width)
                
                # Encode to base64
                buffered = io.BytesIO()
                annotated_pil.save(buffered, format="JPEG", quality=jpeg_quality, optimize=True)
                annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Cleanup
                del annotated_pil
                del buffered
                gc.collect()
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Clear results
            del results
            gc.collect()
            
            return PhraseDetectionResponse(
                success=True,
                total_matches=len(serializable_matches),
                matches=serializable_matches,
                processing_time_ms=processing_time,
                image_dimensions=original_dims,
                annotated_image_base64=annotated_image_base64,
                all_detected_text=""  # Don't return full text to save memory
            )
            
        finally:
            # Clean up temporary file
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            gc.collect()
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return PhraseDetectionResponse(
            success=False,
            total_matches=0,
            matches={},
            processing_time_ms=processing_time,
            error_message=str(e)
        )
    finally:
        gc.collect()
