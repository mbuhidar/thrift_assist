"""
OCR phrase detection endpoints.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List
import json
import os

from backend.services.ocr_service import ocr_service
from backend.services.cache_service import cache_service
from backend.services.image_service import image_service
from backend.api.models.requests import PhraseDetectionRequest
from backend.api.models.responses import PhraseDetectionResponse
from backend.core.config import settings

router = APIRouter(prefix="/ocr", tags=["ocr"])


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
            
            # Convert annotated image to base64
            annotated_image_base64 = None
            if 'annotated_image' in results:
                annotated_image_base64 = image_service.array_to_base64(results['annotated_image'])
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return PhraseDetectionResponse(
                success=True,
                total_matches=results['total_matches'],
                matches=serializable_matches,
                processing_time_ms=processing_time,
                image_dimensions=[image.shape[1], image.shape[0]],
                annotated_image_base64=annotated_image_base64,
                all_detected_text=results.get('all_text', '')
            )
            
        finally:
            # Clean up temporary file
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
    
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


@router.post("/upload")
async def upload_and_detect(
    file: UploadFile = File(...),
    search_phrases: str = Form(...),
    threshold: int = Form(settings.DEFAULT_THRESHOLD),
    text_scale: int = Form(settings.DEFAULT_TEXT_SCALE)
):
    """
    Upload an image file and detect phrases.
    
    This endpoint supports caching to allow fast threshold updates without re-running OCR.
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
            
            # Cache the image info
            cache_service.cache_result(image_hash, {
                'image_path': temp_image_path,
                'image_dimensions': [image.shape[1], image.shape[0]],
                'filename': file.filename,
                'text_scale': text_scale
            })
        
        # Run OCR detection
        results = ocr_service.detect_phrases(
            image_path=temp_image_path,
            search_phrases=phrases_list,
            threshold=threshold,
            text_scale=text_scale,
            show_plot=False
        )
        
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
            annotated_image_base64 = image_service.array_to_base64(results['annotated_image'])
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get image dimensions
        if cached_result and 'image_dimensions' in cached_result:
            image_dims = cached_result['image_dimensions']
        else:
            image_dims = [results['annotated_image'].shape[1], results['annotated_image'].shape[0]] if 'annotated_image' in results else [0, 0]
        
        return JSONResponse({
            "success": True,
            "total_matches": results['total_matches'],
            "matches": serializable_matches,
            "processing_time_ms": processing_time,
            "image_dimensions": image_dims,
            "annotated_image_base64": annotated_image_base64,
            "all_detected_text": results.get('all_text', ''),
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
