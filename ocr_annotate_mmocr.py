import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmocr.apis import MMOCRInferencer
from rapidfuzz import fuzz

# Filepath, search terms, and fuzzy matching threshold
IMAGE_PATH = "image/gw_cds_scaled.jpg"

SEARCH_TERMS = [
    'Dean Martin',
    'Toad the Wet Sprocket',
    'James Taylor',
    'Maxi Priest'
]

FUZZ_THRESHOLD = 85  # Fuzzy matching threshold (0-100)

# Set this to True to draw on the original image, False for resized image
DRAW_ON_ORIGINAL = True


def draw_ocr_boxes(image, result, search_terms=None, scale_x=1.0, scale_y=1.0,
                   fuzz_threshold=80):
    """Draw bounding boxes around detected text on the image"""
    drawn_count = 0
    total_detections = 0
    
    print(f"Processing OCR result. Type: {type(result)}")
    if isinstance(result, list) and len(result) > 0:
        print(f"Result has {len(result)} items")
        print(f"First item type: {type(result[0])}")
        if len(result) > 0:
            print(f"First item sample: {result[0]}")
    
    # SceneOCR returns: [(bbox, text, confidence), ...]
    # where bbox = [x1, y1, x2, y2]
    for item in result:
        total_detections += 1
        
        if len(item) < 3:
            print(f"Skipping item with insufficient data: {item}")
            continue
            
        # Extract bbox, text, and confidence from pytesseract result
        bbox = item[0]  # [x1, y1, x2, y2]
        text = item[1]  # text string
        confidence = item[2]  # confidence score (0-1)
        
        if not text or len(bbox) < 4:
            print(f"Skipping item with insufficient data: {item}")
            continue
        
        print(f"Detected text: '{text}' (confidence: {confidence:.2f})")
        
        # Fuzzy search if search_terms is provided
        if search_terms:
            best_match = None
            best_ratio = 0
            for search_term in search_terms:
                ratio = fuzz.ratio(text.lower(), search_term.lower())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = search_term
            
            if best_ratio < fuzz_threshold:
                print(f"  No match found (best: {best_ratio})")
                continue  # Skip if no good match
            match_msg = f"  Found match: '{text}' -> '{best_match}'"
            print(f"{match_msg} (score: {best_ratio})")
        
        drawn_count += 1
        
        # SceneOCR bbox format: [x1, y1, x2, y2]
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Scale coordinates
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            # Create rectangle points for drawing
            pts = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            
            # Draw bounding box
            cv2.polylines(image, [pts], isClosed=True,
                          color=(0, 255, 0), thickness=2)
            
            # Add text label above the box
            text_pos = (x1, max(y1 - 5, 15))
            cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        else:
            print(f"Skipping bbox with unexpected format: {bbox}")
            continue
    
    msg = f"Drew {drawn_count} boxes out of {total_detections} detections"
    print(msg)
    return image


def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    #denoised = cv2.medianBlur(gray, 3)
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #contrast = clahe.apply(gray)
    # Adaptive thresholding (binarization)
    binarized = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
    plt.imshow(cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB))
    plt.show()
    # Convert back to BGR - EasyOCR can work with both grayscale and color
    # processed = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    processed = binarized
    return processed


if __name__ == "__main__":
    # Initialize MMOCR for scene text detection and recognition
    ocr = MMOCRInferencer(det='DBNet', rec='CRNN')

    # Load image
    image_path = IMAGE_PATH
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        error_msg = "Error loading the image. Please check the file path."
        raise ValueError(error_msg)

    # Preprocess image for better OCR
    pre_img = preprocess_image(orig_img)

    # Resize if too large
    max_side = 2000
    orig_h, orig_w = pre_img.shape[:2]
    scale = 1.0
    img = pre_img.copy()
    if max(orig_h, orig_w) > max_side:
        scale = max_side / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img = cv2.resize(pre_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # OCR on image using MMOCR
    result = ocr(img)
    print(f"MMOCR raw result: {result}")
    
    # Convert MMOCR result to list format for processing
    ocr_results = []
    
    # MMOCR returns a dictionary with 'predictions' key
    if isinstance(result, dict) and 'predictions' in result:
        predictions = result['predictions']
        
        # Handle different prediction formats
        if isinstance(predictions, list):
            for pred in predictions:
                if isinstance(pred, dict):
                    # Extract text and polygons/bboxes from MMOCR prediction
                    texts = pred.get('rec_texts', [])
                    polygons = pred.get('det_polygons', [])
                    scores = pred.get('rec_scores', [])
                    
                    # Pair texts with their corresponding polygons
                    for i, (text, polygon) in enumerate(zip(texts, polygons)):
                        if text.strip():  # Skip empty text
                            score = scores[i] if i < len(scores) else 0.8
                            
                            # Convert polygon to bbox [x1, y1, x2, y2]
                            if len(polygon) >= 4:
                                if isinstance(polygon[0], (list, tuple)):
                                    # Polygon format: [[x1,y1], [x2,y2], ...]
                                    x_coords = [pt[0] for pt in polygon]
                                    y_coords = [pt[1] for pt in polygon]
                                    bbox = [min(x_coords), min(y_coords), 
                                           max(x_coords), max(y_coords)]
                                else:
                                    # Flat coordinate format: [x1,y1,x2,y2,...]
                                    # Every 2nd element starting from 0
                                    x_coords = polygon[::2]
                                    # Every 2nd element starting from 1
                                    y_coords = polygon[1::2]
                                    bbox = [min(x_coords), min(y_coords),
                                            max(x_coords), max(y_coords)]
                                
                                ocr_results.append((bbox, text, score))
                    
                elif isinstance(pred, (list, tuple)) and len(pred) >= 3:
                    # Direct format: (bbox, text, confidence)
                    bbox, text, conf = pred[0], pred[1], pred[2]
                    if text and bbox:
                        ocr_results.append((bbox, text, conf))
    
    # Fallback: try to handle as direct list format
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                bbox = item.get('bbox', item.get('box', []))
                text = item.get('text', '')
                conf = item.get('confidence', item.get('score', 0.8))
                if text and bbox:
                    ocr_results.append((bbox, text, conf))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                bbox, text, conf = item[0], item[1], item[2]
                if text and bbox:
                    ocr_results.append((bbox, text, conf))
    
    print(f"SceneOCR found {len(ocr_results)} text regions")

    # Search for list of specific words/phrases (set to None to draw all)
    search_terms = SEARCH_TERMS
    # search_terms = None # Draw all texts

    # Draw results on the image, scaling coordinates for resized image
    if DRAW_ON_ORIGINAL:
        # Calculate scale factors for mapping boxes to original image
        scale_x = orig_w / img.shape[1]
        scale_y = orig_h / img.shape[0]
    else:
        scale_x = 1.0
        scale_y = 1.0

    annotated = draw_ocr_boxes(orig_img.copy(), ocr_results,
                               search_terms=search_terms,
                               scale_x=scale_x, scale_y=scale_y,
                               fuzz_threshold=FUZZ_THRESHOLD)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("SceneOCR Text Detection Results")
    plt.axis('off')
    plt.show()
    plt.show()
