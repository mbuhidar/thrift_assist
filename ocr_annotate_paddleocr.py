import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
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

# Set this to True to draw on the original image, or False to draw on the resized image
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
        if len(result[0]) > 0:
            sample = result[0][0] if len(result[0]) > 0 else 'empty'
            print(f"First item sample: {sample}")
    
    # PaddleOCR returns: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence]
    for item in result:
        total_detections += 1
        if len(item) < 2:
            print(f"Skipping item with insufficient data: {item}")
            continue
            
        box = item[0]  # Bounding box coordinates
        text_info = item[1]  # Text and confidence
        
        # Extract text (PaddleOCR returns (text, confidence) tuple)
        if isinstance(text_info, tuple) and len(text_info) >= 1:
            text = text_info[0]
            confidence = text_info[1] if len(text_info) > 1 else 0.0
        else:
            text = str(text_info)
            confidence = 0.0
        
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
        
        # PaddleOCR box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        if isinstance(box, list) and len(box) == 4:
            # Scale and convert to int
            pts = [tuple(map(int, (float(pt[0]) * scale_x,
                                   float(pt[1]) * scale_y))) for pt in box]
            
            # Draw bounding box
            cv2.polylines(image, [np.array(pts)], isClosed=True,
                          color=(0, 255, 0), thickness=2)
            
            # Add text label above the box
            text_pos = (pts[0][0], max(pts[0][1] - 5, 15))
            cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        else:
            print(f"Skipping box with unexpected format: {box}")
            continue
    
    msg = f"Drew {drawn_count} boxes out of {total_detections} detections"
    print(msg)
    return image


def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.medianBlur(gray, 3)
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    # Adaptive thresholding (binarization)
    binarized = cv2.adaptiveThreshold(contrast, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
    # Convert back to BGR - PaddleOCR requires 3-channel images
    processed = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
    return processed


if __name__ == "__main__":
    # Initialize OCR
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')

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

    # OCR on image
    result = ocr.predict(img)
    is_list_dict = (isinstance(result, list) and len(result) > 0 and
                    isinstance(result[0], dict))
    if is_list_dict:
        result = result[0]

    # Search for list of specific words/phrases (set to None to draw all)
    search_terms = SEARCH_TERMS
    # search_texts = None # Draw all texts

    # Draw results on the image, scaling coordinates for resized image
    if DRAW_ON_ORIGINAL:
        # Calculate scale factors for mapping boxes to original image
        scale_x = orig_w / img.shape[1]
        scale_y = orig_h / img.shape[0]
    else:
        scale_x = 1.0
        scale_y = 1.0

    annotated = draw_ocr_boxes(orig_img.copy(), result,
                               search_terms=search_terms,
                               scale_x=scale_x, scale_y=scale_y,
                               fuzz_threshold=FUZZ_THRESHOLD)

    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    plt.show()
