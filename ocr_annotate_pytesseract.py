import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from rapidfuzz import fuzz

# Filepath, search terms, and fuzzy matching threshold
IMAGE_PATH = "image/gw_cds.jpg"

SEARCH_TERMS = [
    'Dean Martin', 
    'Toad the Wet Sprocket', 
    'James Taylor', 
    'Maxi Priest',
    'Al Green',
    'MaryMary'
    ]

FUZZ_THRESHOLD = 80  # Fuzzy matching threshold (0-100)

# Set this to True to draw on the original image, or False to draw on the resized image
DRAW_ON_ORIGINAL = True


def draw_ocr_boxes(image, ocr_result, search_texts=None, scale_x=1.0, scale_y=1.0, fuzz_threshold=80):
    # Handle pytesseract result format - list of dictionaries with 'bbox' and 'text'
    if not isinstance(ocr_result, list):
        raise ValueError("Expected list format from pytesseract result.")
    
    boxes_drawn = 0
    total_items = len(ocr_result)
    
    for item in ocr_result:
        text = item.get('text', '').strip()
        bbox = item.get('bbox')
        
        if not text or bbox is None:
            continue
            
        # Use rapidfuzz for fuzzy search if search_texts is provided
        if search_texts:
            match_found = False
            matched_search_term = None
            for s in search_texts:
                fuzzy_score = fuzz.partial_ratio(s.lower(), text.lower())
                if fuzzy_score >= fuzz_threshold:
                    match_found = True
                    matched_search_term = s
                    print(f"FOUND: '{text}' matches '{s}' (fuzzy score: {fuzzy_score})")
                    break
            if not match_found:
                continue
        else:
            # If no search terms, draw all boxes
            print(f"Drawing box for: '{text}'")
        
        # Convert bbox to 4 corner points (x, y, w, h) -> [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        x, y, w, h = bbox
        pts = [
            (x, y),           # top-left
            (x + w, y),       # top-right
            (x + w, y + h),   # bottom-right
            (x, y + h)        # bottom-left
        ]
        
        # Scale and convert to int
        pts = [tuple(map(int, (float(pt[0]) * scale_x, float(pt[1]) * scale_y))) for pt in pts]
        cv2.polylines(image, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(image, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
        boxes_drawn += 1
    
    print(f"Drew {boxes_drawn} boxes out of {total_items} OCR results")
    return image


def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpening_kernel)
    
    # Denoise (after sharpening to reduce noise amplification)
    denoised = cv2.medianBlur(sharpened, 3)
    
    # Increase contrast using CLAHE with higher clip limit
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    contrast = clahe.apply(denoised)
    
    # Additional contrast enhancement using histogram stretching
    # Find min and max values, then stretch to full 0-255 range
    min_val = np.min(contrast)
    max_val = np.max(contrast)
    if max_val > min_val:  # Avoid division by zero
        stretched = ((contrast - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        stretched = contrast
    
    # Optional: Apply gamma correction for additional contrast control
    gamma = 1.2  # Values > 1 increase contrast in mid-tones
    gamma_corrected = np.power(stretched / 255.0, gamma) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    plt.imshow(gamma_corrected, cmap='gray')
    plt.show()
    return gamma_corrected  # Return grayscale directly


if __name__ == "__main__":
    # Load image
    image_path = IMAGE_PATH
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError("Error loading the image. Please check the file path.")

    # Preprocess image for better OCR
    pre_img = preprocess_image(orig_img)

    # Resize if too large
    #max_side = 2000
    #orig_h, orig_w = pre_img.shape[:2]
    #scale = 1.0
    img = pre_img.copy()
    #if max(orig_h, orig_w) > max_side:
    #    scale = max_side / max(orig_h, orig_w)
    #    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    #    img = cv2.resize(pre_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # OCR with pytesseract
    # Get bounding boxes and text
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Convert to our expected format
    result = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if text:  # Only include non-empty text
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            result.append({
                'text': text,
                'bbox': (x, y, w, h)
            })

    # Search for list of specific words/phrases (set to None to draw all)
    search_terms = SEARCH_TERMS
    # search_terms = None  # Uncomment to draw ALL detected text
    
    print(f"\nSearching for terms: {search_terms}")
    print(f"Fuzzy matching threshold: {FUZZ_THRESHOLD}")
    print(f"Total OCR results found: {len(result)}")
    print("-" * 50)

    # Draw results on the image, scaling coordinates for resized image if needed
    if DRAW_ON_ORIGINAL:
        # Use original image but we need to convert processed image to 3-channel for drawing
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        scale_x = 1.0
        scale_y = 1.0
    else:
        display_img = orig_img.copy()
        scale_x = 1.0
        scale_y = 1.0

    annotated = draw_ocr_boxes(display_img, result, 
                               search_texts=search_terms, 
                               scale_x=scale_x, scale_y=scale_y, 
                               fuzz_threshold=FUZZ_THRESHOLD)
    
    print("-" * 50)
    print("Search completed. Check the image for highlighted matches.")

    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    plt.show()
