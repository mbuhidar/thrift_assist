import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from rapidfuzz import fuzz

# Set this to True to draw on the original image, or False to draw on the resized image
DRAW_ON_ORIGINAL = False

def draw_ocr_boxes(image, ocr_result, search_texts=None, scale_x=1.0, scale_y=1.0, fuzz_threshold=80):
    # Accept both dict and list result formats from PaddleOCR
    if isinstance(ocr_result, dict):
        items = zip(ocr_result.get('rec_polys', []), ocr_result.get('rec_texts', []))
    elif isinstance(ocr_result, list):
        items = ((line.get('points'), line.get('text')) for line in ocr_result)
    else:
        raise ValueError("Unknown OCR result format.")

    for box, text in items:
        if box is None or (hasattr(box, '__len__') and len(box) == 0):
            continue
        # Use rapidfuzz for fuzzy search if search_texts is provided
        if search_texts:
            if not isinstance(text, str):
                continue
            match_found = False
            for s in search_texts:
                if fuzz.partial_ratio(s.lower(), text.lower()) >= fuzz_threshold:
                    match_found = True
                    break
            if not match_found:
                continue
        # Normalize box to 4 points
        if isinstance(box, np.ndarray):
            pts = box
        elif len(box) == 8:
            pts = [(box[i], box[i+1]) for i in range(0, 8, 2)]
        elif len(box) == 4 and all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2 for pt in box):
            pts = box
        elif len(box) == 2 and all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2 for pt in box):
            x1, y1 = box[0]
            x2, y2 = box[1]
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            print(f"Skipping box with unexpected format: {box}")
            continue
        # Scale and convert to int
        pts = [tuple(map(int, (float(pt[0]) * scale_x, float(pt[1]) * scale_y))) for pt in pts]
        cv2.polylines(image, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
        if isinstance(text, str):
            cv2.putText(image, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
    return image

def preprocess_image(img):
    return img
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.medianBlur(gray, 3)
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(denoised)
    # Adaptive thresholding (binarization)
    binarized = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
    # Convert back to BGR for OCR if needed
    processed = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
    return processed

if __name__ == "__main__":
    # Initialize OCR
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')

    # Load image
    image_path = "image/big_shot_pinball.jpg"
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError("Error loading the image. Please check the file path.")

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
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        result = result[0]

    # Search for list of specific words/phrases (set to None to draw all)
    search_texts = ['Big Shot', 'EBAY']
    # search_texts = None # Draw all texts

    # Draw results on the image, scaling coordinates for resized image if needed
    if DRAW_ON_ORIGINAL:
        # Calculate scale factors for mapping boxes to original image
        scale_x = orig_w / img.shape[1]
        scale_y = orig_h / img.shape[0]
        annotated = draw_ocr_boxes(orig_img.copy(), result, 
                                   search_texts=search_texts, 
                                   scale_x=scale_x, scale_y=scale_y)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    else:
        annotated = draw_ocr_boxes(img.copy(), result, 
                                   search_texts=search_texts)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.show()
