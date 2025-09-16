from paddleocr import PaddleOCR
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_paddle_ocr_boxes(image, ocr_result, search_text=None):
    h, w = image.shape[:2]

    # Handle dict result (with rec_polys, rec_texts, rec_scores)
    if isinstance(ocr_result, dict):
        rec_polys = ocr_result.get('rec_polys', [])
        rec_texts = ocr_result.get('rec_texts', [])
        rec_scores = ocr_result.get('rec_scores', [])
        items = zip(rec_polys, rec_texts, rec_scores)
    # Handle list result (list of dicts with 'points', 'text', 'score')
    elif isinstance(ocr_result, list):
        items = []
        for line in ocr_result:
            box = line.get('points')
            text = line.get('text')
            score = line.get('score')
            items.append((box, text, score))
    else:
        raise ValueError("Unknown OCR result format.")

    for box, text, score in items:
        print(f"box: {box}, text: {text}, score: {score}")  # DEBUG
        # If search_text is set, only draw boxes for matching text (case-insensitive, supports list or str)
        if search_text is not None:
            if isinstance(search_text, str):
                search_list = [search_text]
            else:
                search_list = list(search_text)
            if not isinstance(text, str) or not any(s.lower() in text.lower() for s in search_list):
                continue
        if box is not None:
            if isinstance(box, np.ndarray):
                if box.shape[0] == 4:
                    box_pts = box
                else:
                    continue
            elif isinstance(box, (list, tuple)) and len(box) == 4:
                box_pts = box
            else:
                continue
            try:
                pts = [tuple(int(float(coord)) for coord in pt) for pt in box_pts]
                print(f"Drawing pts: {pts} for text: {text}")  # DEBUG
                cv2.polylines(image, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
                # cv2.putText(image, str(text), pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            except Exception as e:
                print(f"Error drawing box {box}: {e}")
    return image

if __name__ == "__main__":
    # Initialize the OCR model
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')  # use_textline_orientation enables orientation detection

    # Load image
    # image_path = "image/big_shot_pinball.jpg"
    image_path = "image/gw_cds.jpg"

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error loading the image. Please check the file path.")

    # Automatically resize large images to max_side_limit (default 4000)
    max_side_limit = 2000  # You can adjust this value as needed
    h, w = img.shape[:2]
    # Track original and resized image sizes
    orig_img = img.copy()
    orig_h, orig_w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side_limit:
        scale = max_side_limit / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Resizing image from ({w}x{h}) to ({new_w}x{new_h}) to fit within limit.")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Run OCR
    result = ocr.predict(img)
    print("Raw OCR result:", result)  # DEBUG

    # Extract the first (and only) result dict
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        result = result[0]

    # Draw results
    # Set your search text here (case-insensitive, exact match)
    # search_text = ['Dean Martin', 'Maxi Priest']  # e.g., 'BIG SHOT' or None for all
    search_text = None
    # Draw results on the original image, scaling coordinates if needed

def draw_paddle_ocr_boxes_scaled(image, ocr_result, search_text=None, scale=1.0):
    h, w = image.shape[:2]
    if isinstance(ocr_result, dict):
        rec_polys = ocr_result.get('rec_polys', [])
        rec_texts = ocr_result.get('rec_texts', [])
        rec_scores = ocr_result.get('rec_scores', [])
        items = zip(rec_polys, rec_texts, rec_scores)
    elif isinstance(ocr_result, list):
        items = []
        for line in ocr_result:
            box = line.get('points')
            text = line.get('text')
            score = line.get('score')
            items.append((box, text, score))
    else:
        raise ValueError("Unknown OCR result format.")

    for box, text, score in items:
        if search_text is not None:
            if isinstance(search_text, str):
                search_list = [search_text]
            else:
                search_list = list(search_text)
            if not isinstance(text, str) or not any(s.lower() in text.lower() for s in search_list):
                continue
        if box is not None:
            # Handle different box formats
            box_pts = None
            if isinstance(box, np.ndarray):
                if box.shape[0] == 4:
                    box_pts = box
            elif isinstance(box, (list, tuple)):
                if len(box) == 4 and all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2 for pt in box):
                    box_pts = box
                elif len(box) == 8:
                    # Flat list of 8 numbers (x1, y1, x2, y2, ...)
                    box_pts = [(box[i], box[i+1]) for i in range(0, 8, 2)]
                elif len(box) == 2 and all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2 for pt in box):
                    # Rectangle as two points
                    x1, y1 = box[0]
                    x2, y2 = box[1]
                    box_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            if box_pts is None:
                print(f"Skipping box with unexpected format: {box}")
                continue
            try:
                # Scale coordinates back to original image size if needed
                if scale != 1.0:
                    pts = [tuple(int(float(coord) / scale) for coord in pt) for pt in box_pts]
                else:
                    pts = [tuple(int(float(coord)) for coord in pt) for pt in box_pts]
                print(f"Drawing box: {pts} for text: {text}")
                cv2.polylines(image, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
                cv2.putText(image, str(text), pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            except Exception as e:
                print(f"Error drawing box {box}: {e}")
    return image

    img_with_boxes = draw_paddle_ocr_boxes_scaled(orig_img, result, search_text=search_text, scale=scale)
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()