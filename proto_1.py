import cv2
import easyocr
import matplotlib.pyplot as plt
from rapidfuzz import fuzz


# Initialize the EasyOCR reader
def draw_bounding_boxes(image, detections, threshold=0.25, search_text=None, fuzzy_threshold=70):
    for bbox, text, score in detections:
        if score > threshold:
            match = True
            if search_text:
                similarity = fuzz.ratio(text.lower(), search_text.lower())
                match = similarity >= fuzzy_threshold
            if match:
                cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 7)
                # cv2.putText(image, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255, 0, 0), 2)


# Load the image
image_path = "image/gw_cds.jpg"
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Error loading the image. Please check the file path.")

# Define the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Perform text detection
text_detections = reader.readtext(img)
print(text_detections)

# Draw bounding boxes on the image
threshold = 0.25
search_text = "Patriot"
draw_bounding_boxes(img, text_detections, threshold, search_text)

# Display the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
