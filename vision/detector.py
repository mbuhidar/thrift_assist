"""Main Google Cloud Vision API phrase detector."""

import cv2
import os
import warnings
from contextlib import redirect_stderr
from io import StringIO
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

# Changed to absolute imports
from config.vision_config import VisionConfig
from vision.grouper import TextLineGrouper
from vision.matcher import PhraseMatcher
from vision.annotator import ImageAnnotator

# Suppress warnings
os.environ.update({
    'GRPC_VERBOSITY': 'NONE',
    'GRPC_TRACE': '',
    'GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE': 'true',
    'GLOG_minloglevel': '3'
})
warnings.filterwarnings("ignore")


class FilteredStringIO(StringIO):
    def write(self, s):
        if 'ALTS creds ignored' not in s and 'absl::InitializeLog' not in s:
            return super().write(s)
        return len(s)


def suppress_stderr_warnings():
    return redirect_stderr(FilteredStringIO())


with suppress_stderr_warnings():
    from google.cloud import vision


class VisionPhraseDetector:
    """Main class for Vision API phrase detection."""
    
    def __init__(self, config: VisionConfig = None):
        self.config = config or VisionConfig()
        self.config.setup_credentials()
        self.grouper = TextLineGrouper(self.config.angle_tolerance)
        self.matcher = PhraseMatcher(self.config)
    
    def detect(self, image_path: str, search_phrases: List[str],
              threshold: int = None, show_plot: bool = True,
              text_scale: int = None) -> Optional[Dict]:
        """Detect and annotate phrases in image."""
        threshold = threshold or self.config.fuzz_threshold
        text_scale = text_scale or self.config.default_text_scale
        
        # Load image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Detect text
        text_annotations = self._detect_text(image_path)
        if not text_annotations:
            print("No text detected in image.")
            return None
        
        print(f"📸 Detected {len(text_annotations)-1} text elements")
        
        # Group into lines
        text_lines = self.grouper.group(text_annotations)
        print(f"📋 Grouped into {len(text_lines)} text lines")
        
        # Show text orientations
        self._print_orientation_info(text_lines)
        
        # Find matches
        phrase_matches = {}
        for phrase in search_phrases:
            matches = self.matcher.find_matches(phrase, text_lines, 
                                               text_annotations[0].description, threshold)
            if matches:
                phrase_matches[phrase] = matches
                print(f"🎯 Found {len(matches)} matches for '{phrase}'")
        
        if not phrase_matches:
            print("❌ No phrase matches found.")
            return {'image': image, 'annotated_image': image.copy(), 
                   'matches': {}, 'total_matches': 0}
        
        # Annotate
        annotator = ImageAnnotator(text_scale)
        annotated = annotator.draw_annotations(image, phrase_matches)
        
        if show_plot:
            self._show_results(annotated, phrase_matches)
        
        return {
            'image': image,
            'annotated_image': annotated,
            'matches': phrase_matches,
            'total_matches': sum(len(m) for m in phrase_matches.values()),
            'all_text': text_annotations[0].description
        }
    
    def _detect_text(self, image_path: str):
        """Detect text using Vision API."""
        with suppress_stderr_warnings():
            client = vision.ImageAnnotatorClient()
        
        with open(image_path, 'rb') as f:
            content = f.read()
        
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        
        if not response.text_annotations:
            print("📐 No text found with document detection, trying basic detection...")
            response = client.text_detection(image=image)
        
        return response.text_annotations
    
    def _print_orientation_info(self, text_lines: List[Dict]):
        """Print information about detected text orientations."""
        angles = set(line.get('angle', 0) for line in text_lines)
        if not angles:
            return
        
        angle_info = []
        for angle in sorted(angles):
            if abs(angle) < 5:
                angle_info.append("horizontal")
            elif abs(angle - 90) < 15:
                angle_info.append("vertical↑")
            elif abs(angle + 90) < 15:
                angle_info.append("vertical↓")
            elif abs(abs(angle) - 180) < 15:
                angle_info.append("upside-down")
            else:
                angle_info.append(f"{angle:.0f}°")
        
        if len(angle_info) > 1:
            print(f"📐 Text orientations: {', '.join(angle_info)}")
    
    def _show_results(self, annotated, phrase_matches):
        """Display annotated results."""
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f"Phrase Detection - {sum(len(m) for m in phrase_matches.values())} matches")
        plt.axis('off')
        
        legend = [f"{phrase}: {len(matches)}" 
                 for phrase, matches in phrase_matches.items()]
        if legend:
            plt.figtext(0.02, 0.02, '\n'.join(legend), fontsize=10, va='bottom')
        
        plt.tight_layout()
        plt.show()
