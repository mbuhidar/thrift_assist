"""Main OCR phrase detector with multi-provider support."""

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
from vision.providers import (
    GoogleVisionProvider,
    GeminiProvider,
    GEMINI_AVAILABLE
)

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


class VisionPhraseDetector:
    """Main class for OCR phrase detection with multiple provider support."""
    
    def __init__(self, config: VisionConfig = None):
        self.config = config or VisionConfig()
        self.config.setup_credentials()
        self.grouper = TextLineGrouper(self.config.angle_tolerance)
        self.matcher = PhraseMatcher(self.config)
        
        # Initialize OCR provider based on config
        self._setup_provider()
    
    def _setup_provider(self):
        """Initialize the OCR provider based on configuration."""
        # Use config provider (don't override with env var since we're passing it per-request)
        provider_name = str(self.config.ocr_provider).lower().strip()
        
        # Initialize provider attribute to None
        self.provider = None
        
        if provider_name == "gemini":
            if not GEMINI_AVAILABLE:
                print("⚠️ Gemini provider not available, falling back to "
                      "Google Vision")
                provider_name = "google"
            else:
                project_id = (self.config.google_cloud_project or
                             os.getenv('GOOGLE_CLOUD_PROJECT'))
                location = (self.config.google_cloud_location or
                           os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1'))
                model_name = (getattr(self.config, 'gemini_model', None) or
                             os.getenv('GEMINI_MODEL',
                                      'gemini-1.5-flash-001'))
                
                print(f"🔧 Gemini config: project_id={project_id!r}, location={location!r}, model={model_name!r}")
                
                self.provider = GeminiProvider(
                    project_id=project_id,
                    location=location,
                    model_name=model_name
                )
                if not self.provider.is_available():
                    print("⚠️ Gemini provider not configured (missing "
                          "Google Cloud project), falling back to Google Vision")
                    provider_name = "google"
        
        if provider_name == "google" or self.provider is None:
            self.provider = GoogleVisionProvider(
                credentials_path=self.config.google_credentials_path
            )
            if not self.provider.is_available():
                print("❌ Google Vision provider not available - check "
                      "credentials")
        
        if self.provider:
            print(f"✅ Using OCR provider: {self.provider.name}")
        else:
            print("❌ No OCR provider available")
    
    def detect(
        self, image_path: str, search_phrases: List[str],
        threshold: int = None, show_plot: bool = True,
        text_scale: int = None
    ) -> Optional[Dict]:
        """Detect and annotate phrases in image."""
        threshold = threshold or self.config.fuzz_threshold
        text_scale = text_scale or self.config.default_text_scale
        
        # Load image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            # Create empty image for error case
            import numpy as np
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            return {
                'image': dummy_image,
                'annotated_image': dummy_image.copy(),
                'matches': {},
                'total_matches': 0,
                'all_text': 'Image file not found'
            }
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            import numpy as np
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            return {
                'image': dummy_image,
                'annotated_image': dummy_image.copy(),
                'matches': {},
                'total_matches': 0,
                'all_text': 'Could not load image'
            }
        
        # Detect text using selected provider
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
            matches = self.matcher.find_matches(
                phrase, text_lines,
                text_annotations[0].description, threshold
            )
            if matches:
                phrase_matches[phrase] = matches
                print(f"🎯 Found {len(matches)} matches for '{phrase}'")
        
        if not phrase_matches:
            print("❌ No phrase matches found.")
            return {
                'image': image,
                'annotated_image': image.copy(),
                'matches': {},
                'total_matches': 0,
                'all_text': text_annotations[0].description
            }
        
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
        """Detect text using the configured OCR provider."""
        try:
            full_text, annotations = self.provider.detect_text(image_path)
            
            # Convert provider-specific annotations to format expected
            # by rest of the code. For backward compatibility, create a
            # list where first element is full text object
            class TextAnnotationWrapper:
                def __init__(self, text):
                    self.description = text
            
            result = [TextAnnotationWrapper(full_text)]
            
            # Add individual text annotations
            for ann in annotations:
                class DetailedAnnotation:
                    def __init__(self, text_ann):
                        self.description = text_ann.text
                        self.locale = text_ann.locale
                        
                        # Convert bounding box to Google Vision format
                        class BoundingPoly:
                            def __init__(self, coords):
                                class Vertex:
                                    def __init__(self, x, y):
                                        self.x = int(x)
                                        self.y = int(y)
                                self.vertices = [
                                    Vertex(x, y) for x, y in coords
                                ]
                        
                        self.bounding_poly = BoundingPoly(
                            text_ann.bounding_box
                        )
                
                result.append(DetailedAnnotation(ann))
            
            return result
            
        except Exception as e:
            print(f"❌ OCR detection error: {e}")
            return []
    
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
        total_matches = sum(len(m) for m in phrase_matches.values())
        plt.title(f"Phrase Detection - {total_matches} matches")
        plt.axis('off')
        
        legend = [
            f"{phrase}: {len(matches)}"
            for phrase, matches in phrase_matches.items()
        ]
        if legend:
            plt.figtext(
                0.02, 0.02, '\n'.join(legend), fontsize=10, va='bottom'
            )
        
        plt.tight_layout()
        plt.show()
