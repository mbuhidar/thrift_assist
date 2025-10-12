"""Image annotation utilities for drawing phrase bounding boxes."""

import cv2
from typing import Dict, List, Tuple

# Changed to absolute imports
from utils.geometry_utils import rectangles_overlap, find_non_overlapping_position
from utils.text_utils import normalize_text_for_search


class ImageAnnotator:
    """Handles drawing annotations on images."""
    
    def __init__(self, text_scale: int = 100):
        self.text_scale = text_scale
        self.default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
    
    def draw_annotations(self, image, phrase_matches: Dict, 
                        phrase_colors: Dict = None) -> 'np.ndarray':
        """Draw bounding boxes around detected phrases with smart label placement."""
        annotated = image.copy()
        
        if not phrase_matches:
            return annotated
        
        # Setup colors
        if phrase_colors is None:
            phrase_colors = {phrase: self.default_colors[i % len(self.default_colors)]
                           for i, phrase in enumerate(phrase_matches.keys())}
        
        label_positions = []
        
        for phrase, matches in phrase_matches.items():
            color = phrase_colors.get(phrase, (255, 0, 0))
            
            for match_data, score, match_type in matches:
                bbox = self._extract_bbox(match_data, phrase)
                if bbox is None:
                    continue
                
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                self._draw_label(annotated, phrase, score, (x1, y1, x2, y2),
                               color, label_positions)
        
        return annotated
    
    def _extract_bbox(self, match_data: Dict, phrase: str) -> Tuple:
        """Extract tight bounding box from match data."""
        if 'annotations' not in match_data or not match_data['annotations']:
            return None
        
        phrase_words = normalize_text_for_search(phrase).split()
        matched_annotations = []
        
        for annotation in match_data['annotations']:
            ann_text = normalize_text_for_search(annotation.description)
            if any(word in ann_text or ann_text in word for word in phrase_words):
                matched_annotations.append(annotation)
        
        target = matched_annotations if matched_annotations else match_data['annotations']
        
        all_x, all_y = [], []
        for ann in target:
            if hasattr(ann, 'bounding_poly') and ann.bounding_poly.vertices:
                for v in ann.bounding_poly.vertices:
                    all_x.append(v.x)
                    all_y.append(v.y)
        
        if not all_x or not all_y:
            return None
        
        padding = 3
        return (min(all_x) - padding, min(all_y) - padding,
                max(all_x) + padding, max(all_y) + padding)
    
    def _draw_label(self, image, phrase: str, score: float, bbox: Tuple,
                   color: Tuple, label_positions: List):
        """Draw label with background and leader line if needed."""
        x1, y1, x2, y2 = bbox
        label = f"{phrase} ({score:.0f}%)"
        
        # Calculate text size
        font_scale = 0.8 * (self.text_scale / 100.0)
        thickness = max(1, int(2 * (self.text_scale / 100.0)))
        
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Find position
        label_x, label_y = self._find_label_position(
            bbox, text_w, text_h, image.shape, label_positions)
        
        # Check if leader line needed
        label_rect = (label_x, label_y - text_h - 5, label_x + text_w + 10, label_y + 5)
        adj_rect = find_non_overlapping_position(label_rect, label_positions, image.shape)
        
        if adj_rect != label_rect or rectangles_overlap(label_rect, bbox):
            # Draw leader line
            box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            label_center = (adj_rect[0] + text_w // 2, adj_rect[1] + text_h // 2)
            
            if label_center[0] < x1:
                line_start = (x1, max(y1, min(y2, label_center[1])))
            elif label_center[0] > x2:
                line_start = (x2, max(y1, min(y2, label_center[1])))
            elif label_center[1] < y1:
                line_start = (max(x1, min(x2, label_center[0])), y1)
            else:
                line_start = (max(x1, min(x2, label_center[0])), y2)
            
            cv2.line(image, line_start, label_center, color, 1)
            label_x, label_y = adj_rect[0], adj_rect[1] + text_h
        
        # Draw background
        padding = 2
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - 5 - padding),
                     (label_x + text_w + 10 + padding, label_y + 5 + padding),
                     color, -1)
        
        # Draw text
        cv2.putText(image, label, (label_x + 5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Store position
        label_positions.append((label_x, label_y - text_h - 5,
                              label_x + text_w + 10, label_y + 5))
    
    def _find_label_position(self, bbox: Tuple, text_w: int, text_h: int,
                           img_shape: Tuple, existing_labels: List) -> Tuple[int, int]:
        """Find optimal label position avoiding overlaps."""
        x1, y1, x2, y2 = bbox
        height, width = img_shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        
        positions = [
            # Above centered (close)
            (max(5, min(x1 + (box_w - text_w) // 2, width - text_w - 5)),
             max(text_h + 10, y1 - text_h - 15)),
            # Below centered (close)
            (max(5, min(x1 + (box_w - text_w) // 2, width - text_w - 5)),
             min(height - 15, y2 + text_h + 15)),
            # Left
            (max(5, x1 - text_w - 15),
             max(text_h + 10, min(y1 + (box_h + text_h) // 2, height - 10))),
            # Right
            (min(width - text_w - 5, x2 + 15),
             max(text_h + 10, min(y1 + (box_h + text_h) // 2, height - 10))),
        ]
        
        for pos_x, pos_y in positions:
            if not (5 <= pos_x <= width - text_w - 5 and
                   text_h + 10 <= pos_y <= height - 10):
                continue
            
            test_rect = (pos_x, pos_y - text_h - 5, pos_x + text_w + 10, pos_y + 5)
            if not rectangles_overlap(test_rect, bbox):
                return pos_x, pos_y
        
        # Fallback
        return max(5, min(x1, width - text_w - 15)), max(text_h + 10, y1 - text_h - 10)