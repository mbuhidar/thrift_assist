"""Image annotation utilities for drawing phrase bounding boxes."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

# Changed to absolute imports
from utils.geometry_utils import rectangles_overlap, find_non_overlapping_position
from utils.text_utils import normalize_text_for_search


class ImageAnnotator:
    """Handles drawing annotations on images."""
    
    def __init__(self, text_scale: int = 100):
        self.text_scale = text_scale
        # High contrast, bright colors for better visibility (values BGR)
        self.default_colors = [
            (200, 200, 0),     # Cyan (darker)
            (128, 128, 0),     # Teal
            (255, 0, 255),     # Magenta
            (0, 200, 0),       # Lime green (darker)
            (0, 128, 255),     # Orange
            (255, 0, 128),     # Purple/Violet (high blue, low red)
            (128, 0, 255),     # Hot pink (high red, low blue)
            (0, 255, 128),     # Spring green
            (255, 64, 255)     # Neon pink/fuchsia
        ]
    
    def draw_annotations(self, image, phrase_matches: Dict, 
                        phrase_colors: Dict = None) -> 'np.ndarray':
        """Draw bounding boxes around detected phrases with smart label placement."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        annotated = image.copy()
        
        if not phrase_matches:
            return annotated
        
        # Setup colors
        if phrase_colors is None:
            phrase_colors = {phrase: self.default_colors[i % len(self.default_colors)]
                           for i, phrase in enumerate(phrase_matches.keys())}
        
        label_positions = []
        stats = {'drawn': 0, 'skipped': 0, 'fallback': 0}
        
        for phrase, matches in phrase_matches.items():
            if not matches:
                continue
            
            color = phrase_colors.get(phrase, (255, 0, 0))
            
            for match_data, score, match_type in matches:
                try:
                    # Validate match data structure
                    if not isinstance(match_data, dict):
                        stats['skipped'] += 1
                        continue
                    
                    # Extract bbox with fallback strategies
                    bbox = self._extract_bbox(match_data, phrase)
                    if bbox is None:
                        bbox = self._extract_fallback_bbox(match_data, annotated.shape)
                        if bbox is not None:
                            stats['fallback'] += 1
                    
                    if bbox is None:
                        stats['skipped'] += 1
                        continue
                    
                    # Validate and clip bbox to image bounds
                    bbox = self._validate_and_clip_bbox(bbox, annotated.shape)
                    if bbox is None:
                        stats['skipped'] += 1
                        continue
                    
                    # Determine line style based on validation status
                    # Increased box thickness for better visibility
                    is_validated = match_data.get('validated', True)
                    thickness = 4 if is_validated else 3
                    line_type = cv2.LINE_AA
                    
                    # Try to draw rotated bounding box if vertices available
                    if self._draw_rotated_bbox(annotated, match_data, phrase, color, thickness, line_type):
                        # Successfully drew rotated box
                        pass
                    else:
                        # Fallback to axis-aligned rectangle
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, line_type)
                    
                    # Draw label with metadata
                    self._draw_label(annotated, phrase, score, bbox, color, 
                                   label_positions, match_data, match_type)
                    
                    stats['drawn'] += 1
                    
                except Exception as e:
                    print(f"Warning: Failed to annotate '{phrase}': {str(e)}")
                    stats['skipped'] += 1
        
        # Draw summary overlay
        self._draw_summary(annotated, stats)
        
        return annotated
    
    def _validate_and_clip_bbox(self, bbox: Tuple[int, int, int, int], 
                                img_shape: Tuple) -> Optional[Tuple[int, int, int, int]]:
        """Validate and clip bounding box to image dimensions."""
        x1, y1, x2, y2 = bbox
        height, width = img_shape[:2]
        
        # Check for valid dimensions
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Check if completely outside bounds
        if x2 < 0 or x1 >= width or y2 < 0 or y1 >= height:
            return None
        
        # Clip to image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Re-validate after clipping
        if x2 <= x1 or y2 <= y1:
            return None
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def _extract_bbox(self, match_data: Dict, phrase: str) -> Optional[Tuple]:
        """Extract tight bounding box from match data."""
        if 'annotations' not in match_data or not match_data['annotations']:
            return None
        
        phrase_words = normalize_text_for_search(phrase).split()
        matched_annotations = []
        
        for annotation in match_data['annotations']:
            try:
                ann_text = normalize_text_for_search(annotation.description)
                if any(word in ann_text or ann_text in word for word in phrase_words):
                    matched_annotations.append(annotation)
            except (AttributeError, TypeError):
                continue
        
        target = matched_annotations if matched_annotations else match_data['annotations']
        
        all_x, all_y = [], []
        for ann in target:
            try:
                if hasattr(ann, 'bounding_poly') and ann.bounding_poly.vertices:
                    for v in ann.bounding_poly.vertices:
                        all_x.append(int(v.x))
                        all_y.append(int(v.y))
            except (AttributeError, TypeError, ValueError):
                continue
        
        if not all_x or not all_y:
            return None
        
        padding = 3
        return (min(all_x) - padding, min(all_y) - padding,
                max(all_x) + padding, max(all_y) + padding)
    
    def _extract_fallback_bbox(self, match_data: Dict, 
                               img_shape: Tuple) -> Optional[Tuple]:
        """Extract bounding box using fallback strategies when annotations unavailable."""
        height, width = img_shape[:2]
        
        # Strategy 1: Use y_position and text length
        if 'y_position' in match_data and 'text' in match_data:
            try:
                y_pos = int(match_data['y_position'])
                text_len = len(match_data['text'])
                
                # Estimate dimensions
                char_width = 12  # Average character width
                estimated_width = min(text_len * char_width, width - 40)
                estimated_height = 25
                
                x1 = 20
                y1 = max(5, y_pos - estimated_height // 2)
                x2 = min(x1 + estimated_width, width - 20)
                y2 = min(y1 + estimated_height, height - 5)
                
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2, y2)
            except (ValueError, TypeError):
                pass
        
        # Strategy 2: Use line_indices if available (for spanning matches)
        if 'span_info' in match_data:
            span_info = match_data['span_info']
            if 'line_indices' in span_info and span_info['line_indices']:
                # Create a box spanning estimated line height
                first_line_idx = span_info['line_indices'][0]
                num_lines = len(span_info['line_indices'])
                
                line_height = 30
                y1 = max(5, first_line_idx * line_height)
                y2 = min(y1 + num_lines * line_height, height - 5)
                x1 = 20
                x2 = width - 20
                
                return (x1, y1, x2, y2)
        
        return None
    
    def _draw_rotated_bbox(self, image: np.ndarray, match_data: Dict,
                          phrase: str, color: Tuple, thickness: int,
                          line_type) -> bool:
        """
        Draw a rotated bounding box using only the matched text vertices.
        Returns True if successful, False if vertices unavailable.
        """
        if 'annotations' not in match_data or not match_data['annotations']:
            return False
        
        # Filter annotations to only include words from the matched phrase
        phrase_words = normalize_text_for_search(phrase).split()
        matched_annotations = []
        
        for ann in match_data['annotations']:
            try:
                ann_text = normalize_text_for_search(ann.description)
                # Only include annotations that are part of the phrase
                if any(word in ann_text or ann_text in word 
                       for word in phrase_words):
                    matched_annotations.append(ann)
            except (AttributeError, TypeError):
                continue
        
        # Use matched annotations if found, otherwise fall back to all
        target_annotations = (matched_annotations if matched_annotations 
                            else match_data['annotations'])
        
        # Collect vertices only from matched text
        all_vertices = []
        for ann in target_annotations:
            try:
                if (hasattr(ann, 'bounding_poly') and 
                    ann.bounding_poly.vertices):
                    vertices = ann.bounding_poly.vertices
                    if len(vertices) >= 4:
                        # Convert vertices to numpy array
                        points = np.array([
                            [int(v.x), int(v.y)] for v in vertices[:4]
                        ], dtype=np.int32)
                        all_vertices.append(points)
            except (AttributeError, TypeError, ValueError):
                continue
        
        if not all_vertices:
            return False
        
        # If single word, draw its rotated box
        if len(all_vertices) == 1:
            cv2.polylines(image, [all_vertices[0]], True, color,
                         thickness, line_type)
            return True
        
        # For multiple words, create minimal rotated bounding rectangle
        try:
            all_points = np.vstack(all_vertices)
            # Get minimum area rotated rectangle
            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.polylines(image, [box], True, color, thickness, line_type)
            return True
        except Exception:
            # If minAreaRect fails, draw individual boxes for each word
            for vertices in all_vertices:
                cv2.polylines(image, [vertices], True, color,
                             thickness, line_type)
            return True
    
    def _draw_label(self, image, phrase: str, score: float, bbox: Tuple,
                   color: Tuple, label_positions: List, match_data: Dict = None,
                   match_type: str = None):
        """Draw label with background and leader line if needed."""
        x1, y1, x2, y2 = bbox
        
        # Build label - just phrase and score, no extra indicators
        label = f"{phrase} ({score:.0f}%)"
        
        # Calculate text size
        font_scale = 0.7 * (self.text_scale / 100.0)
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
            label_center = (adj_rect[0] + text_w // 2, adj_rect[1] + text_h // 2)
            
            # Find closest point on bbox edge
            if label_center[0] < x1:
                line_start = (x1, max(y1, min(y2, label_center[1])))
            elif label_center[0] > x2:
                line_start = (x2, max(y1, min(y2, label_center[1])))
            elif label_center[1] < y1:
                line_start = (max(x1, min(x2, label_center[0])), y1)
            else:
                line_start = (max(x1, min(x2, label_center[0])), y2)
            
            cv2.line(image, line_start, label_center, color, 1, cv2.LINE_AA)
            label_x, label_y = adj_rect[0], adj_rect[1] + text_h
        
        # Draw background
        padding = 2
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - 5 - padding),
                     (label_x + text_w + 10 + padding, label_y + 5 + padding),
                     color, -1)
        
        # Draw text
        cv2.putText(image, label, (label_x + 5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 
                   thickness, cv2.LINE_AA)
        
        # Store position
        label_positions.append((label_x, label_y - text_h - 5,
                              label_x + text_w + 10, label_y + 5))
    
    def _draw_summary(self, image: np.ndarray, stats: Dict):
        """Draw summary overlay showing annotation statistics."""
        height, width = image.shape[:2]
        
        # Build summary text
        parts = [f"OK:{stats['drawn']}"]
        if stats['skipped'] > 0:
            parts.append(f"Skip:{stats['skipped']}")
        if stats['fallback'] > 0:
            parts.append(f"FB:{stats['fallback']}")
        
        summary = " | ".join(parts)
        
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(
            summary, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Position in bottom-right corner
        x = width - text_w - 15
        y = height - 10
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (x - 5, y - text_h - 5), 
                     (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Draw text
        cv2.putText(image, summary, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 
                   thickness, cv2.LINE_AA)
    
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