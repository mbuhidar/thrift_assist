"""Text line grouping utilities for OCR."""

import logging
from typing import List, Dict, Optional, Tuple
# Changed to absolute import
from utils.geometry_utils import calculate_text_angle

logger = logging.getLogger(__name__)


class TextLineGrouper:
    """Groups text annotations into logical lines."""
    
    def __init__(self, angle_tolerance: int = 15):
        self.angle_tolerance = angle_tolerance
    
    def group(self, text_annotations) -> List[Dict]:
        """
        Group individual text annotations into text lines based on proximity and angle.
        Handles text at various orientations including upside down.
        """
        if len(text_annotations) <= 1:
            return []
        
        word_annotations = text_annotations[1:]
        angle_groups = self._group_by_angle(word_annotations)
        
        all_text_lines = []
        for angle, annotations in angle_groups.items():
            lines = self._group_by_position(annotations, angle)
            all_text_lines.extend(self._convert_to_text_lines(lines, angle))
        
        return all_text_lines
    
    def _group_by_angle(self, annotations) -> Dict[float, List]:
        """Group annotations by text angle."""
        angle_groups = {}
        
        for annotation in annotations:
            if not annotation.bounding_poly.vertices:
                continue
            
            angle = calculate_text_angle(annotation.bounding_poly.vertices)
            
            # Debug: Log first few angles
            if len(angle_groups) < 5:
                logger.info(f"Annotation '{annotation.description[:20]}': "
                           f"calculated angle = {angle:.2f}°")
            
            # Angles are now in 0-360 range from calculate_text_angle
            angle_key = self._find_angle_key(angle_groups.keys(), angle)
            if angle_key is None:
                angle_key = angle
                angle_groups[angle_key] = []
            
            angle_groups[angle_key].append(annotation)
        
        return angle_groups
    
    def _find_angle_key(self, existing_angles, target_angle) -> Optional[float]:
        """Find existing angle within tolerance."""
        for angle in existing_angles:
            if abs(target_angle - angle) <= self.angle_tolerance:
                return angle
        return None
    
    def _group_by_position(self, annotations, angle: float) -> Dict[float, List]:
        """Group annotations by position considering rotation."""
        lines = {}
        tolerance = 20
        
        for annotation in annotations:
            y_pos, sort_key = self._calculate_position(annotation, angle)
            
            line_key = self._find_line_key(lines.keys(), y_pos, tolerance)
            if line_key is None:
                line_key = y_pos
                lines[line_key] = []
            
            lines[line_key].append((sort_key, annotation.description, annotation))
        
        return lines
    
    def _calculate_position(self, annotation, angle: float) -> Tuple[float, float]:
        """Calculate position and sort key based on text angle (0-360 range)."""
        vertices = annotation.bounding_poly.vertices
        
        # Horizontal text: 0° or 360° (±45°)
        if angle < 45 or angle > 315:
            return vertices[0].y, vertices[0].x
        # Vertical text (bottom left): 90° (±45°)
        elif 45 <= angle < 135:
            return vertices[0].x, -vertices[0].y
        # Upside down: 180° (±45°)
        elif 135 <= angle < 225:
            return -vertices[0].y, -vertices[0].x
        # Vertical text (bottom right): 270° (±45°)
        elif 225 <= angle < 315:
            return -vertices[0].x, vertices[0].y
        # Diagonal (shouldn't normally hit this)
        else:
            center_x = sum(v.x for v in vertices) / 4
            center_y = sum(v.y for v in vertices) / 4
            return center_y, center_x
    
    def _find_line_key(self, existing_positions, target_pos: float, tolerance: int) -> Optional[float]:
        """Find existing line within tolerance."""
        for pos in existing_positions:
            if abs(target_pos - pos) <= tolerance:
                return pos
        return None
    
    def _convert_to_text_lines(self, lines: Dict, angle: float) -> List[Dict]:
        """Convert grouped lines to text line dictionaries."""
        text_lines = []
        
        for line_pos in sorted(lines.keys()):
            line_words = sorted(lines[line_pos], key=lambda x: x[0])
            # Preserve original text case in the 'text' field for display
            line_text = ' '.join([word[1] for word in line_words]).strip()
            
            if line_text:
                text_lines.append({
                    'text': line_text,  # Original case preserved for labels
                    'annotations': [word[2] for word in line_words],
                    'y_position': line_pos,
                    'angle': angle
                })
        
        return text_lines
