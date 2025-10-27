"""Geometric utilities for bounding box operations."""

import math
from typing import Tuple, List


def calculate_text_angle(vertices) -> float:
    """
    Calculate the angle of text baseline from bounding box vertices.
    
    Google Vision API orders vertices starting from a corner and going
    clockwise or counter-clockwise. For text, the longest edges represent
    the text baseline and capline.
    
    Returns angle in degrees (0 = horizontal, positive = clockwise).
    """
    if len(vertices) < 2:
        return 0
    
    if len(vertices) < 4:
        # Fallback: simple calculation if we don't have all 4 vertices
        p1, p2 = vertices[0], vertices[1]
        angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        return math.degrees(angle)

    # Calculate all 4 edges
    edges = []
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 4]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = math.sqrt(dx * dx + dy * dy)
        edges.append((length, dx, dy))

    # Find the longest edge - this is parallel to text baseline
    longest = max(edges, key=lambda e: e[0])
    dx, dy = longest[1], longest[2]
    
    # Calculate angle from horizontal
    angle = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle)
    
    # Keep in 0-180 range:
    # For angles in range [-90, 90], take absolute value
    # For angles outside that range, normalize differently
    if -90 <= angle_degrees <= 90:
        angle_degrees = abs(angle_degrees)
    else:
        # angle is in range (90, 180] or [-180, -90)
        if angle_degrees < 0:
            angle_degrees += 360
        angle_degrees = angle_degrees - 180
    
    return angle_degrees


def rectangles_overlap(rect1: Tuple[int, int, int, int], 
                       rect2: Tuple[int, int, int, int]) -> bool:
    """Check if two rectangles overlap."""
    x1a, y1a, x2a, y2a = rect1
    x1b, y1b, x2b, y2b = rect2
    return not (x2a <= x1b or x2b <= x1a or y2a <= y1b or y2b <= y1a)


def find_non_overlapping_position(rect: Tuple[int, int, int, int],
                                  existing_positions: List[Tuple[int, int, int, int]],
                                  image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Enhanced position finder that better avoids overlaps with improved spacing."""
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    img_height, img_width = image_shape[:2]
    
    for existing_rect in existing_positions:
        if rectangles_overlap(rect, existing_rect):
            spacing = 15
            offsets = [
                (0, -height - spacing), (0, height + spacing),
                (-width - spacing, 0), (width + spacing, 0),
                (-width - spacing, -height - spacing), (width + spacing, -height - spacing),
                (-width - spacing, height + spacing), (width + spacing, height + spacing),
                (-width//2, -height - spacing), (width//2, -height - spacing),
                (-width//2, height + spacing), (width//2, height + spacing),
            ]
            
            for dx, dy in offsets:
                new_x1, new_y1 = x1 + dx, y1 + dy
                margin = 5
                
                if (margin <= new_x1 <= img_width - width - margin and 
                    margin <= new_y1 <= img_height - height - margin):
                    new_rect = (new_x1, new_y1, new_x1 + width, new_y1 + height)
                    if not any(rectangles_overlap(new_rect, ex) for ex in existing_positions):
                        return new_rect
            
            return _find_stacked_position(rect, existing_positions, image_shape)
    
    margin = 5
    x1 = max(margin, min(x1, img_width - width - margin))
    y1 = max(margin, min(y1, img_height - height - margin))
    return (x1, y1, x1 + width, y1 + height)


def _find_stacked_position(rect, existing_positions, image_shape):
    """Find position by stacking labels vertically within image bounds."""
    x1, y1, x2, y2 = rect
    width, height = x2 - x1, y2 - y1
    img_height, img_width = image_shape[:2]
    
    max_y = max((ex[3] for ex in existing_positions 
                if not (x2 <= ex[0] or ex[2] <= x1)), default=0)
    
    new_y1 = min(max_y + 5, img_height - height - 5)
    new_x1 = max(0, min(x1, img_width - width))
    
    if new_y1 <= 0:
        max_x = max((ex[2] for ex in existing_positions 
                    if not (y2 <= ex[1] or ex[3] <= y1)), default=0)
        new_x1 = min(max_x + 5, img_width - width - 5)
        new_y1 = max(0, min(y1, img_height - height))
    
    return (new_x1, new_y1, new_x1 + width, new_y1 + height)
