#!/usr/bin/env python3
"""Debug script to test angle calculation with sample vertices."""

import math

def calculate_text_angle_old(vertices):
    """Old implementation."""
    if len(vertices) < 4:
        return 0
    
    edges = []
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 4]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = math.sqrt(dx * dx + dy * dy)
        edges.append((length, dx, dy))
    
    longest = max(edges, key=lambda e: e[0])
    dx, dy = longest[1], longest[2]
    
    angle = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle)
    
    if -90 <= angle_degrees <= 90:
        angle_degrees = abs(angle_degrees)
    else:
        if angle_degrees < 0:
            angle_degrees += 360
        angle_degrees = angle_degrees - 180
    
    return angle_degrees


def calculate_text_angle_new(vertices):
    """New implementation - 0-360 range."""
    if len(vertices) < 4:
        return 0.0
    
    edges = []
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 4]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = math.sqrt(dx * dx + dy * dy)
        edges.append((length, dx, dy))
    
    longest = max(edges, key=lambda e: e[0])
    dx, dy = longest[1], longest[2]
    
    angle = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle)
    
    # Normalize to 0-360 range
    normalized_angle = angle_degrees % 360
    
    return normalized_angle


# Test with sample vertices
class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Test cases
test_cases = [
    ("Horizontal text", [Vertex(10, 10), Vertex(100, 10), Vertex(100, 20), Vertex(10, 20)]),
    ("Vertical text (90°)", [Vertex(10, 10), Vertex(20, 10), Vertex(20, 100), Vertex(10, 100)]),
    ("Upside down (180°)", [Vertex(100, 20), Vertex(10, 20), Vertex(10, 10), Vertex(100, 10)]),
    ("Vertical text (270°)", [Vertex(20, 100), Vertex(10, 100), Vertex(10, 10), Vertex(20, 10)]),
    ("45° diagonal", [Vertex(10, 10), Vertex(100, 100), Vertex(90, 110), Vertex(0, 20)]),
]

print("Angle Calculation Comparison")
print("=" * 70)
for name, vertices in test_cases:
    old_angle = calculate_text_angle_old(vertices)
    new_angle = calculate_text_angle_new(vertices)
    print(f"{name:25} | Old: {old_angle:6.1f}° | New: {new_angle:6.1f}°")
