"""
Perspective Transformation Module for Document Processing

This module provides functionality for correcting perspective distortion in document images,
particularly useful for processing scanned or photographed ID cards and documents.

Key Features:
- Automatic corner point ordering for quadrilateral shapes
- Perspective transformation to create top-down "bird's eye" view
- Optimized for document scanning and OCR preprocessing

Typical workflow:
1. Detect document corners from edge detection or contour analysis
2. Order the corner points consistently (top-left, top-right, bottom-right, bottom-left)
3. Apply perspective transformation to create a rectangular, undistorted view

Dependencies:
- OpenCV for perspective transformation operations
- NumPy for mathematical operations and array handling
"""

import numpy as np
import cv2


def sort_quadrilateral_corners(corner_points):
    """
    Sort four corner points of a quadrilateral in consistent clockwise order.
    
    This function takes an unordered set of 4 corner points and arranges them
    in a consistent order: top-left, top-right, bottom-right, bottom-left.
    
    The sorting algorithm uses mathematical properties:
    - Top-left: smallest sum of x+y coordinates
    - Bottom-right: largest sum of x+y coordinates  
    - Top-right: smallest difference of x-y coordinates
    - Bottom-left: largest difference of x-y coordinates
    
    Args:
        corner_points (numpy.ndarray): Array of shape (4, 2) containing 
                                     four (x, y) coordinate pairs
    
    Returns:
        numpy.ndarray: Ordered array of shape (4, 2) with corners arranged as:
                      [top-left, top-right, bottom-right, bottom-left]
    
    Example:
        >>> corners = np.array([[100, 200], [300, 50], [350, 250], [80, 300]])
        >>> ordered = sort_quadrilateral_corners(corners)
        >>> print(ordered)  # [[top-left], [top-right], [bottom-right], [bottom-left]]
    """
    # Initialize output array for ordered coordinates
    ordered_rectangle = np.zeros((4, 2), dtype="float32")
    
    # Calculate sum of coordinates (x + y) for each point
    # Top-left will have minimum sum, bottom-right will have maximum sum
    coordinate_sums = corner_points.sum(axis=1)
    ordered_rectangle[0] = corner_points[np.argmin(coordinate_sums)]  # Top-left
    ordered_rectangle[2] = corner_points[np.argmax(coordinate_sums)]  # Bottom-right
    
    # Calculate difference of coordinates (x - y) for each point
    # Top-right will have minimum difference, bottom-left will have maximum difference
    coordinate_differences = np.diff(corner_points, axis=1)
    ordered_rectangle[1] = corner_points[np.argmin(coordinate_differences)]  # Top-right
    ordered_rectangle[3] = corner_points[np.argmax(coordinate_differences)]  # Bottom-left
    
    return ordered_rectangle

def apply_perspective_correction(input_image, quadrilateral_corners, output_size=None):
    """
    Apply perspective transformation to correct document distortion.
    
    This function takes a distorted document image and four corner points,
    then applies perspective transformation to create a rectangular,
    top-down view suitable for OCR and further processing.
    
    The transformation process:
    1. Orders the corner points consistently
    2. Calculates output dimensions from the corners or an explicitly supplied
       canonical card size
    3. Creates a perspective transformation matrix
    4. Applies the transformation to produce a corrected image
    
    Args:
        input_image (numpy.ndarray): Source image containing the distorted document
        quadrilateral_corners (numpy.ndarray): Array of shape (4, 2) containing
                                             four corner points of the document
    
    Returns:
        numpy.ndarray: Perspective-corrected image with rectangular shape
    
    Raises:
        ValueError: If corner points are invalid or image cannot be processed
    
    Example:
        >>> import cv2
        >>> image = cv2.imread('distorted_document.jpg')
        >>> corners = np.array([[50, 100], [400, 80], [420, 300], [30, 320]])
        >>> corrected = apply_perspective_correction(image, corners)
        >>> cv2.imwrite('corrected_document.jpg', corrected)
    """
    # Sort corner points in consistent order: top-left, top-right, bottom-right, bottom-left
    ordered_corners = sort_quadrilateral_corners(quadrilateral_corners)
    top_left, top_right, bottom_right, bottom_left = ordered_corners
    
    # Calculate optimal output width by measuring both top and bottom edges
    # Use the maximum distance to preserve all content
    bottom_edge_width = np.sqrt(
        ((bottom_right[0] - bottom_left[0]) ** 2) + 
        ((bottom_right[1] - bottom_left[1]) ** 2)
    )
    top_edge_width = np.sqrt(
        ((top_right[0] - top_left[0]) ** 2) + 
        ((top_right[1] - top_left[1]) ** 2)
    )
    optimal_output_width = max(int(bottom_edge_width), int(top_edge_width))
    
    # Calculate optimal output height by measuring both left and right edges
    # Use the maximum distance to preserve all content
    right_edge_height = np.sqrt(
        ((top_right[0] - bottom_right[0]) ** 2) + 
        ((top_right[1] - bottom_right[1]) ** 2)
    )
    left_edge_height = np.sqrt(
        ((top_left[0] - bottom_left[0]) ** 2) + 
        ((top_left[1] - bottom_left[1]) ** 2)
    )
    optimal_output_height = max(int(right_edge_height), int(left_edge_height))
    
    if output_size is None:
        output_width = max(1, optimal_output_width)
        output_height = max(1, optimal_output_height)
    else:
        output_width, output_height = map(int, output_size)
        if output_width < 2 or output_height < 2:
            raise ValueError("output_size dimensions must both be at least 2 pixels")

    # Map the detected quadrilateral directly into the requested output size.
    # This avoids an intermediate warp followed by a second, potentially
    # distorting resize.
    destination_corners = np.array([
        [0, 0],                                                    # Top-left
        [output_width - 1, 0],                                    # Top-right
        [output_width - 1, output_height - 1],                    # Bottom-right
        [0, output_height - 1]                                    # Bottom-left
    ], dtype="float32")
    
    # Calculate perspective transformation matrix
    # This matrix maps the distorted quadrilateral to a perfect rectangle
    transformation_matrix = cv2.getPerspectiveTransform(
        ordered_corners, 
        destination_corners
    )
    
    # Apply perspective transformation to create corrected image
    perspective_corrected_image = cv2.warpPerspective(
        input_image, 
        transformation_matrix, 
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,           # Use linear interpolation for smooth results
        borderMode=cv2.BORDER_REPLICATE   # Replicate border pixels to fill gaps
    )
    
    return perspective_corrected_image


