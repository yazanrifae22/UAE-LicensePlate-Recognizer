import cv2
import numpy as np
from typing import List

# Original functions
def next_multiple_of_32(n: int) -> int:
    return (n + 31) // 32 * 32

def sort_by_x(e):
    return e['x']

def sort_by_y(e):
    return e['y']

# Function for character segmentation
def segment_characters(plate_image):
    """
    Segment characters from a license plate image.

    Args:
        plate_image (numpy.ndarray): The license plate image.

    Returns:
        List[numpy.ndarray]: List of segmented character images.
    """
    # Preprocess the plate image (e.g., grayscale, thresholding, etc.)
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise)
    min_contour_area = 100  # Adjust this threshold as needed
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    # Sort contours from left to right based on their x-coordinate
    filtered_contours.sort(key=lambda contour: cv2.boundingRect(contour)[0])

    # Extract and return segmented character images
    segmented_characters = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        character_image = binary_plate[y:y+h, x:x+w]
        segmented_characters.append(character_image)

    return segmented_characters

# Function for license plate resizing
def resize_plate(plate_image, width, height):
    """
    Resize a license plate image to a specified width and height.

    Args:
        plate_image (numpy.ndarray): The license plate image.
        width (int): The target width.
        height (int): The target height.

    Returns:
        numpy.ndarray: The resized license plate image.
    """
    return cv2.resize(plate_image, (width, height))

# Function for applying image filters (e.g., blur, sharpening)
def apply_filter(image, filter_type='blur'):
    """
    Apply an image filter to the input image.

    Args:
        image (numpy.ndarray): The input image.
        filter_type (str): The type of filter to apply ('blur', 'sharpen', etc.).

    Returns:
        numpy.ndarray: The filtered image.
    """
    if filter_type == 'blur':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'sharpen':
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    else:
        raise ValueError("Invalid filter type. Supported types: 'blur', 'sharpen'")

