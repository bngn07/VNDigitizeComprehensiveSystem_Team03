import cv2
import numpy as np

# blank detection on the condition that the input image is dilated / grow or used proper technique

def is_blank(
        image: np.ndarray, 
        gray_threshold_value: float = 127,
        upper_threshold: float = 0.95,
        lower_threshold: float = 0.01
    ) -> bool:
    
    '''
    Detecting blank page that is cropped (no outside edge)

    Args:
        - image: the output of cv2.imread, a matrix image
        - gray_threshold_value: threshold of gray level for binary threshold
        - output_threshold_value: level of black (ratio of black pixels / total pixels) to consider blank

    Output: True if not enough black (ratio < output threshold) 
                or False if there's enough black (ratio > output threshold)
    '''

    if image is None or not isinstance(image, np.ndarray):
        return False
    
    # If RGB then convert to grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Binary threshold
    _, binary_image = cv2.threshold(
        gray_image, 
        gray_threshold_value, 
        255, 
        cv2.THRESH_BINARY
    )

    total_black = np.sum(binary_image == 0)
    total_pixel = binary_image.size

    black_ratio = (total_black / total_pixel)

    return black_ratio < lower_threshold or black_ratio > upper_threshold