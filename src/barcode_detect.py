import cv2
import numpy as np
from typing import Optional

def barcode_detect(
        image: np.ndarray
) -> Optional[np.ndarray]:
    '''
    Detecting barcodes that is cropped (no outside edge)

    Args:
        - image: the output of cv2.imread, a matrix image

    Output: 4 points of the barcode box if it exists
                or None if there's none of them
    '''
     
    if image is None:
        return None

    bardet = cv2.barcode_BarcodeDetector()

    retval, decoded_info, points = bardet.detectAndDecode(image)

    if retval and points is not None:
        return points
    else: return None