import cv2
import numpy as np
from typing import Optional

def qr_detect(
        image: np.ndarray
) -> Optional[np.ndarray]:

    '''
    Detecting qr codes that is cropped (no outside edge)

    Args:
        - image: the output of cv2.imread, a matrix image

    Output: 4 points of the qr box if it exists
                or None if there's none of them
    '''

    if image is None:
        return None

    qcd = cv2.QRCodeDetector();

    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(image);

    if retval and points is not None:
        return points
    else: return None

