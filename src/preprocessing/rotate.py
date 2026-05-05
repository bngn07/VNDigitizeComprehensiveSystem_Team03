import cv2
import pytesseract
from pytesseract import Output
from dataclasses import dataclass

@dataclass 
class RotationResult:
    angle: float
    confidence: float
    script: str = "Unknown"

class RotationDetector:
    def __init__(self, config: str = '--psm 0', tesseract_path: str = None):
        self.config = config
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def detect(self, image) -> RotationResult:
        osd_data = pytesseract.image_to_osd(
            image, 
            output_type=Output.DICT, 
            config=self.config
        )
        
        angle = float(osd_data.get('rotate', 0.0))
        confidence = float(osd_data.get('orientation_conf', 0.0))
        script = str(osd_data.get('script', 'Unknown'))
        
        return RotationResult(
            angle = angle, 
            confidence = confidence / 100.0, 
            # convert to real percentage, not value in [0,100] (%) 
            script = script
        )