import sys
from pathlib import Path
import cv2

from src.preprocessing.blank import BlankDetector
from src.preprocessing.code import CodeDetector


def run(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # 2. Blank detection
    # blank_detector = BlankDetector(model_path='/home/kunnic/code/VNDigitizeComprehensiveSystem_Team03/models/rf_blank_detector.joblib')
    # is_blank = blank_detector.is_blank(gray)
    # print(f"[Blank]  is_blank = {is_blank}")

    # if is_blank:
    #     print("  → Skip OCR, no codes to scan either.")
    #     return

    # 3. Code detection (QR + barcode)
    code_detector = CodeDetector()
    codes = code_detector.detect(gray)
    print(f"[Codes]  found {len(codes)} code(s)")
    for i, c in enumerate(codes, 1):
        print(f"  {i}. type={c['type']:<8} content={c['content'][:60]}")
        print(f"     bbox={c['bbox']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    run(sys.argv[1])