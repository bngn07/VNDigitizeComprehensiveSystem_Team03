
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
from ocr.tesseract import TesseractOCR, TesseractModel, TesseractParams
from ocr.adapter import OCRAdapter
from ocr.confidence import ConfidenceScorer

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ====================================================================
# CẤU HÌNH
# ====================================================================
IMAGE_PATH = "27.png"       # ← đổi thành đường dẫn ảnh của bạn
THRESHOLD  = 0.8                    # ngưỡng flag

# ====================================================================
# CHẠY OCR
# ====================================================================
print("=== BƯỚC 1: Chạy OCR ===")
params = TesseractParams(language="vie")
model  = TesseractModel(params=params)
engine = TesseractOCR(model=model)

image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"❌ Không đọc được ảnh: {IMAGE_PATH}")
    sys.exit(1)

group_result = engine.recognize(image)
print(group_result)
print()

# ====================================================================
# CONVERT SANG FORMAT TASK 3
# ====================================================================
print("=== BƯỚC 2: Convert sang WordResult ===")
adapter   = OCRAdapter()
my_result = adapter.convert(group_result)
print(f"Tổng số từ sau convert: {len(my_result.words)}")
print()

# ====================================================================
# CHẤM ĐIỂM TIN CẬY
# ====================================================================
print("=== BƯỚC 3: Chấm điểm tin cậy ===")
scorer = ConfidenceScorer(threshold=THRESHOLD)
final  = scorer.score(my_result)
print(final)
print()
print("SUMMARY:", scorer.summary(final))