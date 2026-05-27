
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from ocr.tesseract import TesseractOCR, TesseractModel, TesseractParams
from ocr.adapter import OCRAdapter
from extraction.extractor import DocumentExtractor


# list ảnh
IMAGE_PATHS = [
    "dataset/dataset/27.png",
    "dataset/dataset/1.png",
    "dataset/dataset/18.png",
    "dataset/dataset/20.png",
    #"dataset/dataset/30.png",
    "dataset/dataset/img1.png",
    "dataset/dataset/img3.png",
    "dataset/dataset/img14.png",
    #"dataset/dataset/img31.jpg",
]

params    = TesseractParams(language="vie")
model     = TesseractModel(params=params)
engine    = TesseractOCR(model=model)
adapter   = OCRAdapter()
extractor = DocumentExtractor()

print(f"Tổng: {len(IMAGE_PATHS)} ảnh\n")
print("=" * 70)
for i, path in enumerate(IMAGE_PATHS, 1):
    image = cv2.imread(path)
    if image is None:
        print(f"[{i}]  {os.path.basename(path)} — không đọc được")
        print("-" * 70)
        continue

    group_result = engine.recognize(image)
    my_result    = adapter.convert(group_result)
    fields       = extractor.extract(my_result.raw_text, words=my_result.words)

    print(f"[{i}] {os.path.basename(path)}")
    
    # In raw text 400 ký tự đầu để debug
    print(f"     RAW: {my_result.raw_text[:400]}")
    print()
    
    for key, value in fields.items():
        label  = key.replace("_", " ").title()
        output = value if value else "Không xác định"
        print(f"     {label:20s}: {output}")
    print("-" * 70)