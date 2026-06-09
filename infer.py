import os
import sys
import json
import argparse
import pytesseract
import cv2

from src.preprocessing.preprocess import Preprocessing
from src.ocr.ocr import OCRPipeline
from src.postprocessing.autocorrect import AutoCorrector
from src.extraction.document import DocumentExtractor

DEBUG_OUTPUT_DIR = "data/output"


def log(step: str, message: str):
    print(f"[{step}] {message}")


def init_tesseract(tesseract_cmd=None):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return

    if sys.platform == "win32":
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_path):
            pytesseract.pytesseract.tesseract_cmd = default_path


def process_document(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    print(f"Processing: {file_path}")

    log("preprocess", "Preprocessing image...")
    preprocessor = Preprocessing()
    prep_result = preprocessor.process(file_path)

    qrcodes = prep_result.metadata.get("qrcodes", [])
    if qrcodes:
        log("preprocess", f"Found {len(qrcodes)} barcode(s)/QR code(s):")
        for i, qr in enumerate(qrcodes, 1):
            print(f"           {i}. Type={qr.type}, Content={qr.content}")
    else:
        log("preprocess", "No QR codes found")

    debug_image_path = os.path.join(DEBUG_OUTPUT_DIR, "debug_preprocessed.jpg")
    cv2.imwrite(debug_image_path, prep_result.image)
    log("preprocess", f"Saved preprocessed image -> {debug_image_path}")

    log("ocr", "Running OCR (Tesseract)...")
    ocr_pipeline = OCRPipeline(engine="tesseract", threshold=0.8)
    ocr_result = ocr_pipeline.run(prep_result.image)

    log("autocorrect", "Applying correction rules...")
    autocorrector = AutoCorrector(enabled=True)
    correction_result = autocorrector.correct_ocr_result(ocr_result)
    final_text = " ".join(correction_result.corrected_texts)

    log("extract", "Extracting document metadata...")
    extractor = DocumentExtractor()
    extracted_data = extractor.extract(raw_text=final_text, words=ocr_result.words)

    return extracted_data


def main():
    parser = argparse.ArgumentParser(description="Run the document processing pipeline on an image.")
    parser.add_argument("image_path", help="Path to the input image file")
    parser.add_argument("--tesseract-path", help="Path to tesseract executable")
    args = parser.parse_args()

    init_tesseract(args.tesseract_path)

    try:
        results = process_document(args.image_path)

        output_path = os.path.join(DEBUG_OUTPUT_DIR, "result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"\nResult saved -> {output_path}")
        print(json.dumps(results, ensure_ascii=False, indent=4))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()