from __future__ import annotations
import os
import sys
from pathlib import Path

# from intelligence import rag

# 1. Ép tất cả các thư viện dùng chung bộ xử lý Python thuần, bỏ qua lỗi phiên bản Descriptor
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 2. Khóa các tiến trình ẩn của Torch gây lỗi shm.dll cũ
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from unittest import result
import cv2
import matplotlib

# from intelligence import rag
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from digitize import Digitize
from ocr.tesseract import TesseractOCR
from ocr.paddle import Paddle
from src.intelligence.rag import RAGPipeline



IMAGE_PATH = "data/input/43.png"
OUTPUT_DIR = Path("output")
OCR_ENGINE = "paddle"   # "tesseract" | "paddle"


def save_image(image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(
        image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        cmap="gray" if image.ndim == 2 else None,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_ocr():
    if OCR_ENGINE == "paddle":
        return Paddle()
    return TesseractOCR()


def main() -> None:
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    print(f"Loaded image: {IMAGE_PATH}")
    print(f"Image shape: {img.shape}")

    digitizer = Digitize(
        ocr=build_ocr(),
        config={
            "preprocessing": {
                "decide_engine": {"provider": None}
            },
            "postprocessing": {
                "autocorrect": {
                    "enabled": True,
                    "min_confidence": 0.7
                }
            }
        },
    )

    result = digitizer.digitize(img)

    print("=== DIGITIZE RESULT ===")
    print("=== PREPROCESS RESULT ===")
    print(result.preprocess)

    print("=== PREPROCESS METADATA ===")
    print(result.preprocess.metadata)

    print("=== OCR RESULT ===")
    print(result.ocr)

    print("\n=== BUILDING RAG ===")
    rag = RAGPipeline()
    full_text = "\n".join(
        block.text
        for block in result.ocr.texts
    )   

    rag.build(full_text)
    print("RAG initialized successfully")

    questions = [
    "Ai là người kháng cáo?",
    "Số tiền thuế là bao nhiêu?",
    "Cơ quan nào xử lý vụ việc?"
    ]

    for q in questions:

        print("\n")
        print("=" * 60)

        print("Question:", q)

        answer = rag.ask(q)

        print("Answer:")
        print(answer)

    output_path = OUTPUT_DIR / "preprocessed.png"
    save_image(result.image, output_path)
    print(f"Saved preprocessed image to: {output_path}")


if __name__ == "__main__":
    main()