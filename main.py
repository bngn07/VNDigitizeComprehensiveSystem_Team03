from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from digitize import Digitize
from src.ocr.paddle import Paddle
from src.ocr.tesseract import TesseractOCR

IMAGE_PATH = Path("data/input/43.png")
OUTPUT_DIR = Path("output")
OCR_ENGINE = "paddle"


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
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    print(f"Loaded image: {IMAGE_PATH}")
    print(f"Image shape: {image.shape}")

    digitizer = Digitize(
        ocr=build_ocr(),
        config={
            "preprocessing": {
                "decide_engine": {"provider": None},
            },
            "autocorrect": {
                "enabled": True,
                "min_confidence": 0.7,
            },
            "extraction": {
                "enabled": False,
            },
        },
    )

    result = digitizer.digitize(image)

    print("=== DIGITIZE RESULT ===")
    print("=== PREPROCESS RESULT ===")
    print(result.preprocess)

    print("=== PREPROCESS METADATA ===")
    print(result.preprocess.metadata)

    print("=== OCR RESULT ===")
    print(result.ocr)

    output_path = OUTPUT_DIR / "preprocessed.png"
    save_image(result.preprocessed_image, output_path)
    print(f"Saved preprocessed image to: {output_path}")


if __name__ == "__main__":
    main()
