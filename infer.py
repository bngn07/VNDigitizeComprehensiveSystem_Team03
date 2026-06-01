import cv2
from src.ocr.ocr import OCRPipeline

# 1. Initialize the pipeline (using PaddleOCR)
pipeline = OCRPipeline(engine="paddle", threshold=0.8)

# 2. Load your image using OpenCV
image = cv2.imread(r"data/input/preprocess/images/0057.png")

# 3. Run inference on the image
result = pipeline.run(image)

# 4. View the results
print(result)