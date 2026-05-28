# Preprocessing Module

**Overview**
This module contains the code for cleaning up document images before we pass them to the OCR engine. Since real-world images can be pretty messy—like being blurry, poorly lit, or upside down—this module looks at the image first to figure out how bad it is. Then, it automatically picks the right set of OpenCV filters to fix it up. 

### What's inside:
* **`preprocess.py`**: The main script. It takes the raw image, decides what to do, and runs it through the pipeline. 
* **`decision.py`**: This acts like the brain. It looks at the image data and labels it as `CLEAN`, `HEAVY` (needs a lot of work), or `SKIP` (usually blank). It can use a trained model or some basic math rules if the model isn't there.
* **`features.py`**: Calculates the math behind the image, like how much white space there is or how blurry the edges are.
* **`steps.py`**: This holds all the actual OpenCV functions, like `denoise`, `deskew` (straightening), and `autocrop`.
* **`geometry.py` & `rotate.py`**: These handle fixing the physical angle of the paper and turning upside-down images right-side up.
* **`code_detector.py`**: Finds and reads any QR codes or barcodes on the page.

### How the pipeline works:
1. You pass an image to the `Preprocessing` class.
2. It extracts the visual features and decides on a label (`CLEAN`, `HEAVY`, or `SKIP`).
3. Based on that label, it runs a specific "recipe" of steps. For example, a `HEAVY` image gets denoised, thresholded, and sharpened, while a `CLEAN` image just gets straightened and cropped.
4. It reads any QR codes at the very end.
5. It returns a result object containing the fixed image and a summary of what it did.