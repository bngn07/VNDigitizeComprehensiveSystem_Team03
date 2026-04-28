import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import csv

os.makedirs("output/dataset/images", exist_ok=True)
IMG_W, IMG_H = 500,500
NUM_SAMPLES = 500

random.seed(42)
np.random.seed(42)

# --- Load text from file ---
TEXT_FILE_PATH = "data/input/text/lorem.txt"
text_words = []

try:
    with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
        # Read the file and split it into a list of words
        text_words = f.read().split()
except FileNotFoundError:
    print(f"Warning: '{TEXT_FILE_PATH}' not found. Falling back to random characters.")

def add_gaussian_noise(img_array, intensity=0):
    noise = np.random.normal(0, intensity, img_array.shape)
    return np.clip(img_array + noise, 0, 255).astype(np.uint8)

def add_salt_pepper_dots(draw, n_dots, canvas_w, canvas_h):
    for _ in range(n_dots):
        x = random.randint(0, canvas_w - 1)
        y = random.randint(0, canvas_h - 1)
        r = random.randint(1, 3)
        color = random.choice([0, 50, 80, 100, 150])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

def add_text(draw, canvas_w, canvas_h, n_lines):
    chars = string.ascii_letters + string.digits + " .,!?-"
    
    for _ in range(n_lines):
        # If we successfully loaded words from lorem.txt, sample from them
        if text_words:
            num_words = random.randint(2, 6) # Pick 2 to 6 words per line
            if len(text_words) > num_words:
                start_idx = random.randint(0, len(text_words) - num_words)
                text = " ".join(text_words[start_idx : start_idx + num_words])
            else:
                text = " ".join(text_words)
        else:
            # Fallback to random string generation
            line_len = random.randint(10, 40)
            text = "".join(random.choices(chars, k=line_len))
            
        x = random.randint(5, canvas_w // 4)
        y = random.randint(5, canvas_h - 15)
        font_size = random.randint(8, 16)
        gray_val = random.randint(0, 80)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
            
        draw.text((x, y), text, fill=gray_val, font=font)

def generate_sample(idx):
    # Decide content type
    content_type = random.choice(["blank", "noise_only", "dots_only", "text_only", "mixed"])
    bg_val = random.randint(240, 255)
    img_array = np.full((IMG_H, IMG_W), bg_val, dtype=np.uint8)
    
    # Decide if blank
    if content_type == "blank":
        # Possibly add very faint gaussian noise (imperceptible)
        noise_intensity = random.choice([0, 0, 0, 1.5])
        img_array = add_gaussian_noise(img_array, noise_intensity)
        label = 1  # is_blank
    else:
        label = 0  # not_blank
        img = Image.fromarray(img_array, mode="L")
        draw = ImageDraw.Draw(img)
        
        if content_type == "noise_only":
            img_array = add_gaussian_noise(img_array, random.uniform(15, 40))
            img = Image.fromarray(img_array, mode="L")
            
        elif content_type == "dots_only":
            n_dots = random.randint(20, 200)
            add_salt_pepper_dots(draw, n_dots, IMG_W, IMG_H)
            img_array = np.array(img)
            
        elif content_type == "text_only":
            n_lines = random.randint(3, 15)
            add_text(draw, IMG_W, IMG_H, n_lines)
            img_array = np.array(img)
            
        elif content_type == "mixed":
            # Any combo of text, dots, noise
            has_text = random.random() > 0.3
            has_dots = random.random() > 0.3
            has_noise = random.random() > 0.3
            
            if has_noise:
                img_array = add_gaussian_noise(img_array, random.uniform(5, 25))
                img = Image.fromarray(img_array, mode="L")
                draw = ImageDraw.Draw(img)
            if has_dots:
                add_salt_pepper_dots(draw, random.randint(10, 150), IMG_W, IMG_H)
            if has_text:
                add_text(draw, IMG_W, IMG_H, random.randint(2, 12))
            if not (has_text or has_dots or has_noise):
                add_text(draw, IMG_W, IMG_H, 5)
                
            img_array = np.array(img)
            
        img = Image.fromarray(img_array, mode="L")
        
    # Final image save
    filename = f"img_{idx:04d}.png"
    path = f"output/dataset/images/{filename}"
    Image.fromarray(img_array, mode="L").save(path)
    return filename, label, content_type

# Generate dataset
records = []
for i in range(NUM_SAMPLES):
    fname, lbl, ctype = generate_sample(i)
    records.append({"filename": fname, "is_blank": lbl, "content_type": ctype})

# Save CSV
csv_path = "output/dataset/labels.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "is_blank", "content_type"])
    writer.writeheader()
    writer.writerows(records)

# Stats
from collections import Counter
label_counts = Counter(r["is_blank"] for r in records)
type_counts = Counter(r["content_type"] for r in records)

print(f"Total samples: {len(records)}")
print(f"Blank (1): {label_counts[1]}, Not-blank (0): {label_counts[0]}")
print(f"Content types: {dict(type_counts)}")