import os
import random
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# Import functions from your training script
from aksara_train_v3 import (
    otsu_threshold, apply_threshold, closing, 
    center_character, apply_thinning, apply_skeletonize
)

def run_random_test(model_path, data_path='v3/val', mode='thinning'):
    """
    mode: 'thinning', 'skeleton', or 'none'
    """
    # 1. Load the model
    print(f"Loading model from {model_path}...")
    model_data = joblib.load(model_path)
    rf_model = model_data['model']
    classes = model_data['classes']
    img_size = model_data['img_size']

    # 2. Pick a random image from the dataset
    all_classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    random_class = random.choice(all_classes)
    class_dir = os.path.join(data_path, random_class)
    random_file = random.choice(os.listdir(class_dir))
    img_path = os.path.join(class_dir, random_file)

    print(f"Selected: {random_class}/{random_file}")

    # 3. Preprocessing Pipeline
    img = Image.open(img_path).convert('L')
    gray = np.array(img)

    t = otsu_threshold(gray)
    binary = apply_threshold(gray, t)
    cleaned = closing(binary, kernel_size=3)
    
    # Apply Thinning or Skeletonization
    processed = cleaned.copy()
    title_processed = "Closing (Cleaned)"
    
    if mode == 'thinning':
        processed = apply_thinning(cleaned)
        title_processed = "Thinning"
    elif mode == 'skeleton':
        processed = apply_skeletonize(cleaned)
        title_processed = "Skeletonization"

    # Centering
    centered = center_character(processed, target_size=img_size)

    # 4. Prediction
    feat = centered.flatten().reshape(1, -1)
    pred_idx = rf_model.predict(feat)[0]
    probs = rf_model.predict_proba(feat)[0]
    conf = probs[pred_idx] * 100

    # 5. Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 5, 1)
    plt.imshow(gray, cmap='gray')
    plt.title(f"Original\n(True: {random_class})")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(binary, cmap='gray')
    plt.title("Otsu Threshold")
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(cleaned, cmap='gray')
    plt.title("Cleaned")
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(processed, cmap='gray')
    plt.title(title_processed)
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(centered, cmap='gray')
    color = 'green' if classes[pred_idx] == random_class else 'red'
    plt.title(f"Prediction: {classes[pred_idx]}\nConf: {conf:.1f}%", color=color)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- RUN THE TEST ---
# Use the model file you just saved (e.g., the 100 tree version or the main one)
MODEL_FILE = 'aksara_jawa_v3_model_skeleton.joblib' 

# Set mode to 'thinning', 'skeleton', or 'none'
# Make sure this matches how the model was trained!
run_random_test(MODEL_FILE, mode='skeleton') 
