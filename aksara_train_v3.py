import os
import numpy as np
import time
import joblib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1. VERBOSE OTSU THRESHOLDING (Hand-written with NumPy)
# ──────────────────────────────────────────────────────────────────────────────

def otsu_threshold_verbose(gray: np.ndarray) -> int:
    """
    Finds the optimal threshold value (0-255) to binarize an image.
    Maximizes the 'inter-class variance' between background and foreground.
    """
    # Create a histogram of pixel intensities
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total_pixels = gray.size
    
    best_threshold = 0
    max_variance = 0
    
    # Check every possible threshold from 0 to 255
    for t in range(256):
        # Weight (W): fraction of pixels in this group
        weight_bg = np.sum(hist[:t]) / total_pixels
        weight_fg = np.sum(hist[t:]) / total_pixels
        
        # Skip if one group is empty
        if weight_bg == 0 or weight_fg == 0:
            continue
            
        # Mean (M): average intensity of pixels in this group
        mean_bg = np.sum(np.arange(t) * hist[:t]) / (np.sum(hist[:t]) + 1e-10)
        mean_fg = np.sum(np.arange(t, 256) * hist[t:]) / (np.sum(hist[t:]) + 1e-10)
        
        # Variance between classes
        # Formula: W_bg * W_fg * (M_bg - M_fg)^2
        variance = weight_bg * weight_fg * (mean_bg - mean_fg)**2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
            
    return best_threshold

def apply_threshold(gray: np.ndarray, t: int) -> np.ndarray:
    """Binarize image: values <= t become 255 (White ink), others 0 (Black background)."""
    # Note: We assume dark ink on light paper, so lower values are the ink.
    return np.where(gray <= t, 255, 0).astype(np.uint8)

# ──────────────────────────────────────────────────────────────────────────────
# 2. VERBOSE MORPHOLOGY (Loop-based for easy explanation)
# ──────────────────────────────────────────────────────────────────────────────

def erode_verbose(image: np.ndarray, kernel_size: int = 1) -> np.ndarray:
    H, W = image.shape
    pad = kernel_size // 2
    output = np.zeros_like(image)
    padded = np.pad(image, pad_width=pad, mode="constant", constant_values=0)

    for i in range(H):
        for j in range(W):
            current_min = 255
            for ri in range(kernel_size):
                for ci in range(kernel_size):
                    val = padded[i + ri, j + ci]
                    if val < current_min:
                        current_min = val
            output[i, j] = current_min
    return output

def dilate_verbose(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    H, W = image.shape
    pad = kernel_size // 2
    output = np.zeros_like(image)
    padded = np.pad(image, pad_width=pad, mode="constant", constant_values=0)

    for i in range(H):
        for j in range(W):
            current_max = 0
            for ri in range(kernel_size):
                for ci in range(kernel_size):
                    val = padded[i + ri, j + ci]
                    if val > current_max:
                        current_max = val
            output[i, j] = current_max
    return output

def closing_verbose(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return erode_verbose(dilate_verbose(image, kernel_size), kernel_size)

# ──────────────────────────────────────────────────────────────────────────────
# 3. VERBOSE CHARACTER CENTERING (Hand-written)
# ──────────────────────────────────────────────────────────────────────────────

def center_character(binary: np.ndarray, target_size=(64, 64)) -> np.ndarray:
    """
    Finds the ink (white pixels), crops it, and centers it.
    This prevents the model from getting confused by position differences.
    """
    # 1. Find all coordinates where there is ink (value 255)
    coords = np.argwhere(binary == 255)
    if coords.size == 0:
        return np.zeros(target_size, dtype=np.uint8)
    
    # 2. Get the bounding box of the ink
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    crop = binary[y_min:y_max+1, x_min:x_max+1]
    
    # 3. Create a blank canvas of target_size
    new_img = np.zeros(target_size, dtype=np.uint8)
    
    # 4. Calculate where to place the crop so it is centered
    ch, cw = crop.shape
    th, tw = target_size
    
    # Simple resize if crop is bigger than target
    if ch > th or cw > tw:
        # Using PIL just for the resize part
        pil_crop = Image.fromarray(crop).resize((tw-4, th-4), Image.Resampling.LANCZOS)
        crop = np.array(pil_crop)
        ch, cw = crop.shape

    # Find starting coordinates for centering
    y_off = (th - ch) // 2
    x_off = (tw - cw) // 2
    
    # Paste the crop into the center
    new_img[y_off:y_off+ch, x_off:x_off+cw] = crop
    
    return new_img

# ──────────────────────────────────────────────────────────────────────────────
# 4. DATASET LOADING (Aksara Jawa v3)
# ──────────────────────────────────────────────────────────────────────────────

def load_aksara_v3(base_path, target_size=(64, 64), max_per_class=1000):
    X = []
    y = []
    classes = sorted(os.listdir(base_path))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Loading Aksara Jawa from {base_path}...")
    total_files = sum([len(os.listdir(os.path.join(base_path, c))) for c in classes])
    processed_count = 0
    
    for cls in classes:
        cls_dir = os.path.join(base_path, cls)
        # Load ALL files to get all augmented variants (.b, .r5, .rm5, etc.)
        files = os.listdir(cls_dir)
        count = 0
        for f in files:
            img_path = os.path.join(cls_dir, f)
            try:
                # 1. Load and Grayscale
                img = Image.open(img_path).convert('L')
                gray = np.array(img)
                
                # 2. Verbose Otsu
                t = otsu_threshold_verbose(gray)
                binary = apply_threshold(gray, t)
                
                # 3. Verbose Morphology
                cleaned = closing_verbose(binary, kernel_size=3)
                
                # 4. CENTER the character (New Step!)
                centered = center_character(cleaned, target_size=target_size)
                
                # 5. Store
                X.append(centered.flatten())
                y.append(class_to_idx[cls])
                count += 1
                processed_count += 1
                
                if processed_count % 50 == 0:
                    print(f"  Processed {processed_count} images total...")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        print(f"  Class '{cls}': loaded {count} images")
        
    return np.array(X), np.array(y), classes

# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    train_path = "v3/train"
    val_path = "v3/val"
    
    # 1. Load Data - Now using centering and larger images
    MAX_PER_CLASS = 1000 
    IMG_SIZE = (64, 64) 
    
    X_train, y_train, classes = load_aksara_v3(train_path, target_size=IMG_SIZE, max_per_class=MAX_PER_CLASS)
    X_val, y_val, _ = load_aksara_v3(val_path, target_size=IMG_SIZE, max_per_class=MAX_PER_CLASS)
    
    print(f"\nTotal Training Samples: {len(X_train)}")
    print(f"Total Validation Samples: {len(X_val)}")

    # 2. Train Random Forest
    # Increasing trees to 500 for better stability and accuracy
    print(f"\nTraining Random Forest (500 trees) on {len(X_train)} samples...")
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = rf.predict(X_val)
    print("\nAccuracy Score:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=classes))

    # 4. Save
    model_data = {'model': rf, 'classes': classes, 'img_size': IMG_SIZE}
    joblib.dump(model_data, 'aksara_jawa_v3_model.joblib')
    print("\nModel saved to 'aksara_jawa_v3_model.joblib'")

if __name__ == "__main__":
    main()
