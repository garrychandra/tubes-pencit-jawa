import os
import numpy as np
import time
import joblib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize as sk_skeletonize, thin as sk_thin


# ──────────────────────────────────────────────────────────────────────────────
# 1. OTSU THRESHOLDING
# ──────────────────────────────────────────────────────────────────────────────

def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total_pixels = gray.size
    
    best_threshold = 0
    max_variance = 0
    
    # cek semua threshold dari 0 - 255
    for t in range(256):
        # hitung weight
        weight_bg = np.sum(hist[:t]) / total_pixels
        weight_fg = np.sum(hist[t:]) / total_pixels
        
        # skip kalau kosong
        if weight_bg == 0 or weight_fg == 0:
            continue
            
        # hitung mean 
        mean_bg = np.sum(np.arange(t) * hist[:t]) / (np.sum(hist[:t]) + 1e-10)
        mean_fg = np.sum(np.arange(t, 256) * hist[t:]) / (np.sum(hist[t:]) + 1e-10)
        
        # hitung variance between class
        # rumus: W_bg * W_fg * (M_bg - M_fg)^2
        variance = weight_bg * weight_fg * (mean_bg - mean_fg)**2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
            
    return best_threshold

def apply_threshold(gray: np.ndarray, t: int) -> np.ndarray:
    # terapkan thresholding, ubah pixel jadi putih kalo nilainya <= threshold
    return np.where(gray <= t, 255, 0).astype(np.uint8)

# ──────────────────────────────────────────────────────────────────────────────
# 2. MORPHOLOGY
# ──────────────────────────────────────────────────────────────────────────────

def erode(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Optimized erosion using NumPy slicing."""
    H, W = image.shape
    pad = kernel_size // 2
    # Pad with 255 (white) because erosion takes the minimum
    padded = np.pad(image, pad, mode="constant", constant_values=255)
    output = np.full_like(image, 255)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            output = np.minimum(output, padded[i:i+H, j:j+W])
    return output

def dilate(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Optimized dilation using NumPy slicing."""
    H, W = image.shape
    pad = kernel_size // 2
    # Pad with 0 (black) because dilation takes the maximum
    padded = np.pad(image, pad, mode="constant", constant_values=0)
    output = np.zeros_like(image)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            output = np.maximum(output, padded[i:i+H, j:j+W])
    return output

def apply_thinning(binary: np.ndarray) -> np.ndarray:
    """Applies Zhang-Suen thinning via skimage."""
    # skimage works best with boolean arrays
    bool_img = binary > 0
    thinned = sk_thin(bool_img)
    return (thinned * 255).astype(np.uint8)

def apply_skeletonize(binary: np.ndarray) -> np.ndarray:
    """Applies skeletonization via skimage."""
    bool_img = binary > 0
    skeleton = sk_skeletonize(bool_img)
    return (skeleton * 255).astype(np.uint8)

 #dilasi -> erosi
def closing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return erode(dilate(image, kernel_size), kernel_size)

# ──────────────────────────────────────────────────────────────────────────────
# 3. CHARACTER CENTERING
# ──────────────────────────────────────────────────────────────────────────────

def center_character(binary: np.ndarray, target_size=(64, 64)) -> np.ndarray:

    # cari semua koordinat pixel putih (foreground) (asumsi setelah di thresholding)
    coords = np.argwhere(binary == 255)
    if coords.size == 0:
        return np.zeros(target_size, dtype=np.uint8)
    
    # buat bounding box disekitar foreground
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    crop = binary[y_min:y_max+1, x_min:x_max+1]
    
    #buat canvas dengan ukuran target
    new_img = np.zeros(target_size, dtype=np.uint8)
    
    # hitung offset untuk menempatkan crop di tengah canvas
    ch, cw = crop.shape
    th, tw = target_size
    
    # resize jika crop lebih besar dari target
    if ch > th or cw > tw:
        pil_crop = Image.fromarray(crop).resize((tw-4, th-4), Image.Resampling.LANCZOS)
        crop = np.array(pil_crop)
        ch, cw = crop.shape

    # hitung offset untuk menempatkan crop di tengah canvas
    y_off = (th - ch) // 2
    x_off = (tw - cw) // 2
    
    # tempel crop ke canvas baru
    new_img[y_off:y_off+ch, x_off:x_off+cw] = crop
    
    return new_img

# ──────────────────────────────────────────────────────────────────────────────
# 4. DATASET LOADING (Aksara Jawa v3)
# ──────────────────────────────────────────────────────────────────────────────

def load_aksara_v3(base_path, target_size=(64, 64), max_per_class=1000, use_thinning=False, use_skeleton=False):
    X = []
    y = []
    
    # 1. Tentukan folder tujuan preprocess
    suffix = f"_{target_size[0]}x{target_size[1]}"
    if use_thinning: suffix += "_thin"
    if use_skeleton: suffix += "_skel"
    
    # Folder hasil preprocess akan disimpan di folder yang sama dengan v3 tapi ada suffix _processed
    processed_base = base_path.rstrip("/") + "_processed" + suffix
    
    classes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Cek apakah folder processed sudah ada
    if os.path.exists(processed_base):
        print(f"Loading pre-processed images from {processed_base}...")
        for cls in classes:
            cls_dir = os.path.join(processed_base, cls)
            if not os.path.exists(cls_dir): continue
            
            files = [f for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for f in files[:max_per_class]:
                img_path = os.path.join(cls_dir, f)
                img = Image.open(img_path).convert('L')
                X.append(np.array(img).flatten())
                y.append(class_to_idx[cls])
        return np.array(X), np.array(y), classes

    # Jika belum ada, lakukan preprocessing
    print(f"Pre-processing and saving Aksara Jawa to {processed_base}...")
    os.makedirs(processed_base, exist_ok=True)
    
    processed_count = 0
    for cls in classes:
        cls_dir = os.path.join(base_path, cls)
        save_dir = os.path.join(processed_base, cls)
        os.makedirs(save_dir, exist_ok=True)
        
        files = [f for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        count = 0
        for f in files:
            if count >= max_per_class: break
            
            img_path = os.path.join(cls_dir, f)
            try:
                img = Image.open(img_path).convert('L')
                gray = np.array(img)
                
                # Thresholding
                t = otsu_threshold(gray)
                binary = apply_threshold(gray, t)
                
                # Morphologi
                cleaned = closing(binary, kernel_size=3)
                
                # Tambahan: Thinning atau Skeletonize
                if use_thinning:
                    cleaned = apply_thinning(cleaned)
                elif use_skeleton:
                    cleaned = apply_skeletonize(cleaned)
                
                # Centering
                centered = center_character(cleaned, target_size=target_size)
                
                # Simpan ke folder preprocess agar tidak perlu ulang
                save_path = os.path.join(save_dir, f)
                Image.fromarray(centered).save(save_path)
                
                X.append(centered.flatten())
                y.append(class_to_idx[cls])
                count += 1
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} images total...")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        print(f"  Class '{cls}': processed {count} images")
        
    return np.array(X), np.array(y), classes

# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    train_path = "v3/train"
    val_path = "v3/val"
    
    # load data
    MAX_PER_CLASS = 1000 
    IMG_SIZE = (64, 64) 
    USE_THINNING = False  # Ubah jadi True jika ingin pakai thinning
    USE_SKELETON = False # Ubah jadi True jika ingin pakai skeleton
    
    X_train, y_train, classes = load_aksara_v3(train_path, target_size=IMG_SIZE, 
                                             max_per_class=MAX_PER_CLASS,
                                             use_thinning=USE_THINNING, 
                                             use_skeleton=USE_SKELETON)
    X_val, y_val, _ = load_aksara_v3(val_path, target_size=IMG_SIZE, 
                                   max_per_class=MAX_PER_CLASS,
                                   use_thinning=USE_THINNING, 
                                   use_skeleton=USE_SKELETON)
    
    print(f"\nTotal Training Samples: {len(X_train)}")
    print(f"Total Validation Samples: {len(X_val)}")

    # random forest
    print(f"\nTraining Random Forest (500 trees) on {len(X_train)} samples...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # evaluasi
    y_pred = rf.predict(X_val)
    print("\nAccuracy Score:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=classes))

    # simpan model
    model_data = {'model': rf, 'classes': classes, 'img_size': IMG_SIZE}
    joblib.dump(model_data, 'aksara_jawa_v3_model_100.joblib')
    print("\nModel saved to 'aksara_jawa_v3_model.joblib'")

if __name__ == "__main__":
    main()
