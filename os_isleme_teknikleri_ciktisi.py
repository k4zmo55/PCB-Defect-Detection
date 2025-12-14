import cv2
import matplotlib.pyplot as plt
import os
import random
import numpy as np

# --- Sabitler (Kendi Projenizin Klasör Yapısına Göre Ayarlayın) ---
INPUT_DIR = 'PCB_DATASET'
IMG_DIR = os.path.join(INPUT_DIR, 'images')

# Rastgele bir örnek bulmak için tüm görüntü yollarını toplayın
all_image_paths = []
for defect_type in os.listdir(IMG_DIR):
    type_dir = os.path.join(IMG_DIR, defect_type)
    for filename in os.listdir(type_dir):
        if filename.lower().endswith('.jpg'):
            all_image_paths.append(os.path.join(type_dir, filename))

if not all_image_paths:
    print("HATA: Dataset klasöründe JPG dosyası bulunamadı. Lütfen yolu kontrol edin.")
    exit()

# Rastgele bir görüntü yolu seçin
example_path = random.choice(all_image_paths)

# --- Görüntü İşleme Fonksiyonları ---

def apply_clahe(image):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) uygular."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Gaussian Bulanıklığı (Gürültü Kontrolü) uygular."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_adaptive_thresholding(image):
    """Adaptif Eşikleme (Basit Segmentasyon) uygular."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """Canny Kenar Tespiti uygular."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low_threshold, high_threshold)

# --- Ana Çalışma Bloğu ---
try:
    original_bgr = cv2.imread(example_path)
    if original_bgr is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {example_path}")
    
    # Gerekli ön işlemeleri uygula
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    
    # CLAHE, Gauss Blur, Adaptive Threshold ve Canny çıktıları
    clahe_img = apply_clahe(original_bgr)
    blurred_img = apply_gaussian_blur(original_bgr)
    thresholded_img = apply_adaptive_thresholding(original_bgr)
    canny_img = apply_canny_edge_detection(original_bgr)
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    plots = [
        (original_rgb, 'Orijinal Görüntü', None),
        (gray, 'Gri Tonlama', 'gray'),
        (clahe_img, '1. CLAHE (Kontrast Optimizasyonu)', 'gray'),
        (blurred_img, '2. Gauss Bulanıklığı (Gürültü Kontrolü)', 'gray'),
        (thresholded_img, '3. Adaptif Eşikleme (Segmentasyon)', 'gray'),
        (canny_img, '4. Canny Kenar Tespiti', 'gray')
    ]
    
    for ax, (img, title, cmap) in zip(axes.flatten(), plots):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    output_filename = 'sunum_on_isleme_adimlari.jpg'
    plt.savefig(output_filename, dpi=150)
    print(f"\n✅ Ön İşleme Adımları Görseli Başarıyla Kaydedildi: {output_filename}")

except Exception as e:
    print(f"Bir hata oluştu: {e}")