import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Sabitler ---
INPUT_DIR = 'PCB_DATASET'
IMG_DIR = os.path.join(INPUT_DIR, 'images')
TEMPLATE_DIR = os.path.join(INPUT_DIR, 'PCB_USED')
OUTPUT_FILENAME = 'ozel_short_tespit_akisi.jpg' 

# Belirtilen dosyaların yollarını doğrudan tanımlayan fonksiyon
def get_specific_paths():
    
    # Hatalı görsel yolu: 01_short_06.jpg, Short klasöründe
    TEST_FILE_NAME = '01_short_06.jpg'
    TEST_TYPE_DIR = os.path.join(IMG_DIR, 'Short')
    test_path = os.path.join(TEST_TYPE_DIR, TEST_FILE_NAME)
    
    # Referans görsel yolu: 01.jpg, PCB_USED klasöründe
    TEMPLATE_FILE_NAME = '01.jpg'
    template_path = os.path.join(TEMPLATE_DIR, TEMPLATE_FILE_NAME)
    
    return test_path, template_path

# --- Template Matching ve Hata İzolasyonu Fonksiyonu ---
def perform_template_matching_for_specific(test_path, template_path):
    
    # 1. Görüntüleri Yükleme (Gri Tonlama)
    test_img_gray = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    template_img_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if test_img_gray is None or template_img_gray is None:
        raise ValueError("Belirtilen görüntülerden biri veya ikisi yüklenemedi. Dosya yollarını kontrol edin.")

    # Boyut Eşitleme (Şablon boyutu test görüntüsüne göre ayarlanır)
    template_img_gray = cv2.resize(template_img_gray, (test_img_gray.shape[1], test_img_gray.shape[0]))
    
    # 2. Ön İşleme (Gauss Bulanıklığı Uygulama)
    test_blurred = cv2.GaussianBlur(test_img_gray, (5, 5), 0)
    template_blurred = cv2.GaussianBlur(template_img_gray, (5, 5), 0)

    # 3. Fark Alma (Template Matching Mantığı)
    difference = cv2.absdiff(test_blurred, template_blurred)
    
    # 4. Hata İzolasyonu (Eşikleme)
    _, isolated_error = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
    
    # Görselleştirme için orijinal renkli görüntüyü döndür
    original_test_rgb = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)
    
    return original_test_rgb, template_img_gray, test_blurred, difference, isolated_error

# --- Ana Çalışma Bloğu ---
try:
    test_path, template_path = get_specific_paths()
    
    if not os.path.exists(test_path):
        print(f"HATA: Hatalı görsel bulunamadı: {test_path}")
        exit()
    if not os.path.exists(template_path):
        print(f"HATA: Referans görsel bulunamadı: {template_path}")
        exit()
        
    original_test, template_gray, test_blurred, difference, isolated_error = perform_template_matching_for_specific(test_path, template_path)
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    plots = [
        (original_test, '1. Girdi: Hatalı Short Kartı (01_short_06.jpg)', None),
        (template_gray, '2. Referans: Altın Standart Şablon (01.jpg)', 'gray'),
        (test_blurred, '3. Ön İşleme: Gauss Blur Uygulandı', 'gray'),
        (difference, '4. Fark Alma (Short Hata Tespit Edildi)', 'gray'),
        (isolated_error, '5. Nihai Çıktı: İzole Edilmiş Short Hatası', 'gray')
    ]
    
    for ax, (img, title, cmap) in zip(axes.flatten(), plots):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=150)
    print(f"\n✅ Özel Short Hata Akışı Görseli Başarıyla Kaydedildi: {OUTPUT_FILENAME}")
    print("Bu görsel, sunumunuzdaki 'Template Matching Mantığı' slaytında kullanıma hazırdır.")

except Exception as e:
    print(f"Bir hata oluştu: {e}")