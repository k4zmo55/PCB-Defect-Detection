import cv2
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from matplotlib import patches
import xml.etree.ElementTree as ET

# --- Sabitler (Kendi Projenizin Klasör Yapısına Göre Ayarlayın) ---
INPUT_DIR = 'PCB_DATASET'
IMG_DIR = os.path.join(INPUT_DIR, 'images')
ANN_DIR = os.path.join(INPUT_DIR, 'Annotations')

# Hata Tipleri (Klasör İsimleri)
DEFECT_TYPES = [
    'Missing_hole', 
    'Mouse_bite', 
    'Open_circuit', 
    'Short', 
    'Spur', 
    'Spurious_copper'
]

# --- XML Ayrıştırma Fonksiyonu ---
def parse_xml_for_boxes(xml_file):
    """Belirtilen XML dosyasından hata sınır kutularını (bndbox) ayrıştırır."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            data.append({'class': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
        return data
    except Exception as e:
        # XML ayrıştırma hatası olduğunda boş liste döndür
        return []

# --- Ana Görsel Oluşturma Fonksiyonu ---
def generate_single_defect_images(defect_types):
    """
    Her hata tipinden rastgele bir örnek seçer, sınır kutusu çizer ve
    o hata tipinin adıyla ayrı bir dosya olarak kaydeder.
    """
    
    output_dir = 'sunum_tekil_hata_gorselleri'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Oluşturulacak görseller '{output_dir}' klasörüne kaydedilecektir.")

    for i, defect_type in enumerate(defect_types):
        
        # 1. Görüntü ve XML Dosyalarını Bulma
        img_type_dir = os.path.join(IMG_DIR, defect_type)
        ann_type_dir = os.path.join(ANN_DIR, defect_type)
        
        image_files = [f for f in os.listdir(img_type_dir) if f.lower().endswith('.jpg')]
        
        if not image_files:
            print(f"HATA: {defect_type} klasöründe resim bulunamadı.")
            continue

        # Rastgele bir resim seç
        selected_file = random.choice(image_files)
        img_path = os.path.join(img_type_dir, selected_file)
        
        # Karşılık gelen XML dosya adı
        xml_file_name = selected_file.replace('.jpg', '.xml')
        xml_path = os.path.join(ann_type_dir, xml_file_name)
        
        # 2. Görüntüyü Yükleme ve XML'i Ayrıştırma
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"HATA: {img_path} yüklenemedi.")
            continue
            
        # OpenCV BGR'den Matplotlib RGB'ye dönüştürme
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotations = parse_xml_for_boxes(xml_path)
        
        # 3. Görselleştirme (Tekil Figür)
        plt.figure(figsize=(12, 6)) # Sunum için daha geniş boyut
        ax = plt.gca()
        ax.imshow(img)
        ax.set_title(f'Hata Tipi: {defect_type.replace("_", " ").upper()} - (Rastgele Örnek)', 
                     fontsize=16, fontweight='bold')
        
        # Sınır kutularını çiz
        for ann in annotations:
            xmin = ann['xmin']
            ymin = ann['ymin']
            xmax = ann['xmax']
            ymax = ann['ymax']
            width = xmax - xmin
            height = ymax - ymin
            
            # Kutu ve Etiket (Sunum için belirginleştirildi)
            rect = patches.Rectangle((xmin, ymin), width, height, 
                                     linewidth=4, # Kalın çizgi
                                     edgecolor='red', 
                                     facecolor='none')
            ax.add_patch(rect)
            
            # Hata ismini kutunun üzerine yaz
            ax.text(xmin, ymin - 30, ann['class'].replace("_", " "), 
                    color='white', fontsize=14, fontweight='bold', 
                    bbox=dict(facecolor='red', alpha=0.8, pad=3))

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        
        # Görseli kaydet
        output_filename = os.path.join(output_dir, f'{defect_type}.jpg')
        plt.savefig(output_filename, dpi=200) # Yüksek DPI ile net çıktı
        plt.close() # Belleği temizle
        
        print(f"✅ Görsel Kaydedildi: {output_filename}")
    
# --- Fonksiyonu Çalıştır ---
# Not: Bu kodun çalışması için PCB_DATASET klasör yapınızın doğru olması gerekir.
generate_single_defect_images(DEFECT_TYPES)