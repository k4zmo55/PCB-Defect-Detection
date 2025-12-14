# Importing the standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import os
import random
import re
import shutil
# --- Yeni Ekleme: OpenCV kütüphanesini içe aktarın ---
import cv2
# --------------------------------------------------------
sns.set_style('darkgrid')
sns.set_palette('pastel')
import xml.etree.ElementTree as ET

import warnings
warnings.filterwarnings('ignore')


input_dir='PCB_DATASET'
os.listdir(input_dir)

template_dir=os.path.join(input_dir,'PCB_USED')
print(template_dir)

def visualize_img(dir_name,nos_):
    """
    Belirtilen dizindeki görüntüleri görselleştirir.
    """
    k=1
    plt.figure(figsize=(8,(nos_//2)*6))
    for filename in os.listdir(dir_name)[0:nos_]:
        if filename.lower().endswith(('.jpg','.jpeg','.png')):
            ax=plt.subplot((nos_//2)+1,2,k)
            img_path=os.path.join(dir_name,filename)
            img=plt.imread(img_path)
            ax.imshow(img)
            ax.set_xlabel(filename)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            k+=1
    plt.tight_layout()
    plt.show()

visualize_img(template_dir,nos_=4) # Yorum satırından çıkarıldı

print(f'No of template images:{len(os.listdir(template_dir))}')


img_dir=os.path.join(input_dir,'images')

#Listing the types of defects
os.listdir(img_dir)
types_defect=os.listdir(os.path.join(input_dir,'images'))
print(types_defect)


#Creating an image path list for ready refernce
img_path_list=[]
#Creating img_path list
for sub_cat in types_defect:
    for file in os.listdir(os.path.join(img_dir,sub_cat)):
    
        img_path_list.append(os.path.join(img_dir,sub_cat,file))
        

#Vizualizing defect images for each type
df_defect=pd.DataFrame(columns=['No of defect']) #dataframe for counting no of defects
for sub_cat in types_defect:
    visualize_img(os.path.join(img_dir,sub_cat),nos_=2)#Visualizing 2 types of defect for each type # Yorum satırından çıkarıldı

    print(f'No of {sub_cat} images:{len(os.listdir(os.path.join(img_dir,sub_cat)))}')
    
    df_defect.loc[sub_cat]=len(os.listdir(os.path.join(img_dir,sub_cat))) 


print(df_defect)

rotated_dir=os.path.join(input_dir,'rotation')
os.listdir(rotated_dir)

rotated_angle_list=[j for j in os.listdir(rotated_dir) if j.endswith('.txt')]
print(rotated_angle_list)

types_defect_rotated=[j for j in os.listdir(rotated_dir) if j.endswith('.txt')==False]
print(types_defect_rotated)

#Vizualizing rotated defects
df_defect_rotated=pd.DataFrame(columns=['No of defect'])
for sub_cat in types_defect_rotated:
    visualize_img(os.path.join(rotated_dir,sub_cat),nos_=2) # Yorum satırından çıkarıldı

    print(f'No of {sub_cat} images:{len(os.listdir(os.path.join(rotated_dir,sub_cat)))}')
    
    df_defect_rotated.loc[sub_cat]=len(os.listdir(os.path.join(rotated_dir,sub_cat)))

print(df_defect_rotated)


#Reading the rotation text files

df_rotation_angle=pd.DataFrame(columns=['Line','Angle'])
for filename in rotated_angle_list:
    with open(os.path.join(rotated_dir,filename),'r') as f:
        lines=f.readlines()
        for line in lines:
            text,angle=line.split()
            df_rotation_angle=pd.concat([df_rotation_angle,pd.DataFrame({'Line':[text],'Angle':[angle]})],axis=0)

print(df_rotation_angle)

annote_dir=os.path.join(input_dir,'Annotations')
annote_dir

type_annot=os.listdir(annote_dir)
type_annot

df_annot_nos=pd.DataFrame(columns=['No of annotations'])
#Checking the length of annotation itms
for i in type_annot:
    df_annot_nos.loc[i]=len(os.listdir(os.path.join(annote_dir,i)))
print(df_annot_nos)

#Checking the type of files
file_list=os.listdir(os.path.join(annote_dir,'Mouse_bite'))
file_list[0:5]


tree = ET.parse(os.path.join(os.path.join(annote_dir,'Mouse_bite'),'01_mouse_bite_11.xml'))
root = tree.getroot()

#getting the structure of XML file
print(ET.tostring(root, encoding='utf8').decode('utf8'))

#Parsing XML to return Bounding box dimensions 
def parse_xml(xml_file):
    
    data=[]
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        
        data.append({
            'filename': filename,
            'width': width,
            'height': height,
            'class': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
        
    return data

#Retrieving data for all files
data=[]
all_data=[]

for x in type_annot:
    for file in os.listdir(os.path.join(annote_dir,x)):
        xml_file_path=os.path.join(os.path.join(annote_dir,x),file)
        data=parse_xml(xml_file_path)
        all_data.extend(data)

#Creating a dataframe to store the annotations
df_annot=pd.DataFrame(all_data)
print(df_annot.head())

# =========================================================================
# --- GÖRÜNTÜ İŞLEME FONKSİYONLARI BAŞLANGICI ---
# =========================================================================

# --- YENİ FONKSİYON 1: Renk Alanı Dönüşümü (HSV) ---
def convert_to_hsv(image):
    """
    RGB uzayından HSV uzayına dönüşüm yapar.
    Hata ve zemin arasındaki kontrastı maksimize etmek için kullanılır.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# --- YENİ FONKSİYON 2: Arka Plan Çıkarma (Template Matching / Geleneksel Yöntem) ---
def apply_template_matching(test_image_path, template_image_path):
    """
    Test edilen PCB görüntüsü ile kusursuz (altın standart) PCB görüntüsü 
    arasındaki mutlak farkı hesaplar.
    """
    test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if template_image_path is None:
        return None, None
        
    template_img = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

    if test_img is None or template_img is None:
        print("HATA: Test veya Template görüntüsü okunamadı.")
        return None, None

    # Boyutları eşitle (Template matching'in ön koşulu)
    if test_img.shape != template_img.shape:
        template_img = cv2.resize(template_img, (test_img.shape[1], test_img.shape[0]))
    
    # Mutlak Farkı Hesapla
    difference = cv2.absdiff(test_img, template_img) # 
    
    # Hata bölgelerini daha belirgin hale getirmek için eşikleme
    _, thresholded_diff = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
    
    return difference, thresholded_diff

# --- YENİ FONKSİYON 3: Histogram Eşitleme (CLAHE) ---
def apply_clahe(image):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) uygular.
    Görüntünün genel kontrastını artırarak, düşük kontrastlı ortamlarda 
    zor görünen küçük hataları ön plana çıkarır.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

# --- SUNUM İÇİN YENİ EKLEME: GAUSS BULANIKLIĞI ---
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Görüntüye Gaussian Bulanıklığı uygular.
    Gürültüyü azaltmak ve detayları yumuşatmak için kullanılır.
    """
    if len(image.shape) == 3:
        # Renkli görüntü ise gri tonlamaya çevir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # kernel_size tek sayılar olmalıdır
    return cv2.GaussianBlur(image, kernel_size, 0)
# ---------------------------------------------------

# --- SUNUM İÇİN YENİ EKLEME: ADAPTİF EŞİKLEME (BASİT SEGMENTASYON) ---
def apply_adaptive_thresholding(image):
    """
    Görüntüye yerel parlaklık farklılıklarına uyum sağlayan
    Adaptif Eşikleme (Adaptive Thresholding) uygular.
    PCB izlerini arka plandan ayırmak için temel segmantasyon örneği.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    thresholded_img = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, # Block Size
        2   # C
    )
    return thresholded_img


# --- SUNUM İÇİN YENİ EKLEME: CANNY KENAR TESPİTİ ---
def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Canny Kenar Tespiti algoritmasını uygular.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Gürültüyü azaltmak için Gauss Bulanıklığı uygula
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny algoritmasını uygula
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges
# -----------------------------------------------------------------------

# =========================================================================
# --- GÖRÜNTÜ İŞLEME FONKSİYONLARI BİTİŞİ ---
# =========================================================================


# --- Yeni Ek Açıklama: Bir Görüntüye Uygulanan Temel İşlemlerin Görselleştirilmesi ---

# Rastgele bir görüntü alıp yöntemleri uygulayalım
example_path = img_path_list[random.randint(0, len(img_path_list) - 1)]

# Şablon görüntüsü (PCB_USED klasöründen ilkini alalım)
template_files = os.listdir(template_dir)
template_path = os.path.join(template_dir, template_files[0]) if template_files else None

# 1. Orijinal ve Gri
original = cv2.imread(example_path)
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

if original is not None:
    # 2. HSV Dönüşümü
    hsv_img = convert_to_hsv(original)
    
    # 3. CLAHE
    clahe_img = apply_clahe(gray)
    
    # 4. Template Matching (Arka Plan Çıkarma)
    diff_img = np.zeros_like(gray)
    if template_path:
        diff_result, _ = apply_template_matching(example_path, template_path)
        if diff_result is not None:
            diff_img = diff_result

    # Görselleştirme
    plt.figure(figsize=(20, 5))
    
    titles = ['Orijinal (BGR)', 'HSV Dönüşümü (Sat)', 'CLAHE (Kontrast)', 'Arka Plan Farkı (AbsDiff)']
    
    # Görsel Listesi
    images = [
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
        hsv_img[:,:,1], # Saturation kanalı (Doygunluk) ayrıştırıcı olabilir
        clahe_img,
        diff_img
    ]
    
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        if i == 0:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap='gray')
            
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------


# =========================================================================
# --- SUNUM GÖRSELLEŞTİRMELERİ İÇİN EK KOD BAŞLANGICI ---
# =========================================================================

# --- A. GAUSS BULANIKLIĞI VE FARK GÖRSELLEŞTİRMESİ ---
if original is not None and template_path:
    print("\n--- A. Gauss Bulanıklığı ve Bulanık Fark Görselleştirmesi ---")
    
    # Şablonu da aynı şekilde gri ve bulanık hale getir
    template_img_blur = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_img_blur is not None:
        template_img_blur_resized = cv2.resize(template_img_blur, (gray.shape[1], gray.shape[0]))
        template_img_blurred = apply_gaussian_blur(template_img_blur_resized, (9, 9))
    
        # Test görüntüsüne bulanıklık uygula
        blurred_img = apply_gaussian_blur(original, kernel_size=(9, 9))
    
        # Bulanık test ve bulanık şablon arasındaki fark
        diff_blurred_result = cv2.absdiff(blurred_img, template_img_blurred)
        _, thresholded_blurred_diff = cv2.threshold(diff_blurred_result, 30, 255, cv2.THRESH_BINARY)
    
        # Görselleştirme
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Orijinal Görüntü')
        plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 2)
        plt.imshow(blurred_img, cmap='gray')
        plt.title('Gaussian Blur (Gürültü Azaltma)')
        plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 3)
        plt.imshow(thresholded_blurred_diff, cmap='gray')
        plt.title('Bulanık Fark + Eşikleme (Daha Temiz Hata)')
        plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show()
    else:
        print("UYARI: Şablon resmi yüklenemediği için Gauss Bulanıklığı farkı görselleştirilemedi.")


# --- B. ADAPTİF EŞİKLEME (BASİT SEGMENTASYON) GÖRSELLEŞTİRMESİ ---
if original is not None:
    print("\n--- B. Adaptif Eşikleme (Basit Segmantasyon) Görselleştirmesi ---")
    
    # 1. Adaptif Eşikleme Uygula (Segmantasyon)
    segmented_img = apply_adaptive_thresholding(original)
    
    # Görselleştirme
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Görüntü')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img, cmap='gray')
    plt.title('Adaptif Eşikleme (PCB İzi Segmentasyonu)') # 
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()

# --- C. CANNY KENAR TESPİTİ GÖRSELLEŞTİRMESİ ---
if original is not None:
    print("\n--- C. Canny Kenar Tespiti Görselleştirmesi ---")
    
    # Orijinal Görüntü üzerinde Canny Uygula
    canny_edges = apply_canny_edge_detection(original, low_threshold=50, high_threshold=150)
    
    # Görselleştirme
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Görüntü')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Kenar Tespiti (Kenar Vurgulama)')
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()
# --------------------------------------------------------------------

# =========================================================================
# --- SUNUM GÖRSELLEŞTİRMELERİ İÇİN EK KOD BİTİŞİ ---
# =========================================================================

img_dir = "PCB_DATASET/images"

# Define the source directories for each category inside img_dir
source_dirs = [
    os.path.join(img_dir, "Missing_hole"),
    os.path.join(img_dir, "Mouse_bite"),
    os.path.join(img_dir, "Open_circuit"),
    os.path.join(img_dir, "Short"),
    os.path.join(img_dir, "Spur"),
    os.path.join(img_dir, "Spurious_copper")
]

# Define the destination directory for the combined images (Writable location)
# Use a writable, cross-platform folder inside the project
destination_dir = os.path.join(input_dir, 'images_combined')

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Loop through each source directory and copy all files to the destination
for source_dir in source_dirs:
    if os.path.exists(source_dir):
        # Get all files in the current directory
        files = os.listdir(source_dir)
        
        # Copy each file to the destination directory
        for file in files:
            file_path = os.path.join(source_dir, file)
            if os.path.isfile(file_path):
                shutil.copy(file_path, destination_dir)
    else:
        print(f"Directory {source_dir} does not exist.")

# Now check how many files are in the destination folder
files_in_combined = os.listdir(destination_dir)
print(f"Number of files copied: {len(files_in_combined)}")


# Ensure destination exists and copy files
os.makedirs(destination_dir, exist_ok=True)

for source_dir in source_dirs:
    if os.path.exists(source_dir):
        for file in os.listdir(source_dir):
            file_path = os.path.join(source_dir, file)
            if os.path.isfile(file_path):
                shutil.copy(file_path, destination_dir)
    else:
        print(f"Directory {source_dir} does not exist.")

# Now check how many files are in the destination folder
files_in_combined = os.listdir(destination_dir)
print(f"Number of files copied: {len(files_in_combined)}")


# Visualizing the no of defects in each PCB
# Count the number of defects per filename and reset index to get a proper column
df_multiple_defects = df_annot['filename'].value_counts().reset_index(name='count')
df_multiple_defects.columns = ['filename', 'count']
sns.countplot(x='count', data=df_multiple_defects)
plt.xlabel('No of defects in one PCB')
plt.title('PCB Başına Hata Sayısı Dağılımı') # Başlık eklendi
plt.show() # Grafiği göstermek için eklendi


#Defin|g a function to view image along with bounding box

def draw_bounding_boxes(image_path, bounding_boxes, annotations=None):
    """
    Sınır kutularını ve etiketleri görüntü üzerine çizer.
    """
    
    # Load the image
    img = plt.imread(image_path)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15,10))

    # Display the image
    ax.imshow(img)

    # Draw each bounding box
    for idx, bbox in enumerate(bounding_boxes):
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y
        # Sınır kutusu
        rect = patches.Rectangle((min_x, min_y), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Etiketi sınır kutusunun sol üst köşesine ekle
        text_to_show = None
        if annotations is not None:
            try:
                text_to_show = annotations[idx]
            except Exception:
                text_to_show = str(annotations)

        if text_to_show:
            ax.text(min_x, min_y - 10, text_to_show,
                     color='red', fontsize=12, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.6, pad=3))

    plt.title(f"Hata Tespit Görselleştirme: {os.path.basename(image_path)}") # Başlık eklendi
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    
    
    # Show the plot
    plt.show()

#Getting filename from filepath
filepath=img_path_list[0]
filename=re.sub(r'.+/([\w_]+\.jpg)',r'\1',filepath)
print(filename)

def visualize_annotations(list_image_path, df):
    """
    Belirtilen resim yollarındaki etiketlenmiş hataları gösterir.
    """
    for i in list_image_path:
        filepath = i
        # Use os.path.basename to extract the filename so windows paths also work
        filename = os.path.basename(filepath)
        df_selected = df[df['filename'] == filename]
        
        # *** BURAYI EKLEYİN: Veri olup olmadığını kontrol et ***
        if df_selected.empty:
            print(f"UYARI: {filename} dosyası için ek açıklama (annotation) verisi bulunamadı ve atlandı.")
            continue # Döngünün bir sonraki öğesine geç
            
        width = df_selected['width'].values
        height = df_selected['height'].values
        class_name = df_selected['class'].values
        xmin = df_selected['xmin'].values
        # ... (Geri kalan kod aynı kalır)

        ymin = df_selected['ymin'].values
        xmax = df_selected['xmax'].values
        ymax = df_selected['ymax'].values

        bbox = list(zip(xmin, ymin, xmax, ymax))
        # Pass the class labels corresponding to each bounding box
        class_list = list(class_name)
        draw_bounding_boxes(filepath, bbox, class_list) # 

image_path_shuffle=img_path_list
random.shuffle(image_path_shuffle)

# visualize_annotations(image_path_shuffle[0:5],df_annot)      # Yorum satırından çıkarıldı