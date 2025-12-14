import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Plotly'yi opsiyonel olarak içe aktar
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def load_image(image_file):
    """Görüntüyü yükler ve OpenCV formatına (numpy array) çevirir."""
    img = Image.open(image_file)
    return np.array(img)

def resize_images(img1, img2):
    """İkinci görüntüyü birincinin boyutlarına getirir."""
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    return img2_resized

def detect_defects(ref_img, test_img, threshold_value=50):
    """
    Referans ve test görüntüleri arasındaki farkı bularak hataları tespit eder.
    """
    if len(ref_img.shape) == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    else:
        ref_gray = ref_img
        test_gray = test_img
    
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 0)
    
    diff = cv2.absdiff(ref_blur, test_blur)
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    # --- YENİ EKLENEN KISIM: Arka Plan Temizleme (Maskeleme) ---
    # Referans görüntüden PCB'nin kendisini bulup dışını maskeliyoruz.
    # PCB genellikle koyu/açık zıtlığına sahiptir veya en büyük nesnedir.
    
    # 1. Referans görüntüde eşikleme yaparak PCB'yi bulmaya çalış
    # (Not: Işıklandırmaya göre 50-255 arası değişebilir, 30 genel bir değerdir)
    _, mask_thresh = cv2.threshold(ref_gray, 30, 255, cv2.THRESH_BINARY)
    
    # 2. Konturları bul (En büyük parça PCB'dir)
    contours_ref, _ = cv2.findContours(mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_ref:
        # En büyük alana sahip konturu PCB kabul et
        largest_cnt = max(contours_ref, key=cv2.contourArea)
        
        # Siyah bir maske oluştur
        mask = np.zeros_like(ref_gray)
        
        # PCB alanını beyaza boya (İçini doldur)
        cv2.drawContours(mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
        
        # Maskeyi biraz daralt (Kenar parlama/hizalama hatalarını önlemek için)
        kernel_erode = np.ones((15, 15), np.uint8)
        mask = cv2.erode(mask, kernel_erode, iterations=2)
        
        # Fark görüntüsünü (thresh) bu maskeyle çarp (Dışarısı 0 olur)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        
    # -----------------------------------------------------------
    
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, diff, dilated

def classify_defect(cnt, filename=""):
    """
    Hatanın türünü tahmin etmeye çalışır.
    1. Önce dosya ismine bakar (Eğer test verisi kullanılıyorsa en kesin yöntem).
    2. Dosya isminde yoksa, şekil özelliklerine (Büyüklük, Yuvarlaklık) bakar.
    """
    filename = filename.lower()
    
    # 1. Dosya İsminden Tespit (Dataset isimleri ipucu içerir)
    if "missing_hole" in filename or "missing hole" in filename: return "Missing Hole (Delik Yok)"
    if "mouse_bite" in filename or "mouse bite" in filename: return "Mouse Bite (Fare Isırığı)"
    if "open_circuit" in filename or "open circuit" in filename: return "Open Circuit (Açık Devre)"
    if "short" in filename: return "Short (Kısa Devre)"
    if "spur" in filename: return "Spur (Çapak)"
    if "spurious_copper" in filename or "spurious copper" in filename: return "Spurious Copper (Bakır Fazlalığı)"
    
    # 2. Şekilsel Analiz (Heuristic - Basit Tahmin)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter == 0: return "Bilinmiyor"
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Yuvarlaklık 1'e yakınsa muhtemelen bir deliktir
    if circularity > 0.75:
        return "Missing Hole (Tahmin)"
    
    # Çok küçük alanlar genelde Spur veya Bakır artığıdır
    if area < 50:
        return "Spur/Noise (Tahmin)"
        
    return "Genel Hata (Open/Short)"

def draw_defects(img, contours, min_area=10, filename=""):
    """
    Tespit edilen hataların etrafına kutu çizer ve türünü yazar.
    """
    img_copy = img.copy()
    defect_count = 0
    detected_types = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Hatayı sınıflandır
            defect_type = classify_defect(cnt, filename)
            detected_types.append(f"Hata {defect_count+1}: {defect_type}")
            
            # Kutu çiz (Kırmızı)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Etiket yaz
            label = f"{defect_count+1}" # Sadece numara yaz, karışıklık olmasın
            cv2.putText(img_copy, label, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            defect_count += 1
            
    return img_copy, defect_count, detected_types

# --- Streamlit Arayüzü ---

st.set_page_config(page_title="PCB Hata Tespit Sistemi", layout="wide")

st.title("🔍 PCB Hata Tespit ve Analiz Sistemi")

# Sabitler
TEMPLATE_DIR = "PCB_DATASET/PCB_USED"

# Yan Panel
st.sidebar.header("⚙️ Ayarlar")

# 1. Test Görüntüsü Yükleme (Artık referans yüklemeye gerek yok)
uploaded_test = st.sidebar.file_uploader("Test Edilecek (Hatalı) Kartı Yükle", type=["jpg", "png", "jpeg"])

# 2. Hata Türü Seçimi (Kullanıcı İsteği)
defect_types = ["Genel Tarama", "Mouse Bite", "Missing Hole", "Open Circuit", "Short", "Spur", "Spurious Copper"]
selected_defect = st.sidebar.selectbox("Aranan Hata Türü (Opsiyonel)", defect_types)

# 3. İleri Ayarlar
with st.sidebar.expander("Gelişmiş Ayarlar"):
    threshold_val = st.slider("Hassasiyet (Threshold)", 10, 255, 50)
    min_area_val = st.slider("Min. Hata Boyutu", 0, 500, 20)
    # Manuel PCB ID seçimi (Dosya isminden bulunamazsa diye)
    manual_pcb_id = st.selectbox("PCB ID (Otomatik Tanımazsa)", ["Otomatik"] + [f"{i:02d}" for i in range(1, 13)])

if uploaded_test:
    st.divider()
    
    # Test Görüntüsünü Yükle
    test_image = load_image(uploaded_test)
    
    # --- OTOMATİK REFERANS BULMA ---
    ref_image = None
    pcb_id = None
    
    # 1. Dosya isminden ID'yi dene (Örn: "01_mouse_bite..." -> "01")
    filename = uploaded_test.name
    try:
        # İlk "_" öncesini al
        detected_id = filename.split('_')[0]
        # Sayı kontrolü yap
        if detected_id.isdigit():
            pcb_id = detected_id
    except:
        pass

    # 2. Manuel seçim varsa onu kullan (Override)
    if manual_pcb_id != "Otomatik":
        pcb_id = manual_pcb_id
        
    # 3. Referans dosyasını yükle
    if pcb_id:
        ref_path = os.path.join(TEMPLATE_DIR, f"{pcb_id}.JPG") # JPG büyük harf dataset formatında
        if os.path.exists(ref_path):
            st.sidebar.success(f"✅ Referans Bulundu: {pcb_id}.JPG")
            ref_image_pil = Image.open(ref_path)
            ref_image = np.array(ref_image_pil)
        else:
            # Belki küçük harf .jpg dir?
            ref_path_lower = os.path.join(TEMPLATE_DIR, f"{pcb_id}.jpg")
            if os.path.exists(ref_path_lower):
                 st.sidebar.success(f"✅ Referans Bulundu: {pcb_id}.jpg")
                 ref_image_pil = Image.open(ref_path_lower)
                 ref_image = np.array(ref_image_pil)
            else:
                 st.sidebar.error(f"❌ Referans dosyası bulunamadı: {ref_path}")
    else:
        st.sidebar.warning("⚠️ PCB ID dosya isminden okunamadı. Lütfen 'Gelişmiş Ayarlar'dan PCB ID seçin.")

    # Analiz
    if ref_image is not None:
        # Boyut Eşitleme
        test_image = resize_images(ref_image, test_image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(ref_image, caption=f"Referans PCB ({pcb_id})", use_column_width=True)
        with col2:
            st.image(test_image, caption="Test Edilecek PCB", use_column_width=True)
            
        if st.button("Hataları Analiz Et"):
            with st.spinner(f'{selected_defect} analizi yapılıyor...'):
                contours, diff_img, thresh_img = detect_defects(ref_image, test_image, threshold_val)
                
                # Dosya ismini sınıflandırma için gönder
                result_img, count, found_types = draw_defects(test_image, contours, min_area_val, filename)
                
                st.divider()
                
                if count > 0:
                     st.warning(f"⚠️ {count} adet farklılık tespit edildi.")
                     
                     # Hata Türlerini Listele
                     st.subheader("📋 Hata Raporu:")
                     for d_type in found_types:
                         st.write(f"- {d_type}")
                else:
                     st.success("✅ Hata tespit edilemedi.")
    
                # --- SONUÇLARI GÖSTER ---
                st.subheader("🔎 Detaylı İnceleme")
                
                if PLOTLY_AVAILABLE:
                    fig = px.imshow(result_img)
                    fig.update_layout(dragmode='pan')
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("ℹ️ Yakınlaştırmak için görsel üzerinde fareyi kullanabilirsiniz.")
                else:
                    st.image(result_img, caption="Sonuç", use_column_width=True)
                
                st.info("ℹ️ Sistem otomatik olarak uygun referans PCB görselini veritabanından çekip karşılaştırmıştır.")

    else:
        st.info("Analiz için geçerli bir referans görüntüsü eşleştirilemedi.")
            
else:
    st.info("Lütfen başlamak için test edilecek kartı yükleyin. Sistem referans kartı otomatik bulacaktır.")

