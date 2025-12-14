# PCB Defect Detection (Baskılı Devre Kartı Hata Tespiti)

Bu proje, geleneksel Bilgisayarlı Görme (Computer Vision - CV) tekniklerini kullanarak baskılı devre kartları (PCB) üzerindeki üretim hatalarını otomatik, hızlı ve güvenilir bir şekilde tespit etmeyi amaçlar. Temel yöntem olarak **Template Matching (Şablon Eşleştirme)** kullanılmıştır ve gelecekteki Derin Öğrenme (DL) entegrasyonu için veri etiketleme (Annotation) süreci tamamlanmıştır.

## 🎯 Proje Amacı ve Motivasyon

PCB'ler, elektronik cihazların temelini oluşturur. Üretim sürecindeki hataların manuel kontrolü yavaş, maliyetli ve insan hatasına açıktır.

* **Amaç:** Kusursuz (Altın Standart) bir referans görüntü ile hatalı PCB'yi karşılaştırarak, aradaki farkı izole etmek ve hatanın tipini belirlemektir.
* **Gelecek Vizyonu:** Mevcut Template Matching sisteminin zorluklarını (ışık değişimi, hizalama) aşmak için, etiketlenmiş veriyi kullanarak YOLO/CNN tabanlı bir Derin Öğrenme modeline geçiş altyapısını kurmaktır.

## 🖼️ Dataset ve Hata Türleri

Projede, her biri kritik öneme sahip 6 yaygın PCB hata türü incelenmiştir:

| Hata Tipi | Açıklama |
| :--- | :--- |
| **Short** | İki iletken iz arasında istenmeyen kısa devre. |
| **Open\_circuit** | İletken izde kopukluk. |
| **Missing\_hole** | Kart üzerindeki deliklerin eksik olması. |
| **Mouse\_bite** | İletken kenarında küçük çentikler veya aşınma. |
| **Spur** | İletken izden çıkan, istenmeyen küçük uzantı. |
| **Spurious\_copper** | İletken olmayan bölgelerde fazladan bakır artığı. |

## ⚙️ Sistem Mimarisi ve Akış

Proje, geleneksel CV yöntemlerini kullanarak aşağıdaki ardışık adımları izler: 

1.  **Görüntü Girişi:** Hatalı Kart ve Kusursuz Referans Kartı alınır.
2.  **Ön İşleme:** Gürültü kontrolü (Gauss Blur), kontrast optimizasyonu (CLAHE) ve gri tonlama/HSV dönüşümü uygulanır.
3.  **Template Matching (Fark Alma):** Ön işlenmiş hatalı kart ile ön işlenmiş referans kart arasındaki **Mutlak Fark** hesaplanır.
4.  **Hata İzolasyonu:** Fark görüntüsüne eşikleme (Thresholding) uygulanarak sadece hata bölgeleri beyaz olarak izole edilir.
5.  **Görselleştirme:** İzole edilen hatanın etrafına Sınır Kutusu (Bounding Box) çizilir ve hata tipi etiketlenir.

## 💻 Kullanılan Temel Görüntü İşleme Teknikleri

| Teknik | Amaç |
| :--- | :--- |
| **Gauss Bulanıklığı** | Kameradan kaynaklanan gürültü ve yüzey pürüzlerini temizleyerek sahte hata tespitini engeller. |
| **CLAHE** | Düşük kontrastlı, gölgeli bölgelerdeki küçük hataların görünürlüğünü artırır. |
| **cv2.absdiff()** | İki görüntü arasındaki piksel farkını mutlak değer olarak hesaplayarak hata konumunu belirler. |
| **Eşikleme (Thresholding)** | Fark görüntüsünden, arka planı kaldırarak yalnızca hatanın pikselini beyaz olarak izole eder. |

## 🔍 Veri Etiketleme (Annotation)

Projenin en kritik adımlarından biri, gelecekteki DL modelleri için veri hazırlığıdır.

* **Format:** Tüm hatalar, **PASCAL VOC** formatında XML dosyaları kullanılarak etiketlenmiştir.
* **İçerik:** Her bir XML dosyası, ilgili görüntüdeki her bir hata için koordinatları (`xmin, ymax, xmax, ymin`) ve sınıf etiketini içerir.
* **Amaç:** Bu etiketli veriler, ileride **YOLO** veya **Faster R-CNN** gibi Derin Öğrenme tabanlı Nesne Tespiti modellerini eğitmeyi ve referans kart bağımlılığını tamamen ortadan kaldırmayı sağlayacaktır. 

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler

* Python 3.x
* OpenCV (`cv2`)
* NumPy
* Matplotlib
* lxml (XML işlemleri için)

```bash
pip install opencv-python numpy matplotlib
