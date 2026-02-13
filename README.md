# 🎯 Computer Vision 101

<p align="center">
  <strong>🇹🇷 Türkçe | Uygulamalı | 3 Günlük Yoğun Eğitim</strong>
</p>

---

Bu repo, [**Sakarya Üniversitesi Yapay Zeka Topluluğu**](https://www.linkedin.com/company/sauyapayzekaa/) ve [**Türkiye Yapay Zeka Topluluğu**](https://turkishai.community/) ortaklığında düzenlenen **Computer Vision 101** eğitiminin tüm kaynaklarını içermektedir.

## 👩‍🏫 Eğitimci

<a href="https://www.linkedin.com/in/aysenur-tak/">
  <img src="https://img.shields.io/badge/Ayşenur_Tak-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
</a>

**Ayşenur Tak** - Türkiye Yapay Zeka Topluluğu Community Lead

---

## 📚 İçerik

<details>
<summary><strong>📅 GÜN 1 - Görüntü İşleme Temelleri</strong></summary>

### 📓 Notebook: `DAY-1/Comp-Vis-Day-1.ipynb`

Bu günde bilgisayarlı görünün temellerini ve OpenCV ile görüntü işleme tekniklerini öğreniyoruz.

#### Kapsanan Konular:

**🔹 Computer Vision'a Giriş**
- Bilgisayarlı görü nedir?
- Tarihçe ve temel problemler (1960'lardan günümüze)
- Gerçek dünya uygulamaları (sağlık, otonom araçlar, güvenlik, perakende vb.)
- İnsan gözü vs bilgisayar karşılaştırması

**🔹 Görüntü Formatları ve Temsili**
- Dijital görüntü formatları (JPG, PNG, BMP, TIFF, WebP)
- Video formatları (MP4, AVI, MKV)
- Piksel matrisi kavramı

**🔹 Renk Uzayları**
- RGB, BGR, Grayscale
- HSV (Hue, Saturation, Value)
- LAB (CIELAB)

**🔹 Geometrik Dönüşümler**
- Scaling (yeniden boyutlandırma)
- Translation (öteleme)
- Rotation (döndürme)
- Flip (çevirme)

**🔹 Görüntü İyileştirme**
- Histogram analizi
- Histogram eşitleme
- Normalizasyon

**🔹 Thresholding (Eşikleme)**
- Simple thresholding
- Adaptive thresholding
- Otsu thresholding
- Basit segmentasyon

**🔹 Filtreleme ve Gürültü Azaltma**
- Konvolüsyon (Evrişim)
- Gaussian Blur
- Median Blur
- Gürültü tipleri (Salt & Pepper, Gaussian)

**🔹 Morfolojik İşlemler**
- Erosion (aşındırma)
- Dilation (genişletme)
- Opening ve Closing

**🔹 Kenar Tespiti**
- Canny Edge Detection
- Sobel operatörü

**🔹 Hough Transform**
- Doğru tespiti (`HoughLinesP`)
- Çember tespiti

**🔹 Kontur ve Şekil Analizi**
- `findContours()` ile kontur bulma
- Bounding box ve ROI (Region of Interest)

#### 🎬 Video İşleme: `DAY-1/video-capturing.py`

Canlı kamera üzerinde farklı modlarla çalışma:
- Normal görüntü
- Negatif (invert)
- Canny kenar tespiti
- **Motion Detection** (hareket algılama) - Background Subtractor kullanarak

</details>

<details>
<summary><strong>📅 GÜN 2 - Feature Detection & Object Detection</strong></summary>

### 📓 Notebook: `DAY-2/Comp-Vis-Day-2.ipynb`

Bu günde öznitelik çıkarımı, nesne tespiti ve derin öğrenme temellerini öğreniyoruz.

#### Kapsanan Konular:

**🔹 Feature Detection Temelleri**
- Kenar, köşe ve blob kavramları
- Aperture problemi
- Canny/Sobel vs Feature Detection algoritmaları farkı

**🔹 Köşe Tabanlı Yöntemler**
- **Harris Corner Detection** - Gradyan matrisi ve özdeğerler
- **Shi-Tomasi (Good Features to Track)** - Harris'in iyileştirilmiş versiyonu

**🔹 Modern Feature Detectors**
- **ORB (Oriented FAST + BRIEF)** - Hızlı ve verimli
- **SIFT (Scale-Invariant Feature Transform)** - Ölçek ve dönüşe dayanıklı
- Keypoint + Descriptor kavramları

**🔹 Template Matching**
- SIFT ile template matching
- ORB ile template matching
- Homography ve perspektif dönüşümü

**🔹 HOG (Histogram of Oriented Gradients)**
- Yaya tespiti için klasik yöntem
- Gradient yönelim histogramları

**🔹 Haar Cascades**
- Viola-Jones algoritması
- Yüz ve göz tespiti
- Cascade mantığı

**🔹 CNN (Convolutional Neural Networks)**
- ANN vs DNN farkı
- CNN katmanları:
  - Convolution (Evrişim)
  - Aktivasyon fonksiyonları (ReLU)
  - Pooling (Havuzlama)
  - Flattening
  - Fully Connected
- MNIST örneği ile CNN eğitimi

**🔹 YOLO (You Only Look Once)**
- Single-stage object detection
- YOLOv3 ile nesne tespiti
- COCO dataset sınıfları

**🔹 Gelecek Perspektifi**
- CNN mimarileri (ResNet, EfficientNet)
- Vision Transformers (ViT)
- Multimodal modeller (CLIP, GPT-4o) ve ilgili modellerin paperları

#### 📁 Ek Dosyalar

- `DAY-2/MNIST_cnn_model.ipynb` - CNN model eğitimi notebook'u
- `DAY-2/panaroma/` - Panorama stitching örneği
- `DAY-2/yolo-source/` - YOLO model dosyaları ve COCO sınıfları

</details>

<details>
<summary><strong>📅 GÜN 3 - Uygulamalı Projeler</strong></summary>

Üçüncü gün, öğrendiklerimizi gerçek dünya projelerine uyguluyoruz. Her proje kendi klasöründe detaylı README ile birlikte sunulmaktadır.

### 📁 Proje 1: File Scanner (Dosya Tarayıcı)

**Konum:** `DAY-3/File-Scanner/`

Kağıt belge fotoğraflarını profesyonel taranmış PDF'lere dönüştüren uygulama.

**Kullanılan Teknikler:**
- Canny Edge Detection
- Kontur analizi ve dörtgen tespiti
- Douglas-Peucker algoritması
- **Perspektif dönüşümü (Homography)**
- Adaptif eşikleme
- Morfolojik işlemler

**Özellikler:**
- ✅ Eğik çekilmiş belgeleri düzeltme
- ✅ Perspektif bozulmasını giderme
- ✅ Kontrast artırma
- ✅ A4/A5 boyutunda PDF çıktı
- ✅ Streamlit web arayüzü

```bash
cd DAY-3/File-Scanner
pip install -r requirements.txt
streamlit run app.py
```

---

### 📁 Proje 2: Image Captioning with CLIP

**Konum:** `DAY-3/Image-Captioning/`

OpenAI CLIP modeli ile zero-shot görüntü sınıflandırma ve analiz.

**Kullanılan Teknikler:**
- **CLIP (Contrastive Language-Image Pre-training)**
- Vision Transformer (ViT)
- Zero-shot learning
- Cosine similarity
- Multimodal embedding

**Özellikler:**
- ✅ Görüntü sınıflandırma (kedi, köpek, araba vb.)
- ✅ Görüntü tipi tespiti (meme, fotoğraf, çizim)
- ✅ Duygu analizi
- ✅ Özel etiketlerle eşleştirme
- ✅ Görüntü-metin benzerlik skoru

```bash
cd DAY-3/Image-Captioning
pip install -r requirements.txt
streamlit run app.py
```

---

### 📁 Proje 3: Realtime Car Detection

**Konum:** `DAY-3/Realtime-Car-Detection/`

YOLOv8 ile videolarda gerçek zamanlı araç tespiti.

**Kullanılan Teknikler:**
- **YOLOv8 (Ultralytics)**
- CSPDarknet backbone
- FPN + PANet neck
- Non-Maximum Suppression (NMS)
- Confidence thresholding

**Özellikler:**
- ✅ Gerçek zamanlı video analizi
- ✅ Bounding box çizimi
- ✅ Confidence skoru gösterimi
- ✅ FPS ve istatistik takibi
- ✅ Model boyutu seçimi (nano → xlarge)

```bash
cd DAY-3/Realtime-Car-Detection
pip install -r requirements.txt
streamlit run app.py
```

</details>

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.8+
- pip

### Hızlı Başlangıç

```bash
# Repoyu klonlayın
git clone https://github.com/Turkish-AI-Community/sakarya-uni-yapay-zeka-toplulugu-computer-vision-101
cd SAU-Yapay-Zeka-ComVis

# Sanal ortam oluşturun (önerilir)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# veya
venv\Scripts\activate     # Windows

# Temel bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Ana Bağımlılıklar

```txt
opencv-python    # Görüntü işleme
matplotlib       # Görselleştirme
numpy            # Sayısal hesaplamalar
tensorflow       # Derin öğrenme (CNN)
```

> 💡 **Not:** Her GÜN-3 projesi kendi `requirements.txt` dosyasına sahiptir.

---

## 📂 Proje Yapısı

```
SAU-Yapay-Zeka-ComVis/
│
├── 📁 DAY-1/                          # Görüntü İşleme Temelleri
│   ├── 📓 Comp-Vis-Day-1.ipynb        # Ana notebook
│   ├── 🐍 video-capturing.py          # Canlı kamera + motion detection
│   └── 📁 img/                        # Örnek görseller
│
├── 📁 DAY-2/                          # Feature & Object Detection
│   ├── 📓 Comp-Vis-Day-2.ipynb        # Ana notebook
│   ├── 📓 MNIST_cnn_model.ipynb       # CNN eğitim notebook'u
│   ├── 📁 img/                        # Örnek görseller
│   ├── 📁 panaroma/                   # Panorama stitching örneği
│   └── 📁 yolo-source/                # YOLO model dosyaları (*)
│
├── 📁 DAY-3/                          # Uygulamalı Projeler
│   ├── 📁 File-Scanner/               # Belge tarayıcı
│   ├── 📁 Image-Captioning/           # CLIP ile görüntü analizi
│   └── 📁 Realtime-Car-Detection/     # YOLOv8 araç tespiti
│
├── 📄 requirements.txt                # Ana bağımlılıklar
└── 📄 README.md                       # Bu dosya
```

> 💡 **Not:** DAY-2/yolo-source yanlızca `coco.names` içermektedir. `yolov3.weights`, `yolov3.cfg` dosyaları eklenmemiştir.
---

## 🎓 Öğrenme Yolu

| Gün | Seviye | Odak | Çıktı |
|-----|--------|------|-------|
| **1** | 🟢 Başlangıç | OpenCV temelleri, görüntü işleme | Temel CV operasyonları |
| **2** | 🟡 Orta | Feature detection, CNN, YOLO | Nesne tespiti anlayışı |
| **3** | 🔴 İleri | Gerçek dünya projeleri | 3 çalışan uygulama |


---

## 🤝 Topluluklar

<p align="center">
  <a href="https://www.linkedin.com/company/sauyapayzekaa/">
    <img src="https://img.shields.io/badge/SAU_Yapay_Zeka-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="SAU YZ">
  </a>
  <a href="https://turkishai.community/">
    <img src="https://img.shields.io/badge/Türkiye_Yapay_Zeka_Topluluğu-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="TR AI">
  </a>
</p>

---

## 📜 Lisans

Bu proje eğitim amaçlı hazırlanmıştır. Özgürce kullanabilir ve geliştirebilirsiniz.

---

<p align="center">
  <strong>🚀 Happy Learning!</strong>
</p>
