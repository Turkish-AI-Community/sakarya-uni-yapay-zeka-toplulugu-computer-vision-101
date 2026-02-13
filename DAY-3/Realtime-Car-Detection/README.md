# Realtime Car Detection

YOLOv8 modeli kullanarak videolarda gercek zamanli araba tespiti yapan bir web uygulamasi. Streamlit arayuzu ile yuklediginiz videoda arabalarin tespit edilmesini canli olarak izleyebilirsiniz.

## Ne Ise Yarar?

Bu uygulama, yukleyeceginiz herhangi bir videoyu analiz eder ve:

- Videodaki arabalari gercek zamanli tespit eder
- Her araba icin bounding box (sinir kutusu) cizer
- Confidence (guven) skorunu gosterir
- Anlık ve toplam tespit sayisini takip eder
- FPS ve islem suresi istatistiklerini gosterir
- Opsiyonel olarak tum tasitlari (kamyon, otobus, motosiklet) tespit eder

## Uygulama Akis Diyagrami

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REALTIME CAR DETECTION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────┐
                              │  BASLANGIC    │
                              │ Video Yukle   │
                              └───────┬───────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VIDEO OKUMA                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ cv2.VideoCapture│───▶│ Frame Okuma     │───▶│ Video Bilgileri        │  │
│  │ (video_path)    │    │ (BGR Format)    │    │ (FPS, Boyut, Sure)     │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │   FRAME DONGUSU     │◄─────────────────────┐
                           │   (While Loop)      │                      │
                           └──────────┬──────────┘                      │
                                      │                                 │
                      ┌───────────────┴───────────────┐                 │
                      │                               │                 │
                  FRAME VAR                      FRAME YOK              │
                      │                               │                 │
                      ▼                               ▼                 │
┌─────────────────────────────────┐         ┌────────────────┐          │
│ FRAME ATLAMA KONTROLU           │         │     BITIS      │          │
│ (skip_frames > 0?)              │         │ Video Bitti    │          │
└─────────────┬───────────────────┘         └────────────────┘          │
              │                                                          │
              ▼                                                          │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        YOLO INFERENCE (Tespit)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         ON-ISLEME                                   │    │
│  │  ┌───────────┐    ┌───────────────┐    ┌───────────────────────┐   │    │
│  │  │ BGR Frame │───▶│ Boyutlandirma │───▶│ Normalizasyon (0-1)   │   │    │
│  │  │ (HxWx3)   │    │ (640x640)     │    │ BGR -> RGB            │   │    │
│  │  └───────────┘    └───────────────┘    └───────────┬───────────┘   │    │
│  └────────────────────────────────────────────────────┼────────────────┘    │
│                                                       │                     │
│                                                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       BACKBONE (CSPDarknet)                         │    │
│  │                                                                     │    │
│  │    Goruntu ───▶ Conv ───▶ C2f ───▶ SPPF ───▶ Feature Maps          │    │
│  │                                                                     │    │
│  │    Cikti: P3 (1/8), P4 (1/16), P5 (1/32) olceklerinde              │    │
│  └──────────────────────────────────┬──────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       NECK (FPN + PANet)                            │    │
│  │                                                                     │    │
│  │         P5 ────────────────────────▶ ┐                              │    │
│  │          │                           │                              │    │
│  │          ▼                           │                              │    │
│  │    ┌──────────┐                      │                              │    │
│  │    │ Upsample │                      │  Feature                     │    │
│  │    └────┬─────┘                      │  Fusion                      │    │
│  │         ▼                            │                              │    │
│  │    P4 + P5' ─────────────────────▶   │                              │    │
│  │          │                           │                              │    │
│  │          ▼                           ▼                              │    │
│  │    ┌──────────┐              Multi-Scale                            │    │
│  │    │ Upsample │              Features                               │    │
│  │    └────┬─────┘                                                     │    │
│  │         ▼                                                           │    │
│  │    P3 + P4' ─────────────────────▶                                  │    │
│  └──────────────────────────────────┬──────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       HEAD (Decoupled Head)                         │    │
│  │                                                                     │    │
│  │    ┌─────────────────────┐      ┌─────────────────────┐             │    │
│  │    │ Siniflandirma Dali  │      │   Regresyon Dali    │             │    │
│  │    │ (80 sinif icin)     │      │   (x, y, w, h)      │             │    │
│  │    └──────────┬──────────┘      └──────────┬──────────┘             │    │
│  │               │                            │                        │    │
│  │               └────────────┬───────────────┘                        │    │
│  │                            │                                        │    │
│  │                            ▼                                        │    │
│  │                  ┌──────────────────┐                               │    │
│  │                  │ Ham Tahminler    │                               │    │
│  │                  │ (Binlerce kutu)  │                               │    │
│  │                  └────────┬─────────┘                               │    │
│  └───────────────────────────┼─────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       POST-PROCESSING                               │    │
│  │                                                                     │    │
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│    │
│  │   │ Confidence      │───▶│ NMS (Non-Max   │───▶│ Sinif           ││    │
│  │   │ Filtreleme      │    │ Suppression)   │    │ Filtreleme      ││    │
│  │   │ (conf > 0.5)    │    │ (IoU > 0.45)   │    │ (Sadece araba)  ││    │
│  │   └─────────────────┘    └─────────────────┘    └────────┬────────┘│    │
│  │                                                          │         │    │
│  └──────────────────────────────────────────────────────────┼─────────┘    │
│                                                             │              │
└─────────────────────────────────────────────────────────────┼──────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NON-MAXIMUM SUPPRESSION (NMS)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ONCE:                              SONRA:                                 │
│   ┌─────────────────────┐            ┌─────────────────────┐                │
│   │  ┌───┐ ┌───┐        │            │     ┌───┐           │                │
│   │  │ A │ │ B │        │            │     │ A │           │                │
│   │  └───┘ └───┘        │  ───────▶  │     └───┘           │                │
│   │    └─┬─┘            │   NMS      │                     │                │
│   │      │ IoU > 0.45   │            │   Tek kutu kalir    │                │
│   └──────┼──────────────┘            └─────────────────────┘                │
│          │                                                                  │
│          │   IoU = Kesisim / Birlesim                                       │
│          │                                                                  │
│          │   ┌─────┬─────┐           IoU = 0.0  → Farkli nesne              │
│          │   │     │/////│           IoU = 1.0  → Ayni kutu                 │
│          │   │     │/////│           IoU > 0.45 → Muhtemelen ayni nesne     │
│          │   └─────┴─────┘                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GORSELLESTIME                                       │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                                                                   │     │
│   │    ┌─────────────────────┐                                        │     │
│   │    │ car: 95%            │  ◄── Etiket + Confidence               │     │
│   │    ├─────────────────────┤                                        │     │
│   │    │                     │                                        │     │
│   │    │   ┌───────────┐     │                                        │     │
│   │    │   │           │     │  ◄── Bounding Box (Yesil)              │     │
│   │    │   │   ARABA   │     │                                        │     │
│   │    │   │           │     │                                        │     │
│   │    │   └───────────┘     │                                        │     │
│   │    │                     │                                        │     │
│   │    └─────────────────────┘                                        │     │
│   │                                                    ┌────────────┐ │     │
│   │                                                    │ FPS: 45.2 │ │     │
│   │                                                    └────────────┘ │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│   cv2.rectangle() + cv2.putText()                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ISTATISTIK GUNCELLEME                                  │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│   │ Anlik Tespit:   │  │ Toplam Tespit:  │  │ FPS = frame_count / sure    │ │
│   │      3          │  │      127        │  │      45.2                   │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
│                                                                             │
│   Progress Bar: [████████████████████░░░░░░░░░░] 65%                        │
│                                                                             │
└───────────────────────────────────────────────────────┬─────────────────────┘
                                                        │
                                                        │
                                    ┌───────────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │ Sonraki Frame  │────────────────────┐
                           │ ret, frame =   │                    │
                           │ cap.read()     │                    │
                           └────────────────┘                    │
                                                                 │
                                    ┌────────────────────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │     BITIS      │
                           │ Sonuc Ozeti    │
                           │ - Toplam Tespit│
                           │ - Ortalama FPS │
                           │ - Toplam Sure  │
                           └────────────────┘
```

## YOLO Nedir?

**YOLO (You Only Look Once)**, gercek zamanli nesne tespiti icin tasarlanmis bir derin ogrenme modelidir. 2015 yilinda Joseph Redmon tarafindan gelistirilmistir.

### Temel Ozellikler

| Ozellik | Aciklama |
|---------|----------|
| **Gercek Zamanli** | Tek bir ileri gecis ile tespit (cok hizli) |
| **End-to-End** | Dogrudan goruntuden siniflandirmaya |
| **Unified** | Tek bir sinir agi, tek bir tahmin |
| **COCO Dataset** | 80 sinif uzerinde egitilmis |

### YOLO Nasil Calisir?

YOLO, goruntulerde nesne tespitini su adimlarla yapar:

1. **Grid Bolme**
   - Goruntu S x S'lik bir grida bolunur
   - Her grid hucresinden tahmin yapilir

2. **Bounding Box Tahmini**
   - Her hucre B adet bounding box tahmin eder
   - Her kutu icin: x, y, w, h, confidence

3. **Sinif Tahmini**
   - Her hucre C sinif olasiligi tahmin eder
   - Tum siniflar icin tek tahmin

4. **Non-Maximum Suppression (NMS)**
   - Cakisan kutular elenir
   - En iyi kutular secilir

```
Goruntu --> Grid --> BBox + Sinif Tahmini --> NMS --> Sonuc
  |           |              |                 |         |
640x640    13x13      x,y,w,h,conf,class    Filtreleme  Kutular
```

### YOLOv8 Modelleri

| Model | Parametre | mAP | Hiz (ms) | Kullanim |
|-------|-----------|-----|----------|----------|
| YOLOv8n (nano) | 3.2M | 37.3 | 0.6 | Mobil/Edge |
| YOLOv8s (small) | 11.2M | 44.9 | 0.9 | Genel kullanim |
| YOLOv8m (medium) | 25.9M | 50.2 | 1.7 | Dengeli |
| YOLOv8l (large) | 43.7M | 52.9 | 2.9 | Yuksek dogruluk |
| YOLOv8x (xlarge) | 68.2M | 53.9 | 4.7 | Maksimum dogruluk |

*mAP: COCO val2017 uzerinde mean Average Precision*
*Hiz: NVIDIA A100 GPU uzerinde*

### YOLO vs Diger Yontemler

| Ozellik | YOLO | R-CNN | SSD |
|---------|------|-------|-----|
| Hiz | Cok hizli | Yavas | Hizli |
| Dogruluk | Yuksek | En yuksek | Orta |
| Mimari | Single-stage | Two-stage | Single-stage |
| Gercek zamanli | Evet | Hayir | Evet |

## Kullanilan Teknolojiler

| Teknoloji | Aciklama |
|-----------|----------|
| **Python 3.8+** | Programlama dili |
| **PyTorch** | Derin ogrenme kutuphanesi |
| **Ultralytics** | YOLOv8 implementasyonu |
| **OpenCV** | Video ve goruntu isleme |
| **Streamlit** | Web arayuzu |
| **NumPy** | Sayisal hesaplamalar |

## Kurulum

```bash
# Proje klasorune gidin
cd Realtime-Car-Detection

# Sanal ortam olusturun (onerilir)
python -m venv venv

# Sanal ortami aktiflestirin
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Bagimliliklari yukleyin
pip install -r requirements.txt

# Veya manuel olarak:
pip install ultralytics torch torchvision opencv-python streamlit numpy pillow
```

**Not:** GPU kullanmak icin uygun PyTorch versiyonunu yukleyin:
- https://pytorch.org/get-started/locally/

```bash
# CUDA 11.8 icin ornek:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Kullanim

### Web Arayuzu (Streamlit)

```bash
streamlit run app.py
```

Tarayicinizda `http://localhost:8501` adresinde acilir.

**Ozellikler:**
- Video yukleme (MP4, AVI, MOV, MKV, WEBM)
- Model boyutu secimi (nano, small, medium, large, xlarge)
- Confidence threshold ayari
- Frame atlama (performans icin)
- Gercek zamanli istatistikler
- FPS gosterimi

### Python ile Kullanim

```python
from detector import CarDetector, create_detector
import cv2

# Detektoru olustur
detector = create_detector(
    model_size="nano",      # Model boyutu
    confidence=0.5,         # Confidence threshold
    detect_all_vehicles=False  # Sadece araba
)

# Tek goruntu uzerinde tespit
image = cv2.imread("traffic.jpg")
output, detections = detector.detect(image)

# Sonuclari goster
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.0%}")
    print(f"  Konum: {det['bbox']}")

# Islenmis goruntuyu kaydet
cv2.imwrite("output.jpg", output)
```

### Video Isleme

```python
import cv2
from detector import CarDetector

detector = CarDetector()

cap = cv2.VideoCapture("traffic_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Tespit yap
    output, detections = detector.detect(frame)
    
    print(f"Tespit edilen araba sayisi: {len(detections)}")
    
    # Goster
    cv2.imshow("Car Detection", output)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Proje Yapisi

```
Realtime-Car-Detection/
│
├── app.py              # Streamlit web arayuzu
│                       # - Video yukleme
│                       # - Gercek zamanli tespit
│                       # - Istatistik gosterimi
│
├── detector.py         # YOLO detector modulu
│                       # - CarDetector sinifi
│                       # - detect() metodu
│                       # - process_video_frame()
│
├── video/              # Ornek videolar
│   └── traffic_video.mp4
│
├── yolov8n.pt          # YOLOv8 nano model (otomatik indirilir)
│
├── requirements.txt    # Bagimliliklar
│
└── README.md           # Bu dosya
```

## Ayarlar ve Parametreler

### Model Boyutu

Model boyutu, hiz ve dogruluk arasinda denge kurmanizi saglar:

- **nano**: En hizli, mobil cihazlar icin uygun
- **small**: Genel kullanim, iyi denge
- **medium**: Daha yuksek dogruluk, orta hiz
- **large**: Profesyonel kullanim, yuksek dogruluk
- **xlarge**: Maksimum dogruluk, en yavas

### Confidence Threshold

- **Dusuk (0.1-0.3)**: Daha fazla tespit, daha fazla yanlis pozitif
- **Orta (0.4-0.6)**: Dengeli sonuclar
- **Yuksek (0.7-1.0)**: Sadece kesin tespitler, bazi nesneler kacirilabilir

### Frame Atlama

Video islemede performans icin frame atlayabilirsiniz:
- **0**: Tum frame'ler islenir
- **1**: Her 2. frame islenir
- **2**: Her 3. frame islenir
- vb.

## COCO Sinif Listesi (Tasitlar)

YOLO, COCO dataset'inde 80 sinif uzerinde egitilmistir. Tasit siniflari:

| ID | Sinif | Turkce |
|----|-------|--------|
| 2 | car | Araba |
| 3 | motorcycle | Motosiklet |
| 5 | bus | Otobus |
| 7 | truck | Kamyon |
| 1 | bicycle | Bisiklet |

Bu uygulamada varsayilan olarak sadece **car (araba)** sinifi tespit edilir. 
"Tum tasitlari tespit et" secenegi ile diger tasitlar da dahil edilebilir.

## Performans Ipuclari

1. **GPU Kullanin**: NVIDIA GPU varsa otomatik kullanilir (10-50x hiz artisi)
2. **Model Boyutunu Ayarlayin**: Gercek zamanli icin nano/small onerilir
3. **Frame Atlayin**: Her 2-3 frame'de bir isleyerek hiz kazanin
4. **Video Cozunurlugunu Dusurün**: Yuksek cozunurluk = daha yavas isleme

## Bilinen Sinirlamalar

- Gece veya dusuk isikli videolarda performans dusebilir
- Cok kucuk veya uzak arabalar tespit edilemeyebilir
- Kapali/gizli arabalar (diger araclar arkasinda) kacirilabilir
- Trafik sikisikligi durumlarinda cakisan kutular olabilir

## Kaynaklar

- [YOLOv8 Dokumantasyonu](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- [YOLO Paper (orijinal)](https://arxiv.org/abs/1506.02640)

## Lisans

Bu proje egitim amacli hazirlanmistir. YOLOv8 modeli Ultralytics tarafindan AGPL-3.0 lisansi ile sunulmaktadir.

---

*SAU Yapay Zeka & Bilgisayarli Goru Egitimi - Gun 3*
