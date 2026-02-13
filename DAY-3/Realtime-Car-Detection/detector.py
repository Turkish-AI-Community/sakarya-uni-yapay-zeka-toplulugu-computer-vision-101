"""
================================================================================
YOLO Car Detector - Araba Tespit Modulu
================================================================================

YOLOv8 modeli kullanarak video ve goruntulerde araba tespiti yapar.

================================================================================
YOLOV8 DETAYLI ACIKLAMA
================================================================================

YOLO (You Only Look Once) TARIHCESI:
------------------------------------
YOLO, 2016'da Joseph Redmon tarafindan ortaya konulmus devrimsel bir
nesne tespiti yaklasimidir. "Sadece bir kez bak" felsefesiyle calisir.

NEDEN "YOU ONLY LOOK ONCE"?
---------------------------
Geleneksel nesne tespiti yontemleri (R-CNN, Fast R-CNN):
1. Once "bolge oneri" (region proposal) algoritmasi calisir
2. Her bolge icin ayri ayri siniflandirma yapilir
3. Binlerce bolge = Binlerce forward pass
4. YAVAS!

YOLO yaklasimi:
1. Tum goruntu tek seferde islenir
2. Grid hucrelerinde direkt tahmin
3. Tek forward pass = Tum tespitler
4. HIZLI!

YOLOV8 ARSITEKTUR DETAYLARI:
----------------------------

1. BACKBONE (Omurga):
   CSPDarknet53 (Cross Stage Partial)
   
   Gorev: Ham piksellarden ozellik cikarma
   
   Yapi:
   - Konvolusyon katmanlari
   - Batch Normalization
   - SiLU aktivasyon (Sigmoid Linear Unit)
   - CSP bloklari (gradient akisini iyilestirir)
   
   Cikti: Farkli olceklerde feature maps
   - P3: 1/8 boyut (kucuk nesneler icin)
   - P4: 1/16 boyut (orta nesneler icin)
   - P5: 1/32 boyut (buyuk nesneler icin)

2. NECK (Boyun):
   FPN + PANet kombinasyonu
   
   FPN (Feature Pyramid Network):
   - Yukaridan asagi bilgi akisi
   - Buyuk ozellik haritalarindan kucuklere
   - Anlamsal bilgi tasir
   
   PANet (Path Aggregation Network):
   - Asagidan yukari bilgi akisi
   - Kucuk ozellik haritalarindan buyuklere
   - Konumsal bilgi tasir
   
   Sonuc: Zengin multi-scale ozellik haritalari

3. HEAD (Bas):
   Decoupled Head (Ayrilmis Bas)
   
   YOLOv8'in yeniligi: Siniflandirma ve konum ayri
   
   Eski YOLO (coupled):
   [x, y, w, h, objectness, class1, class2, ...]
   
   YOLOv8 (decoupled):
   - Siniflandirma dali: [class1, class2, ...]
   - Regresyon dali: [x, y, w, h]
   
   Avantajlari:
   - Daha hizli yakinlaşma
   - Daha iyi dogruluk
   - Anchor-free yapı

ANCHOR-FREE TESPIT:
-------------------
Eski YOLO versiyonlari "anchor box" kullaniyordu:
- Onceden tanimlanmis kutu sekilleri
- Her grid hucresi icin N anchor
- Karmasik, hyperparameter hassas

YOLOv8 anchor-free:
- Dogrudan merkez noktasi tahmini
- Merkeze gore kenar mesafeleri
- Daha basit, daha genel

COCO DATASET SINIFLARI (80 sinif):
----------------------------------
Bu uygulamada odaklanilan tasit siniflari:

Index | Sinif         | Aciklama
------|---------------|------------------
  2   | car           | Araba (otomobil)
  3   | motorcycle    | Motosiklet
  5   | bus           | Otobus
  7   | truck         | Kamyon

Diger ilginc siniflar:
  0   | person        | Insan
  1   | bicycle       | Bisiklet
 16   | dog           | Kopek
 17   | cat           | Kedi
 62   | laptop        | Dizustu bilgisayar
 67   | cell phone    | Cep telefonu

INFERENCE (CIKARIM) SURECI:
---------------------------
1. Goruntu On-Isleme:
   - Boyutlandirma: 640x640 (veya baska)
   - Normalizasyon: 0-255 -> 0-1
   - BGR -> RGB donusumu
   - NCHW formatina cevir

2. Model Forward Pass:
   - GPU'da tensor islemleri
   - Backbone -> Neck -> Head

3. Post-Processing:
   - Confidence filtreleme
   - Non-Maximum Suppression (NMS)
   - Koordinat donusumu

NON-MAXIMUM SUPPRESSION (NMS):
------------------------------
Ayni nesne icin birden fazla tespit olabilir.
NMS bunlari tek tespite indirger.

Algoritma:
1. Tespitleri confidence'a gore sirala
2. En yuksek confidence'li tespiti sec
3. Bu tespitle IoU > threshold olan diger tespitleri at
4. Kalan tespitler icin 2-3'u tekrarla

IoU (Intersection over Union):
IoU = Kesisim Alani / Birlesim Alani

IoU = 1.0 -> Tamamen ust uste
IoU = 0.0 -> Hic kesisim yok
IoU > 0.5 -> Genellikle ayni nesne kabul edilir

Yazar: SAU Yapay Zeka & Bilgisayarli Goru Toplulugu
================================================================================
"""

# =============================================================================
# KUTUPHANELERIN ICE AKTARILMASI
# =============================================================================

# OpenCV: Bilgisayarli goru kutuphanesi
# - Goruntu/video okuma
# - Goruntu uzerinde cizim
# - Renk donusumleri
import cv2

# NumPy: Sayisal hesaplama
# - Goruntu verileri icin array islemleri
import numpy as np

# Ultralytics: YOLOv8 resmi kutuphanesi
# - Model yukleme
# - Inference (cikarim)
# - Sonuc isleme
from ultralytics import YOLO

# Typing: Python tip ipuclari
# - Kod okunabilirligini arttirir
# - IDE destegi saglar
from typing import Tuple, List, Optional

# PyTorch: Derin ogrenme framework'u
# - GPU/CPU cihaz yonetimi
# - Tensor islemleri
import torch


# =============================================================================
# COCO DATASET TASIT SINIFLARI
# =============================================================================
# YOLO, COCO dataset'i ile egitilmistir.
# Asagida tasit olarak kabul edilen siniflar tanimlanmistir.

VEHICLE_CLASSES = {
    2: "car",        # Araba (otomobil, sedan, SUV, vb.)
    3: "motorcycle", # Motosiklet
    5: "bus",        # Otobus (sehir icı, otokar, vb.)
    7: "truck"       # Kamyon (TIR, pikap, vb.)
}

# Sadece araba tespiti icin sinif ID'si
# Bu uygulamanin varsayilan modu sadece araba tespitidir
CAR_CLASS_ID = 2


# =============================================================================
# YARDIMCI FONKSIYONLAR
# =============================================================================

def get_device_info() -> dict:
    """
    Sistem GPU/CPU bilgilerini dondurur.
    
    Bu fonksiyon, kullaniciya hangi donanim uzerinde calistiklarini
    gostermek ve uygun yapılandırmayı secmelerine yardımcı olmak icin kullanilir.
    
    GPU VS CPU KARSILASTIRMASI:
    ---------------------------
    Ozellik     | CPU              | GPU (CUDA)
    ------------|------------------|------------------
    Cekirdek    | 4-32             | 1000-10000+
    Islem Tipi  | Genel amacli     | Paralel (SIMD)
    Bant Genisligi | Dusuk         | Cok yuksek
    YOLO Hizi   | 5-30 FPS         | 60-200+ FPS
    
    CUDA NEDIR?
    -----------
    CUDA (Compute Unified Device Architecture), NVIDIA'nin GPU'lari
    genel amacli hesaplama icin programlamaya olanak taniyan platformudur.
    
    PyTorch CUDA destegi:
    - torch.cuda.is_available(): CUDA kullanilabilir mi?
    - torch.cuda.get_device_name(0): GPU adi
    - model.to("cuda"): Modeli GPU'ya tasi
    
    DONDURUR:
    ---------
    dict:
        - cuda_available: bool - CUDA kullanilabilir mi?
        - device: str - "cuda" veya "cpu"
        - gpu_name: str veya None - GPU model adi (orn: "NVIDIA GeForce RTX 3090")
        - gpu_memory: str veya None - GPU bellek miktari (orn: "24.0 GB")
    """
    
    # CUDA kontrolu
    cuda_available = torch.cuda.is_available()
    
    info = {
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "gpu_name": None,
        "gpu_memory": None
    }
    
    if cuda_available:
        # GPU adi
        # get_device_name(0): Ilk GPU'nun adi
        # Birden fazla GPU varsa 0, 1, 2, ... ile erisilebilir
        info["gpu_name"] = torch.cuda.get_device_name(0)
        
        # GPU bellek miktari
        # get_device_properties: GPU ozelliklerini dondurur
        # total_memory: Toplam bellek (bytes cinsinden)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["gpu_memory"] = f"{memory_gb:.1f} GB"
    
    return info


# =============================================================================
# CARDETECTOR SINIFI
# =============================================================================

class CarDetector:
    """
    YOLOv8 ile araba/tasit tespiti yapan sinif.
    
    Bu sinif, YOLO modelini yukler ve video frame'lerinde veya
    tek goruntulerde araba tespiti yapar.
    
    SINIF OZELLIKLERI (Attributes):
    --------------------------------
    model : YOLO
        Yuklu YOLOv8 modeli (Ultralytics)
        
    device : str
        Kullanilan cihaz ("cuda" veya "cpu")
        
    confidence_threshold : float
        Minimum kabul edilen guven skoru (0-1)
        
    detect_all_vehicles : bool
        True: Tum tasitlari tespit et
        False: Sadece araba (car) sinifini tespit et
    
    TEMEL METODLAR:
    ---------------
    detect(frame, draw_boxes) -> (frame, detections)
        Tek frame'de tespit yap
        
    process_video_frame(frame, frame_count, show_info) -> (frame, count)
        Video frame'ini isle, bilgi overlay'i ekle
    
    KULLANIM ORNEGI:
    ----------------
    >>> detector = CarDetector(model_name="yolov8n.pt", confidence_threshold=0.5)
    >>> frame = cv2.imread("trafik.jpg")
    >>> result_frame, detections = detector.detect(frame)
    >>> print(f"Tespit edilen araba sayisi: {len(detections)}")
    """
    
    def __init__(
        self, 
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        detect_all_vehicles: bool = False
    ):
        """
        CarDetector'u baslatir ve YOLO modelini yukler.
        
        MODEL BOYUTLARI VE PERFORMANS:
        ------------------------------
        Model      | Boyut   | mAP   | GPU Hizi | CPU Hizi
        -----------|---------|-------|----------|----------
        yolov8n.pt | 6.3 MB  | 37.3  | 0.6 ms   | 78 ms
        yolov8s.pt | 22.5 MB | 44.9  | 0.9 ms   | 128 ms
        yolov8m.pt | 52.0 MB | 50.2  | 1.7 ms   | 234 ms
        yolov8l.pt | 87.7 MB | 52.9  | 2.9 ms   | 375 ms
        yolov8x.pt | 136.7MB | 53.9  | 4.7 ms   | 479 ms
        
        mAP: Mean Average Precision (COCO val2017)
        Hiz: NVIDIA V100 GPU ve Intel CPU (batch=1, image=640x640)
        
        MODEL INDIRME:
        --------------
        Model dosyasi bulunamazsa otomatik indirilir.
        Ilk calistirmada internet baglantisi gerekir.
        Indirilen model ~/.cache/ultralytics/ altinda saklanir.
        
        PARAMETRELER:
        -------------
        model_name : str
            YOLO model dosya adi veya yolu.
            - "yolov8n.pt": Nano (en hizli, en dusuk dogruluk)
            - "yolov8s.pt": Small (hizli, iyi dogruluk)
            - "yolov8m.pt": Medium (dengeli)
            - "yolov8l.pt": Large (yavas, yuksek dogruluk)
            - "yolov8x.pt": XLarge (en yavas, en yuksek dogruluk)
            
        confidence_threshold : float
            Minimum kabul edilen guven skoru.
            - Dusuk (0.1-0.3): Daha fazla tespit, daha fazla yanlis pozitif
            - Orta (0.4-0.6): Dengeli
            - Yuksek (0.7-0.9): Daha az tespit, daha kesin
            
        detect_all_vehicles : bool
            - True: Araba, kamyon, otobus, motosiklet tespit et
            - False: Sadece araba (car) tespit et
        """
        
        # Parametreleri kaydet
        self.confidence_threshold = confidence_threshold
        self.detect_all_vehicles = detect_all_vehicles
        
        # -----------------------------------------
        # Cihaz Secimi
        # -----------------------------------------
        # CUDA (GPU) varsa kullan, yoksa CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # -----------------------------------------
        # Model Yukleme
        # -----------------------------------------
        print(f"[INFO] YOLO modeli yukleniyor: {model_name}")
        print(f"[INFO] Cihaz: {self.device.upper()}")
        
        # YOLO() constructor'i:
        # - Dosya varsa yukler
        # - Yoksa Ultralytics sunucularindan indirir
        # - Model agirliklarini parse eder
        self.model = YOLO(model_name)
        
        # Modeli secilen cihaza tasi
        # GPU'ya tasimak, tensor islemlerinin GPU'da yapilmasini saglar
        self.model.to(self.device)
        
        print(f"[INFO] Model basariyla yuklendi!")
    
    def detect(
        self, 
        frame: np.ndarray,
        draw_boxes: bool = True
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Tek bir frame'de araba/tasit tespiti yapar.
        
        TESPIT SURECI:
        --------------
        1. On-Isleme (Preprocessing):
           - Goruntu boyutlandirma (ornegin 640x640)
           - Normalizasyon (0-255 -> 0-1)
           - BGR -> RGB donusumu
           - Batch dimension ekleme
        
        2. Model Inference:
           - GPU/CPU'da forward pass
           - Tahminler: [x_center, y_center, width, height, confidence, class_probs]
        
        3. Post-Processing:
           - Confidence filtreleme
           - Non-Maximum Suppression (NMS)
           - Koordinat donusumu (normalize -> piksel)
        
        4. Sinif Filtreleme:
           - Sadece istenen siniflari tut
           - Araba veya tum tasitlar
        
        5. Gorsellestime (opsiyonel):
           - Bounding box cizimi
           - Etiket ve confidence yazdirma
        
        BOUNDING BOX FORMATI:
        ---------------------
        YOLO ciktisi: [x_center, y_center, width, height]
        OpenCV cizim: [x1, y1, x2, y2] (sol-ust, sag-alt)
        
        Donusum:
        x1 = x_center - width/2
        y1 = y_center - height/2
        x2 = x_center + width/2
        y2 = y_center + height/2
        
        PARAMETRELER:
        -------------
        frame : np.ndarray
            BGR formatinda OpenCV goruntusu.
            Shape: (height, width, 3)
            
        draw_boxes : bool
            True: Tespit kutularini ciz
            False: Sadece tespit yap, cizim yapma
        
        DONDURUR:
        ---------
        Tuple[np.ndarray, List[dict]]
            - processed_frame: Islenmis goruntu (bounding box'larla veya olmadan)
            - detections: Tespit listesi, her biri dict:
                - class_id: int - COCO sinif ID'si
                - class_name: str - Sinif adi ("car", "truck", vb.)
                - confidence: float - Guven skoru (0-1)
                - bbox: Tuple[int, int, int, int] - (x1, y1, x2, y2)
        """
        
        # -----------------------------------------
        # YOLO Inference
        # -----------------------------------------
        # model() metodu inference yapar
        # conf: Minimum confidence threshold
        # verbose=False: Konsol ciktisini sustur
        # 
        # Donus: List[Results] - her goruntu icin bir Results nesnesi
        # [0]: Ilk (ve tek) goruntu icin sonuclar
        results = self.model(
            frame, 
            conf=self.confidence_threshold,
            verbose=False
        )[0]
        
        # Tespit listesi ve cikti frame'i
        detections = []
        output_frame = frame.copy() if draw_boxes else frame
        
        # -----------------------------------------
        # Tespitleri Isle
        # -----------------------------------------
        # results.boxes: Tespit edilen nesnelerin kutulari
        # Her box icin: xyxy, conf, cls, vb.
        boxes = results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # -----------------------------------------
                # Sinif Filtreleme
                # -----------------------------------------
                # box.cls: Sinif ID'si tensor'u
                # [0]: Ilk (ve tek) deger
                class_id = int(box.cls[0])
                
                # Istenen sinif mi kontrol et
                if self.detect_all_vehicles:
                    # Tum tasitlar modu
                    if class_id not in VEHICLE_CLASSES:
                        continue  # Tasit degilse atla
                    class_name = VEHICLE_CLASSES[class_id]
                else:
                    # Sadece araba modu
                    if class_id != CAR_CLASS_ID:
                        continue  # Araba degilse atla
                    class_name = "car"
                
                # -----------------------------------------
                # Bounding Box Koordinatlari
                # -----------------------------------------
                # box.xyxy: [x1, y1, x2, y2] formatinda
                # [0]: Batch boyutunu kaldir
                # map(int, ...): Float -> int donusumu
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # -----------------------------------------
                # Confidence Skoru
                # -----------------------------------------
                # box.conf: Guven skoru tensor'u
                # [0]: Ilk (ve tek) deger
                # float(): Tensor -> Python float
                confidence = float(box.conf[0])
                
                # -----------------------------------------
                # Tespit Bilgisini Kaydet
                # -----------------------------------------
                detection = {
                    "class_id": class_id,       # COCO sinif ID'si
                    "class_name": class_name,   # Sinif adi
                    "confidence": confidence,   # Guven skoru
                    "bbox": (x1, y1, x2, y2)   # Kutu koordinatlari
                }
                detections.append(detection)
                
                # -----------------------------------------
                # Gorsellestime
                # -----------------------------------------
                if draw_boxes:
                    # Sinifa gore renk sec
                    color = self._get_class_color(class_id)
                    
                    # Bounding Box Ciz
                    # rectangle(goruntu, sol_ust, sag_alt, renk, kalinlik)
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Etiket Arka Plani
                    # Etiketin okunabilmesi icin siyah arka plan
                    label = f"{class_name}: {confidence:.0%}"
                    
                    # Metin boyutunu hesapla
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6,  # Font olcegi
                        2     # Kalinlik
                    )
                    
                    # Etiket arka plan dikdortgeni
                    cv2.rectangle(
                        output_frame, 
                        (x1, y1 - label_h - 10),           # Sol-ust
                        (x1 + label_w + 10, y1),           # Sag-alt
                        color,                              # Renk
                        -1                                  # Icini doldur
                    )
                    
                    # Etiket Metni
                    cv2.putText(
                        output_frame, 
                        label, 
                        (x1 + 5, y1 - 5),                  # Konum
                        cv2.FONT_HERSHEY_SIMPLEX,          # Font
                        0.6,                                # Olcek
                        (255, 255, 255),                   # Beyaz renk
                        2                                   # Kalinlik
                    )
        
        return output_frame, detections
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        Sinif ID'sine gore renk dondurur.
        
        Her tasit sinifi icin farkli bir renk kullanmak,
        gorsellestirmeyi daha anlasilir yapar.
        
        RENK PALETI:
        ------------
        Sinif       | Renk      | BGR Degeri
        ------------|-----------|---------------
        Araba       | Yesil     | (0, 255, 0)
        Motosiklet  | Turuncu   | (255, 165, 0)
        Otobus      | Mavi      | (255, 0, 0)
        Kamyon      | Kirmizi   | (0, 0, 255)
        
        NOT: OpenCV BGR (Mavi-Yesil-Kirmizi) sirasi kullanir,
        RGB (Kirmizi-Yesil-Mavi) degil!
        
        PARAMETRELER:
        -------------
        class_id : int
            COCO sinif ID'si
        
        DONDURUR:
        ---------
        Tuple[int, int, int]
            BGR formatinda renk degeri
        """
        colors = {
            2: (0, 255, 0),     # Araba - Yesil
            3: (255, 165, 0),   # Motosiklet - Turuncu
            5: (255, 0, 0),     # Otobus - Mavi
            7: (0, 0, 255)      # Kamyon - Kirmizi
        }
        
        # Tanimli degilse varsayilan yesil
        return colors.get(class_id, (0, 255, 0))
    
    def process_video_frame(
        self, 
        frame: np.ndarray,
        frame_count: int,
        show_info: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Video frame'ini isle ve bilgi overlay'i ekle.
        
        Bu metod, detect() metoduna ek olarak:
        - Frame numarasi
        - Tespit sayisi
        gibi bilgileri goruntu uzerine ekler.
        
        VIDEO ISLEME PIPELINE'I:
        ------------------------
        1. Tespit yap (detect metodu)
        2. Bilgi kutusu olustur
        3. Frame ve tespit sayisini ciz
        4. Sonucu dondur
        
        PARAMETRELER:
        -------------
        frame : np.ndarray
            BGR formatinda video frame'i
            
        frame_count : int
            Suanki frame numarasi (gosterim icin)
            
        show_info : bool
            True: Bilgi overlay'i ekle
            False: Sadece tespit kutularini ciz
        
        DONDURUR:
        ---------
        Tuple[np.ndarray, int]
            - processed_frame: Islenmis goruntu
            - detection_count: Bu frame'deki tespit sayisi
        """
        
        # Tespit yap
        output_frame, detections = self.detect(frame)
        
        # Bilgi overlay'i ekle
        if show_info:
            # Tespit sayisi
            car_count = len(detections)
            
            # -----------------------------------------
            # Bilgi Kutusu Arka Plani
            # -----------------------------------------
            # Siyah yarı saydam dikdortgen
            cv2.rectangle(
                output_frame, 
                (10, 10),       # Sol-ust
                (250, 80),      # Sag-alt
                (0, 0, 0),      # Siyah
                -1              # Icini doldur
            )
            
            # Yesil cerceve
            cv2.rectangle(
                output_frame, 
                (10, 10), 
                (250, 80), 
                (0, 255, 0),    # Yesil
                2               # Cerceve kalinligi
            )
            
            # -----------------------------------------
            # Frame Bilgisi
            # -----------------------------------------
            cv2.putText(
                output_frame, 
                f"Frame: {frame_count}", 
                (20, 35),                      # Konum
                cv2.FONT_HERSHEY_SIMPLEX,      # Font
                0.6,                           # Olcek
                (255, 255, 255),               # Beyaz
                2                              # Kalinlik
            )
            
            # -----------------------------------------
            # Tespit Sayisi
            # -----------------------------------------
            cv2.putText(
                output_frame, 
                f"Araba Sayisi: {car_count}", 
                (20, 65),                      # Konum
                cv2.FONT_HERSHEY_SIMPLEX,      # Font
                0.6,                           # Olcek
                (0, 255, 0),                   # Yesil
                2                              # Kalinlik
            )
        
        return output_frame, len(detections)


# =============================================================================
# FACTORY FONKSIYONU
# =============================================================================

def create_detector(
    model_size: str = "nano",
    confidence: float = 0.5,
    detect_all_vehicles: bool = False
) -> CarDetector:
    """
    Kolayca CarDetector instance'i olusturur.
    
    Bu fonksiyon, model boyutu isminden (nano, small, vb.)
    otomatik olarak dogru model dosyasini (.pt) belirler.
    
    FACTORY PATTERN:
    ----------------
    Factory fonksiyonu, nesne olusturma mantıgını merkezi bir yere toplar.
    Avantajlari:
    - Kullanici model dosya isimlerini bilmek zorunda degil
    - Varsayilan degerler merkezi olarak yonetilir
    - Gelecekte farkli detector turleri eklenebilir
    
    MODEL BOYUTU -> DOSYA ESLESMESI:
    --------------------------------
    "nano"    -> "yolov8n.pt"
    "small"   -> "yolov8s.pt"
    "medium"  -> "yolov8m.pt"
    "large"   -> "yolov8l.pt"
    "xlarge"  -> "yolov8x.pt"
    
    PARAMETRELER:
    -------------
    model_size : str
        Model boyutu: "nano", "small", "medium", "large", "xlarge"
        
    confidence : float
        Confidence threshold (0.0 - 1.0)
        
    detect_all_vehicles : bool
        True: Tum tasitlar
        False: Sadece araba
    
    DONDURUR:
    ---------
    CarDetector
        Yapilandirilmis detector instance'i
    
    ORNEK:
    ------
    >>> detector = create_detector(model_size="small", confidence=0.6)
    >>> frame = cv2.imread("trafik.jpg")
    >>> result, detections = detector.detect(frame)
    """
    
    # Model boyutu -> dosya adi eslesmesi
    model_map = {
        "nano": "yolov8n.pt",
        "small": "yolov8s.pt", 
        "medium": "yolov8m.pt",
        "large": "yolov8l.pt",
        "xlarge": "yolov8x.pt"
    }
    
    # Gecersiz boyut icin varsayilan: nano
    model_name = model_map.get(model_size, "yolov8n.pt")
    
    # CarDetector instance'i olustur ve dondur
    return CarDetector(
        model_name=model_name,
        confidence_threshold=confidence,
        detect_all_vehicles=detect_all_vehicles
    )


# =============================================================================
# TEST / ORNEK KULLANIM
# =============================================================================
# Bu dosya dogrudan calistirildiginda test kodu calisir

if __name__ == "__main__":
    # Baslık
    print("\n" + "="*50)
    print("YOLO Car Detector Test")
    print("="*50)
    
    # -----------------------------------------
    # Sistem Bilgisi
    # -----------------------------------------
    device_info = get_device_info()
    print(f"\nCihaz: {device_info['device'].upper()}")
    
    if device_info['cuda_available']:
        print(f"GPU: {device_info['gpu_name']}")
        print(f"Bellek: {device_info['gpu_memory']}")
    
    # -----------------------------------------
    # Detector Olustur
    # -----------------------------------------
    print("\n[TEST] Detector olusturuluyor...")
    detector = create_detector(model_size="nano", confidence=0.5)
    
    # -----------------------------------------
    # Test Goruntusu Olustur
    # -----------------------------------------
    # Gri renkli bos goruntu (gercek test icin gercek goruntu kullanin)
    print("[TEST] Test goruntusu olusturuluyor...")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (100, 100, 100)  # Gri
    
    # -----------------------------------------
    # Tespit Dene
    # -----------------------------------------
    print("[TEST] Tespit yapiliyor...")
    output, detections = detector.detect(test_frame)
    
    print(f"[TEST] Tespit sayisi: {len(detections)}")
    print("\n[SUCCESS] Test tamamlandi!")
