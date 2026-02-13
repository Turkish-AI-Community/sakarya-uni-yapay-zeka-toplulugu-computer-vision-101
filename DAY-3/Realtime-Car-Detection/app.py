"""
================================================================================
Realtime Car Detection - Streamlit Web Arayuzu
================================================================================

YOLOv8 modeli kullanarak videolarda gercek zamanli araba tespiti yapan
web uygulamasi.

================================================================================
YOLO NEDIR?
================================================================================

YOLO (You Only Look Once):
--------------------------
Gercek zamanli nesne tespiti icin tasarlanmis bir derin ogrenme ailesidir.

TARIHCE:
--------
- YOLOv1 (2016): Joseph Redmon - Ilk versiyon
- YOLOv2 (2017): Batch normalization, anchor boxes
- YOLOv3 (2018): Multi-scale detection
- YOLOv4 (2020): CSPDarknet, PANet
- YOLOv5 (2020): Ultralytics - PyTorch tabanlı
- YOLOv6 (2022): Meituan
- YOLOv7 (2022): E-ELAN, model reparameterization
- YOLOv8 (2023): Ultralytics - En guncel versiyon

YOLO'NUN FARKI:
---------------
Geleneksel Yontem (R-CNN ailesi):
1. Bolge oneri (Region Proposal)
2. Her bolge icin siniflandirma
3. Yavas, cok adimli

YOLO Yontemi:
1. Tek ileri gecis (forward pass)
2. Goruntuyu grid'e bol
3. Her grid hucresi nesne ve konum tahmin eder
4. Cok hizli, gercek zamanli

YOLOV8 ARSITEKTURU:
-------------------
1. Backbone (Omurga):
   - CSPDarknet53
   - Ozellikleri cikarir
   - Farkli olceklerde feature maps

2. Neck (Boyun):
   - FPN (Feature Pyramid Network)
   - PANet (Path Aggregation Network)
   - Multi-scale feature fusion

3. Head (Bas):
   - Decoupled head (ayrilmis)
   - Siniflandirma ve konum ayri
   - Anchor-free detection

COCO DATASET:
-------------
YOLO, COCO (Common Objects in Context) dataset'i ile egitilmistir.
- 80 nesne sinifi
- 330K+ goruntu
- 1.5M+ nesne etiketi

Arac siniflar:
- car (2): Araba
- motorcycle (3): Motosiklet
- bus (5): Otobus
- truck (7): Kamyon

KULLANIM:
---------
Terminal'de calistirmak icin:
    $ streamlit run app.py

Tarayicida http://localhost:8501 adresinde acilir.

GEREKSINIMLER:
--------------
- Python 3.8+
- streamlit
- opencv-python
- ultralytics (YOLO)
- torch

Yazar: SAU Yapay Zeka & Bilgisayarli Goru Toplulugu
================================================================================
"""

# =============================================================================
# KUTUPHANELERIN ICE AKTARILMASI
# =============================================================================

# Streamlit: Interaktif web uygulamalari framework'u
# - Hizli prototipleme
# - Otomatik UI olusturma
# - Canli video gosterimi
import streamlit as st

# OpenCV: Bilgisayarli goru kutuphanesi
# - Video okuma ve isleme
# - Frame manipulasyonu
# - Renk donusumleri
import cv2

# NumPy: Sayisal hesaplama
# - Goruntu verileri icin array islemleri
import numpy as np

# tempfile: Gecici dosya olusturma
# - Yuklenen videoyu gecici dosyaya kaydetme
import tempfile

# time: Zaman islemleri
# - FPS hesaplama
# - Performans olcumu
import time

# pathlib: Dosya yolu islemleri
# - Cross-platform path handling
from pathlib import Path

# =============================================================================
# YEREL MODUL IMPORTLARI
# =============================================================================
# detector.py dosyasindan YOLO detector sinifi ve yardimci fonksiyonlar

from detector import (
    CarDetector,      # Ana YOLO detector sinifi
    get_device_info,  # GPU/CPU bilgisi
    create_detector   # Factory fonksiyonu
)


# =============================================================================
# STREAMLIT SAYFA YAPILANDIRMASI
# =============================================================================
# set_page_config: Sayfa ayarlarini belirler
# Streamlit scriptinin EN BASINDA cagirilmalidir

st.set_page_config(
    page_title="Realtime Car Detection",  # Tarayici sekmesinde gorunur
    page_icon="🚗",                        # Emoji veya dosya yolu
    layout="wide"                           # Genis ekran modu
)


# =============================================================================
# OZEL CSS STILLERI
# =============================================================================
# Streamlit'in gorunumunu ozellestirmek icin CSS

st.markdown("""
<style>
    /* =============================================
       Ana Baslik Stili
       =============================================
       Koyu tema gradient ile modern gorunum
    */
    .main-header {
        text-align: center;
        padding: 1.5rem;
        /* Koyu mavi-yesil gradient */
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        /* Hafif golge efekti */
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Baslik metni */
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        /* Metin golgesi */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Alt baslik */
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* =============================================
       Istatistik Karti Stili
       =============================================
       FPS, tespit sayisi gibi metrikleri gostermek icin
    */
    .stat-card {
        /* Koyu gradient arka plan */
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    /* Buyuk sayi gosterimi */
    .stat-card h3 {
        margin: 0;
        font-size: 2rem;
        color: #00ff88;  /* Neon yesil */
    }
    
    /* Aciklama metni */
    .stat-card p {
        margin: 0.3rem 0 0 0;
        opacity: 0.8;
        font-size: 0.9rem;
    }
    
    /* =============================================
       Bilgi Kutusu Stili
       =============================================
       Aciklama ve yardim metinleri icin
    */
    .info-box {
        background: #f0f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2c5364;  /* Sol kenarda vurgu */
        margin: 1rem 0;
    }
    
    /* Basari mesaji kutusu */
    .success-box {
        background: #d4edda;  /* Acik yesil */
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;  /* Yesil vurgu */
        margin: 1rem 0;
    }
    
    /* =============================================
       Video Konteyner Stili
       =============================================
       Video oynatici cercevesi
    */
    .video-container {
        border: 3px solid #2c5364;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* =============================================
       Streamlit Buton Stilleri
       =============================================
       Varsayilan butonlari ozellestirme
    */
    .stButton > button {
        background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;  /* Hover animasyonu */
    }
    
    /* Hover durumu */
    .stButton > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE BASLANGICI
# =============================================================================
# Session state, sayfa yenilendiginde bile korunan degiskenler icin kullanilir
# Streamlit her interaksiyonda tum scripti yeniden calistirdigindan,
# kalici durum icin session_state kullanmak gerekir

if "detector" not in st.session_state:
    # YOLO detector nesnesi
    st.session_state.detector = None
    
if "is_detecting" not in st.session_state:
    # Tespit islemi aktif mi?
    st.session_state.is_detecting = False
    
if "total_cars" not in st.session_state:
    # Toplam tespit edilen araba sayisi
    st.session_state.total_cars = 0
    
if "frames_processed" not in st.session_state:
    # Islenen frame sayisi
    st.session_state.frames_processed = 0


# =============================================================================
# MODEL YUKLEME (CACHE ILE)
# =============================================================================
# @st.cache_resource: Sonucu cache'ler, ayni parametrelerle tekrar yuklenmez
# Bu ozellikle buyuk modeller icin bellek ve zaman tasarrufu saglar

@st.cache_resource
def load_detector(model_size: str, confidence: float, detect_all: bool):
    """
    YOLO detectorunu yukler ve cache'ler.
    
    Cache Mantigi:
    - Ayni parametrelerle cagirilirsa cache'ten doner
    - Farkli parametrelerle cagirilirsa yeni model yukler
    
    PARAMETRELER:
    -------------
    model_size : str
        Model boyutu: "nano", "small", "medium", "large", "xlarge"
        
    confidence : float
        Minimum confidence threshold (0.0 - 1.0)
        
    detect_all : bool
        True: Tum tasitlari tespit et (araba, kamyon, otobus, motor)
        False: Sadece araba (car) sinifini tespit et
    
    DONDURUR:
    ---------
    CarDetector
        Yuklu YOLO detector instance'i
    """
    return create_detector(
        model_size=model_size,
        confidence=confidence,
        detect_all_vehicles=detect_all
    )


# =============================================================================
# ANA BASLIK
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🚗 Realtime Car Detection</h1>
    <p>YOLOv8 ile Gercek Zamanli Araba Tespiti</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# YAN PANEL (SIDEBAR) - AYARLAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # -----------------------------------------
    # Model Ayarlari
    # -----------------------------------------
    st.subheader("Model Ayarlari")
    
    # Model boyutu secimi
    # Buyuk modeller daha dogru ama daha yavas
    model_size = st.selectbox(
        "Model Boyutu:",
        ["nano", "small", "medium", "large", "xlarge"],
        index=0,  # Varsayilan: nano
        help="Buyuk modeller daha dogru ama daha yavas"
    )
    
    # Model boyutu aciklamalari
    model_info = {
        "nano": "En hizli, dusuk dogruluk",
        "small": "Hizli, iyi dogruluk",
        "medium": "Dengeli hiz/dogruluk",
        "large": "Yuksek dogruluk, yavas",
        "xlarge": "En yuksek dogruluk, en yavas"
    }
    st.caption(f"ℹ️ {model_info[model_size]}")
    
    st.divider()
    
    # -----------------------------------------
    # Confidence Threshold
    # -----------------------------------------
    # Bu deger, tespitin kabul edilmesi icin gereken minimum guven skoru
    confidence = st.slider(
        "Confidence Threshold:",
        min_value=0.1,   # Minimum deger
        max_value=1.0,   # Maximum deger
        value=0.5,       # Varsayilan
        step=0.05,       # Artis miktari
        help="Dusuk deger = daha fazla tespit, yuksek deger = daha kesin tespit"
    )
    
    st.divider()
    
    # -----------------------------------------
    # Tespit Modu
    # -----------------------------------------
    # Sadece araba mi yoksa tum tasitlar mi tespit edilecek
    detect_all = st.checkbox(
        "Tum tasitlari tespit et",
        value=False,
        help="Isaretlenirse araba, kamyon, otobus, motosiklet tespit edilir"
    )
    
    st.divider()
    
    # -----------------------------------------
    # Video Isleme Ayarlari
    # -----------------------------------------
    st.subheader("Video Ayarlari")
    
    # Frame atlama - performans optimizasyonu
    # Her N. frame'de bir tespit yaparak hizi artirir
    skip_frames = st.slider(
        "Frame Atlama:",
        min_value=0,
        max_value=5,
        value=0,  # 0 = atlama yok
        help="Performans icin frame atlayabilirsiniz (0 = atlama yok)"
    )
    
    # FPS gosterim secenegi
    show_fps = st.checkbox("FPS Goster", value=True)
    
    # Bounding box gosterim secenegi
    show_boxes = st.checkbox("Bounding Box Goster", value=True)
    
    st.divider()
    
    # -----------------------------------------
    # Sistem Bilgisi
    # -----------------------------------------
    st.subheader("Sistem Bilgisi")
    
    device_info = get_device_info()
    
    if device_info['cuda_available']:
        # GPU mevcut
        st.success(f"✅ GPU: {device_info['gpu_name']}")
        st.caption(f"Bellek: {device_info['gpu_memory']}")
    else:
        # Sadece CPU
        st.warning("⚠️ CPU modunda calisiliyor")
        st.caption("GPU ile daha hizli olur")
    
    st.divider()
    st.caption("SAU Yapay Zeka & Bilgisayarli Goru")


# =============================================================================
# MODEL YUKLEME
# =============================================================================
# Secilen ayarlarla YOLO modelini yukle

with st.spinner("🔄 YOLO modeli yukleniyor..."):
    detector = load_detector(model_size, confidence, detect_all)
    st.session_state.detector = detector


# =============================================================================
# ANA ICERIK - VIDEO YUKLEME
# =============================================================================

st.subheader("📹 Video Yukle")

# Video dosyasi yukleme widget'i
video_file = st.file_uploader(
    "Analiz edilecek videoyu secin",
    type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
    help="Desteklenen formatlar: MP4, AVI, MOV, MKV, WEBM"
)


# =============================================================================
# VIDEO ANALIZI
# =============================================================================

if video_file is not None:
    # -----------------------------------------
    # Video Dosyasini Gecici Dosyaya Kaydet
    # -----------------------------------------
    # Streamlit file_uploader BytesIO dondurur
    # OpenCV dosya yolu bekler, bu yuzden gecici dosyaya kaydediyoruz
    
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name
    
    # -----------------------------------------
    # Video Bilgilerini Al
    # -----------------------------------------
    # OpenCV VideoCapture ile video meta verilerini oku
    
    cap = cv2.VideoCapture(video_path)
    
    # Frame sayisi
    # CAP_PROP_FRAME_COUNT: Videodaki toplam frame sayisi
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # FPS (Frames Per Second)
    # CAP_PROP_FPS: Videonun saniyedeki frame sayisi
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video boyutlari
    # CAP_PROP_FRAME_WIDTH: Frame genisligi (piksel)
    # CAP_PROP_FRAME_HEIGHT: Frame yuksekligi (piksel)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video suresi (saniye)
    duration = total_frames / fps if fps > 0 else 0
    
    # VideoCapture'u kapat (sonra tekrar acacagiz)
    cap.release()
    
    # -----------------------------------------
    # Video Bilgilerini Goster
    # -----------------------------------------
    st.markdown("### Video Bilgileri")
    
    # 4 sutunlu istatistik kartlari
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{total_frames}</h3>
            <p>Toplam Frame</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{fps}</h3>
            <p>FPS</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{width}x{height}</h3>
            <p>Cozunurluk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{duration:.1f}s</h3>
            <p>Sure</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # -----------------------------------------
    # Tespit Baslat Butonu
    # -----------------------------------------
    col_btn1, col_btn2 = st.columns([1, 4])
    
    with col_btn1:
        start_detection = st.button(
            "🚀 Tespiti Baslat", 
            type="primary", 
            use_container_width=True
        )
    
    with col_btn2:
        st.caption("Video islenerek arabalar tespit edilecek")
    
    # -----------------------------------------
    # Video Isleme Dongusu
    # -----------------------------------------
    if start_detection:
        # Session state'i guncelle
        st.session_state.is_detecting = True
        st.session_state.total_cars = 0
        st.session_state.frames_processed = 0
        
        # Layout: Video (3/4) + Istatistikler (1/4)
        video_col, stats_col = st.columns([3, 1])
        
        with video_col:
            st.markdown("### 🎬 Gercek Zamanli Tespit")
            # Placeholder: Dinamik olarak guncellenecek alan
            video_placeholder = st.empty()
        
        with stats_col:
            st.markdown("### 📊 Canli Istatistikler")
            stats_placeholder = st.empty()
            progress_placeholder = st.empty()
        
        # -----------------------------------------
        # Video Frame'lerini Isle
        # -----------------------------------------
        
        # Videoyu tekrar ac
        cap = cv2.VideoCapture(video_path)
        
        # Sayaclar ve zamanlayicilar
        frame_count = 0          # Islenen frame sayisi
        total_detections = 0     # Toplam tespit sayisi
        start_time = time.time() # Baslangic zamani
        
        # Her frame'i isle
        while cap.isOpened():
            # Frame oku
            # ret: Basarili mi? (True/False)
            # frame: BGR formatinda goruntu (numpy array)
            ret, frame = cap.read()
            
            # Video bittiyse cik
            if not ret:
                break
            
            frame_count += 1
            
            # -----------------------------------------
            # Frame Atlama (Performans Optimizasyonu)
            # -----------------------------------------
            # skip_frames > 0 ise her N. frame'de bir isle
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue
            
            # -----------------------------------------
            # YOLO Tespiti
            # -----------------------------------------
            if show_boxes:
                # Bounding box'larla birlikte tespit
                processed_frame, detections = detector.detect(frame, draw_boxes=True)
            else:
                # Sadece tespit (cizim yok)
                processed_frame, detections = detector.detect(frame, draw_boxes=False)
            
            # Tespit sayisini guncelle
            total_detections += len(detections)
            
            # -----------------------------------------
            # FPS Hesaplama
            # -----------------------------------------
            # Gecen sure ve islenen frame sayisindan FPS hesapla
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # -----------------------------------------
            # FPS Overlay Ekleme
            # -----------------------------------------
            if show_fps:
                # Siyah dikdortgen arka plan (okunabirlik icin)
                cv2.rectangle(
                    processed_frame, 
                    (width - 150, 10),      # Sol-ust kose
                    (width - 10, 50),       # Sag-alt kose
                    (0, 0, 0),              # Siyah renk
                    -1                       # Icini doldur
                )
                
                # FPS metni
                cv2.putText(
                    processed_frame,
                    f"FPS: {current_fps:.1f}",
                    (width - 140, 38),       # Konum
                    cv2.FONT_HERSHEY_SIMPLEX, # Font
                    0.8,                      # Font boyutu
                    (0, 255, 0),             # Yesil renk
                    2                         # Kalinlik
                )
            
            # -----------------------------------------
            # Renk Donusumu (BGR -> RGB)
            # -----------------------------------------
            # OpenCV BGR, Streamlit RGB bekler
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # -----------------------------------------
            # Frame'i Goster
            # -----------------------------------------
            # placeholder.image() ile dinamik guncelleme
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # -----------------------------------------
            # Istatistikleri Guncelle
            # -----------------------------------------
            progress = frame_count / total_frames
            
            # HTML ile canli istatistikler
            stats_placeholder.markdown(f"""
            <div class="stat-card" style="margin-bottom: 1rem;">
                <h3>{len(detections)}</h3>
                <p>Anlık Araba</p>
            </div>
            <div class="stat-card" style="margin-bottom: 1rem;">
                <h3>{total_detections}</h3>
                <p>Toplam Tespit</p>
            </div>
            <div class="stat-card" style="margin-bottom: 1rem;">
                <h3>{current_fps:.1f}</h3>
                <p>İşleme FPS</p>
            </div>
            <div class="stat-card">
                <h3>{frame_count}/{total_frames}</h3>
                <p>Frame</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            progress_placeholder.progress(progress, text=f"İlerleme: {progress:.0%}")
            
            # -----------------------------------------
            # Kucuk Gecikme (Streamlit Uyumlulugu)
            # -----------------------------------------
            # Streamlit'in UI'i guncelleyebilmesi icin kisa bekleme
            time.sleep(0.01)
        
        # -----------------------------------------
        # Video Bitti - Temizlik
        # -----------------------------------------
        cap.release()
        
        # Session state guncelle
        st.session_state.is_detecting = False
        st.session_state.total_cars = total_detections
        st.session_state.frames_processed = frame_count
        
        # -----------------------------------------
        # Sonuc Ozeti
        # -----------------------------------------
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        st.divider()
        st.markdown("### ✅ Tespit Tamamlandi!")
        
        # 4 sutunlu sonuc metrikleri
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        
        with result_col1:
            st.metric("Toplam Tespit", total_detections)
        
        with result_col2:
            st.metric("İşlenen Frame", frame_count)
        
        with result_col3:
            st.metric("Ortalama FPS", f"{avg_fps:.1f}")
        
        with result_col4:
            st.metric("Toplam Süre", f"{total_time:.1f}s")
        
        # Kutlama animasyonu
        st.balloons()

else:
    # =========================================================================
    # VIDEO YUKLENMEMISSE - BILGILENDIRME
    # =========================================================================
    
    st.info("👆 Baslamak icin yukaridan bir video yukleyin.")
    
    # Kullanim adimlari
    st.markdown("### 📖 Nasil Kullanilir?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>1. Video Yukle</h4>
            <p>Araba tespiti yapilacak videoyu yukleyin. MP4, AVI, MOV formatlarini destekler.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>2. Ayarlari Yap</h4>
            <p>Yan panelden model boyutu ve confidence degerini ayarlayin.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>3. Tespiti Baslat</h4>
            <p>Butona tiklayin ve YOLO'nun arabalari tespit etmesini izleyin!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # -----------------------------------------
    # YOLO Hakkinda Bilgi
    # -----------------------------------------
    st.divider()
    st.markdown("### 🤖 YOLO Nedir?")
    
    st.markdown("""
    **YOLO (You Only Look Once)** gercek zamanli nesne tespiti icin tasarlanmis 
    bir derin ogrenme modelidir.
    
    **Temel Ozellikler:**
    - ⚡ **Gercek zamanli**: Tek bir ileri gecis ile tespit
    - 🎯 **Yuksek dogruluk**: COCO dataset'inde egitildi
    - 🔧 **Kolay kullanim**: Ultralytics kutuphanesi ile
    - 📦 **80 sinif**: Araba, insan, hayvan ve daha fazlasi
    
    **Bu Uygulamada:**
    - YOLOv8 modeli kullanilir
    - Sadece araba (car) sinifi filtrelenir
    - Opsiyonel olarak tum tasitlar tespit edilebilir
    """)
    
    # -----------------------------------------
    # Model Karsilastirma Tablosu
    # -----------------------------------------
    st.markdown("### 📊 Model Karsilastirmasi")
    
    # Python dict olarak tablo verisi
    model_data = {
        "Model": ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"],
        "Boyut": ["6.3 MB", "22.5 MB", "52.0 MB", "87.7 MB", "136.7 MB"],
        "mAP": ["37.3", "44.9", "50.2", "52.9", "53.9"],
        "Hiz (ms)": ["0.6", "0.9", "1.7", "2.9", "4.7"]
    }
    
    # Streamlit tablo gorunumu
    st.table(model_data)


# =============================================================================
# SAYFA ALT BILGISI
# =============================================================================

st.divider()

st.caption("""
**Not:** Bu uygulama YOLOv8 modelini kullanmaktadir. Ilk calistirmada model 
otomatik olarak indirilir. GPU varsa otomatik olarak kullanilir, yoksa CPU 
uzerinde calisir (daha yavas).
""")
