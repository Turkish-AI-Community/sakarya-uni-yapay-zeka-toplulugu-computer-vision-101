"""
================================================================================
CLIP Image Analyzer - Streamlit Web Arayuzu
================================================================================

OpenAI CLIP modeli kullanarak goruntu analizi yapan web uygulamasi.

CLIP NEDIR?
===========
CLIP (Contrastive Language-Image Pre-training), OpenAI tarafindan gelistirilen
ve goruntuleri metinlerle eslestirme yetenegine sahip bir yapay zeka modelidir.

TEMEL OZELLIKLER:
-----------------
1. Zero-Shot Ogrenme:
   - Model, hic gormedigi kategorileri tanimlar
   - Yeni etiketler icin yeniden egitim gerektirmez
   - "a photo of {label}" sablonuyla sinirsiz kategori

2. Multimodal Temsil:
   - Goruntu ve metni ayni vektorel uzayda temsil eder
   - 512 boyutlu vektorler kullanir
   - Cosine similarity ile benzerlik olcer

3. Genis Egitim Verisi:
   - 400 milyon goruntu-metin cifti
   - Internet'ten toplanan veriler
   - Dogal dil aciklamalari

UYGULAMA OZELLIKLERI:
---------------------
1. Genel Siniflandirma:
   - Hayvan, insan, yemek, arac vb.
   
2. Meme/Goruntu Turu Tespiti:
   - Reaction meme, text meme, comic vb.
   
3. Duygu Analizi:
   - Happy, sad, angry, surprised vb.
   
4. Stil Analizi:
   - Fotograf, cizim, boyama, 3D render vb.
   
5. Ozel Etiketler:
   - Kullanicinin belirledigi etiketlerle analiz

KULLANIM:
---------
Terminal'de calistirmak icin:
    $ streamlit run app.py

Tarayicida http://localhost:8501 adresinde acilir.

GEREKSINIMLER:
--------------
- Python 3.8+
- streamlit
- torch
- transformers
- Pillow
- numpy

Yazar: SAU Yapay Zeka & Bilgisayarli Goru Toplulugu
================================================================================
"""

# =============================================================================
# KUTUPHANELERIN ICE AKTARILMASI
# =============================================================================

# Streamlit: Interaktif web uygulamalari olusturmak icin Python framework'u
# - Hizli prototipleme
# - Otomatik UI olusturma
# - Canli yenileme (hot reload)
import streamlit as st

# NumPy: Sayisal hesaplama kutuphanesi
# - Goruntu verilerini array olarak isler
# - Benzerlik skorlari icin array islemleri
import numpy as np

# PIL (Pillow): Python goruntu kutuphanesi
# - Goruntu dosyalarini okuma
# - Format donusumleri
# - Boyut degistirme
from PIL import Image

# io: Byte stream islemleri
# - Bellekteki goruntu verilerini isleme
import io

# =============================================================================
# YEREL MODUL IMPORTLARI
# =============================================================================
# clip_model.py dosyasindan CLIP analiz sinifi ve yardimci fonksiyonlar

from clip_model import (
    ClipAnalyzer,      # Ana CLIP analiz sinifi
    DEFAULT_LABELS,    # Onceden tanimlanmis etiket listeleri
    get_device_info    # GPU/CPU bilgisi fonksiyonu
)


# =============================================================================
# STREAMLIT SAYFA YAPILANDIRMASI
# =============================================================================
# set_page_config: Sayfa basligini, ikonunu ve layout'unu belirler
# Bu fonksiyon Streamlit scriptinin EN BASINDA cagirilmalidir

st.set_page_config(
    page_title="CLIP Image Analyzer",  # Tarayici sekmesinde gorunur
    page_icon="page_facing_up",        # Sekme ikonu
    layout="wide"                       # Genis ekran modu
)


# =============================================================================
# OZEL CSS STILLERI
# =============================================================================
# Streamlit'in varsayilan gorunumunu ozellestirmek icin CSS
# unsafe_allow_html=True ile HTML/CSS enjekte edebiliriz

st.markdown("""
<style>
    /* =============================================
       Ana Baslik Stili
       =============================================
       Koyu mavi gradient ile profesyonel gorunum
    */
    .main-header {
        text-align: center;
        padding: 1.5rem;
        /* Koyu mavi gradient - soldan saga */
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* =============================================
       Skor Cubugu Konteyner
       =============================================
       Benzerlik skorlarini gorsel olarak gostermek icin
    */
    .score-bar {
        background: #e9ecef;        /* Acik gri arka plan */
        border-radius: 10px;        /* Yuvarlatilmis koseler */
        height: 25px;               /* Cubuk yuksekligi */
        margin: 5px 0;              /* Ust/alt bosluk */
    }
    
    /* Skor cubugu dolumu (animasyonlu) */
    .score-fill {
        /* Mor gradient doluş */
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        height: 100%;
        /* Genislik degisiminde animasyon */
        transition: width 0.3s ease;
    }
    
    /* =============================================
       Bilgi Kutusu Stili
       =============================================
       Aciklama ve yardim metinleri icin
    */
    .info-box {
        background: #f8f9fa;               /* Acik gri arka plan */
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e3c72;    /* Sol kenarda mavi vurgu */
        margin: 1rem 0;
    }
    
    /* =============================================
       Sonuc Karti Stili
       =============================================
       Analiz sonuclarini gostermek icin kart gorunumu
    */
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        /* Hafif golge efekti */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL YUKLEME (CACHE ILE)
# =============================================================================
# @st.cache_resource: Fonksiyonun sonucunu cache'ler
# - Ayni model tekrar yuklenmez
# - Sayfa yenilendiginde bile cache'ten gelir
# - Bellek ve zaman tasarrufu saglar
#
# NEDEN CACHE ONEMLI?
# - CLIP modeli ~300MB
# - GPU'ya yukleme 5-10 saniye surer
# - Her istekte yeniden yuklemek kullanici deneyimini bozar

@st.cache_resource
def load_model():
    """
    CLIP modelini yukler ve cache'ler.
    
    Bu fonksiyon sadece ilk cagirida modeli yukler.
    Sonraki cagirilarda cache'teki modeli dondurur.
    
    Kullanilan Model:
    - openai/clip-vit-base-patch32
    - ViT-B/32 arsitekturu
    - Hizli ve hafif
    
    Returns:
        ClipAnalyzer: Yuklu CLIP model instance'i
    """
    return ClipAnalyzer(model_name="openai/clip-vit-base-patch32")


# =============================================================================
# ANA BASLIK
# =============================================================================
# HTML ile ozel tasarimli baslik

st.markdown("""
<div class="main-header">
    <h1>CLIP Image Analyzer</h1>
    <p>OpenAI CLIP modeli ile goruntu analizi</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# YAN PANEL (SIDEBAR) - AYARLAR
# =============================================================================
# Sidebar: Ana icerikten bagimsiz ayarlar paneli

with st.sidebar:
    st.header("Ayarlar")
    
    # -----------------------------------------
    # Analiz Modu Secimi
    # -----------------------------------------
    st.subheader("Analiz Modu")
    
    # Dropdown menu ile analiz turunu sec
    # Her mod farkli etiket seti kullanir
    analiz_modu = st.selectbox(
        "Analiz turunu secin:",
        [
            "Genel Siniflandirma",  # Genel kategoriler
            "Meme/Goruntu Turu",    # Meme cesitleri
            "Duygu Analizi",        # Duygusal icerik
            "Stil Analizi",         # Gorsel stil
            "Ozel Etiketler"        # Kullanici tanimli
        ]
    )
    
    # -----------------------------------------
    # Ozel Etiketler Girisi
    # -----------------------------------------
    # Sadece "Ozel Etiketler" modu secildiyse goster
    if analiz_modu == "Ozel Etiketler":
        st.subheader("Ozel Etiketler")
        
        # Cok satirli metin alani
        # Her satirda bir etiket
        ozel_etiketler = st.text_area(
            "Her satira bir etiket yazin:",
            value="cat\ndog\nbird\ncar\nhouse",
            height=150
        )
    
    st.divider()
    
    # -----------------------------------------
    # Prompt Sablonu Ayari
    # -----------------------------------------
    st.subheader("Prompt Sablonu")
    
    # CLIP icin prompt template
    # {} yerine etiket yerlestirilir
    # Ornek: "a photo of {}" -> "a photo of cat"
    prompt_template = st.text_input(
        "Prompt sablonu ({} = etiket):",
        value="a photo of {}",
        help="Ornek: 'a photo of {}', 'an image showing {}'"
    )
    
    st.divider()
    
    # -----------------------------------------
    # Sistem Bilgisi
    # -----------------------------------------
    st.subheader("Sistem Bilgisi")
    
    # GPU/CPU durumunu goster
    device_info = get_device_info()
    
    st.write(f"**Cihaz:** {device_info['device'].upper()}")
    
    # GPU varsa detaylari goster
    if device_info['cuda_available']:
        st.write(f"**GPU:** {device_info['gpu_name']}")
        st.write(f"**GPU Bellek:** {device_info['gpu_memory']}")
    
    st.divider()
    st.caption("SAU Yapay Zeka & Bilgisayarli Goru")


# =============================================================================
# MODEL YUKLEME
# =============================================================================
# Spinner: Yukleme sirasinda animasyon goster

with st.spinner("CLIP modeli yukleniyor..."):
    analyzer = load_model()


# =============================================================================
# ANA ICERIK - GORUNTU YUKLEME
# =============================================================================

st.subheader("Goruntu Yukle")

# Dosya yukleme widget'i
yuklenen_dosya = st.file_uploader(
    "Analiz edilecek goruntuyu secin",
    type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
    help="Desteklenen formatlar: JPG, PNG, WEBP, BMP"
)


# =============================================================================
# GORUNTU ANALIZI
# =============================================================================

if yuklenen_dosya is not None:
    # -----------------------------------------
    # Goruntuyu Oku ve Hazirla
    # -----------------------------------------
    
    # PIL ile goruntu dosyasini ac
    image = Image.open(yuklenen_dosya)
    
    # CLIP RGB goruntu bekler
    # RGBA, L, P gibi modlari RGB'ye donustur
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # -----------------------------------------
    # Layout: Goruntu ve Sonuclar Yan Yana
    # -----------------------------------------
    col1, col2 = st.columns([1, 1])
    
    # Sol Sutun: Goruntu Onizleme
    with col1:
        st.markdown("**Yuklenen Goruntu**")
        st.image(image, use_container_width=True)
    
    # Sag Sutun: Analiz Sonuclari
    with col2:
        st.markdown("**Analiz Sonuclari**")
        
        # -----------------------------------------
        # Secilen Moda Gore Etiketleri Belirle
        # -----------------------------------------
        if analiz_modu == "Genel Siniflandirma":
            # Genel kategoriler: person, animal, food, vehicle, etc.
            labels = DEFAULT_LABELS["genel"]
            
        elif analiz_modu == "Meme/Goruntu Turu":
            # Meme turleri: reaction meme, text meme, etc.
            labels = DEFAULT_LABELS["meme_turleri"]
            
        elif analiz_modu == "Duygu Analizi":
            # Duygular: happy, sad, angry, etc.
            labels = DEFAULT_LABELS["duygular"]
            
        elif analiz_modu == "Stil Analizi":
            # Stil: photograph, drawing, painting, etc.
            labels = DEFAULT_LABELS["stil"]
            
        else:  # Ozel Etiketler
            # Kullanicinin girdigi etiketleri parse et
            # Her satir bir etiket, bos satirlari atla
            labels = [l.strip() for l in ozel_etiketler.split('\n') if l.strip()]
        
        # -----------------------------------------
        # CLIP Analizi Yap
        # -----------------------------------------
        with st.spinner("Goruntu analiz ediliyor..."):
            # analyze metodu:
            # 1. Goruntuyu encode eder
            # 2. Etiketleri prompt_template ile metne cevirir
            # 3. Metinleri encode eder
            # 4. Cosine similarity hesaplar
            # 5. Softmax ile olasiliklara cevirir
            results = analyzer.analyze(image, labels, prompt_template)
        
        # -----------------------------------------
        # Sonuclari Goster
        # -----------------------------------------
        # results: [(etiket, skor), ...] - skora gore sirali
        for label, score in results:
            # Her sonuc icin iki sutun: etiket + cubuk, skor
            col_label, col_score = st.columns([3, 1])
            
            with col_label:
                st.write(f"**{label}**")
                # Progress bar ile skor gorsellestirme
                # st.progress 0.0-1.0 arasi deger bekler
                st.progress(float(score / 100))
            
            with col_score:
                st.write(f"**{score:.1f}%**")
        
        # En yuksek skorlu sonuc - basari mesaji
        st.success(f"En yuksek eslesme: **{results[0][0]}** ({results[0][1]:.1f}%)")
    
    # =========================================================================
    # DETAYLI ANALIZ (Genisletilebilir Panel)
    # =========================================================================
    # Tum kategorilerde analiz yaparak detayli sonuclar goster
    
    with st.expander("Detayli Analiz"):
        st.subheader("Tum Kategorilerde Analiz")
        
        # Her kategori icin bir sutun
        kategori_cols = st.columns(len(DEFAULT_LABELS))
        
        # Tum varsayilan etiket kategorilerini isle
        for idx, (kategori, etiketler) in enumerate(DEFAULT_LABELS.items()):
            with kategori_cols[idx]:
                # Kategori basligini goster
                st.markdown(f"**{kategori.upper()}**")
                
                # Bu kategori icin analiz yap
                cat_results = analyzer.analyze(image, etiketler, prompt_template)
                
                # Sadece ilk 3 sonucu goster (en yuksek 3)
                for label, score in cat_results[:3]:
                    st.write(f"{label}: {score:.1f}%")


else:
    # =========================================================================
    # GORUNTU YUKLENMEMISSE - BILGILENDIRME
    # =========================================================================
    
    st.info("Baslamak icin yukaridan bir goruntu yukleyin.")
    
    # Kullanim adimlari - 3 kart
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>1. Goruntu Yukle</h4>
            <p>Analiz edilecek goruntuyu veya meme'i yukleyin.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>2. Mod Sec</h4>
            <p>Yan panelden analiz turunu secin.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>3. Sonuclari Gor</h4>
            <p>CLIP modeli goruntuyu analiz edecek.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # -----------------------------------------
    # CLIP Hakkinda Bilgi
    # -----------------------------------------
    st.divider()
    st.subheader("CLIP Nedir?")
    
    st.markdown("""
    **CLIP (Contrastive Language-Image Pre-training)** OpenAI tarafindan gelistirilen 
    bir yapay zeka modelidir.
    
    **Temel Ozellikler:**
    - 400 milyon goruntu-metin cifti ile egitildi
    - Zero-shot ogrenme: Hic gormedigi kategorileri tanimlar
    - Goruntu ve metni ayni vektorel uzayda temsil eder
    - Cosine similarity ile benzerlik hesaplar
    
    **Kullanim Alanlari:**
    - Goruntu siniflandirma
    - Goruntu arama
    - Icerik moderasyonu
    - Goruntu-metin eslestirme
    """)


# =============================================================================
# SAYFA ALT BILGISI
# =============================================================================

st.divider()

st.caption("""
**Not:** CLIP modeli zero-shot siniflandirma yapar. Yani modele onceden bu etiketleri 
ogretmenize gerek yoktur. Model, goruntu ve metin arasindaki anlami karsilastirir.
""")
