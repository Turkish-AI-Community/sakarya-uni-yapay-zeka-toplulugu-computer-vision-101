"""
================================================================================
Dosya Tarayici - Streamlit Web Arayuzu (Document Scanner - Web Interface)
================================================================================

Bu dosya, belge tarama uygulamasinin web arayuzunu saglar. Streamlit framework'u
kullanilarak olusturulmustur.

UYGULAMA AMACI:
---------------
Kullanicilarin telefon veya kamera ile cektikleri belge fotograflarini
profesyonel gorunumlu, taranmis PDF dosyalarina donusturmelerini saglar.

KULLANILAN TEKNOLOJILER:
------------------------
1. Streamlit: Python ile hizli web uygulamasi gelistirme framework'u
   - st.file_uploader: Dosya yukleme
   - st.image: Goruntu gosterme
   - st.columns: Sutun bazli layout
   - st.sidebar: Yan panel ayarlari

2. OpenCV (cv2): Goruntu isleme islemleri
   - Renk donusumu (BGR <-> RGB)
   - Goruntu uzerinde cizim (daire, cizgi, metin)

3. PIL/Pillow: Python goruntu kutuphanesi
   - Dosya formatlarini okuma
   - NumPy array donusumleri

4. NumPy: Sayisal islemler
   - Goruntu verileri icin array yapilar

KULLANIM:
---------
Terminal'de calistirmak icin:
    $ streamlit run app.py

Tarayicida http://localhost:8501 adresinde acilir.

DOSYA YAPISI:
-------------
1. Import ve Kurulum
2. CSS Stilleri (Gorsel tasarim)
3. Sidebar (Ayarlar paneli)
4. Ana Icerik (Goruntu yukleme ve isleme)
5. Sonuc Gosterimi ve PDF Indirme

Yazar: SAU Yapay Zeka & Bilgisayarli Goru Toplulugu
================================================================================
"""

# =============================================================================
# KUTUPHANELERIN ICE AKTARILMASI (Import Statements)
# =============================================================================

# Streamlit: Web uygulamasi framework'u
# Bu kutuphane ile Python kodunu kolayca web arayuzune donusturuyoruz
import streamlit as st

# OpenCV: Bilgisayarli goru kutuphanesi
# Goruntu isleme, kenar tespiti, perspektif duzeltme gibi islemler icin kullanilir
import cv2

# NumPy: Sayisal hesaplama kutuphanesi
# Goruntuleri cok boyutlu sayisal diziler olarak islemek icin kullanilir
import numpy as np

# PIL (Python Imaging Library): Goruntu dosyalarini okumak ve yazmak icin
from PIL import Image

# io: Byte stream islemleri icin (goruntu verilerini bellekte isleme)
import io

# tempfile: Gecici dosya olusturmak icin (PDF ciktilarini gecici olarak kaydetme)
import tempfile

# os: Dosya sistemi islemleri (dosya adi ayirma, dosya silme vb.)
import os

# =============================================================================
# YEREL MODULLERIN ICE AKTARILMASI
# =============================================================================
# Bu fonksiyonlar utils.py dosyasinda tanimlanmistir ve belge tarama
# isleminin temel adimlarini gerceklestirirler

from utils import (
    kenar_tespit,       # Canny edge detection ile kenar tespiti
    belge_tespit,       # Belge konturunun (dis hatlarinin) bulunmasi
    koseler_sirala,     # 4 koseyi saat yonunde siralama
    perspektif_duzelt,  # Homografi ile perspektif duzeltme
    goruntu_iyilestir,  # Adaptif esikleme ile taranmis gorunum
    pdf_olustur         # Sonucu PDF formatinda kaydetme
)


# =============================================================================
# STREAMLIT SAYFA YAPILANDIRMASI
# =============================================================================
# set_page_config() fonksiyonu sayfa basligini, ikonunu ve layout'unu belirler
# Bu fonksiyon Streamlit uygulamasinin EN BASINDA cagirilmalidir

st.set_page_config(
    page_title="Dosya Tarayici",  # Tarayici sekmesinde gorunecek baslik
    page_icon="page_facing_up",   # Sekme ikonu (emoji veya dosya yolu)
    layout="wide"                  # Genis ekran layout'u ("wide" veya "centered")
)


# =============================================================================
# OZEL CSS STILLERI
# =============================================================================
# Streamlit'in varsayilan gorunumunu ozellestirmek icin CSS kullaniyoruz
# st.markdown() ile HTML/CSS kodu enjekte edebiliriz
# unsafe_allow_html=True: Guvenlik kontrolunu devre disi birakir (HTML izni)

st.markdown("""
<style>
    /* =========================================
       Ana Baslik Alani Stili
       =========================================
       Gradient arka plan ve yuvarlatilmis koseler
       ile profesyonel bir baslik olusturur.
    */
    .main-header {
        text-align: center;                /* Metni ortala */
        padding: 1rem;                     /* Ic bosluk (16px) */
        /* Linear gradient: Sol-alt -> Sag-ust yonunde renk gecisi */
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;               /* Yuvarlatilmis koseler */
        color: white;                      /* Metin rengi */
        margin-bottom: 2rem;               /* Alt bosluk */
    }
    
    /* =========================================
       Bilgi Karti Stili
       =========================================
       Sol kenarda renkli cizgi olan bilgi kutulari.
       Material Design'dan esinlenmistir.
    */
    .info-card {
        background: #f8f9fa;               /* Acik gri arka plan */
        padding: 1rem;                     /* Ic bosluk */
        border-radius: 8px;                /* Yuvarlatilmis koseler */
        border-left: 4px solid #667eea;    /* Sol kenarda vurgu cizgisi */
        margin: 1rem 0;                    /* Ust ve alt bosluk */
    }
    
    /* =========================================
       Adim Gostergesi (Step Indicator)
       =========================================
       Islem adimlarini gostermek icin kullanilir.
       Pill (hap) seklinde gorunum.
    */
    .step-indicator {
        background: #e9ecef;               /* Varsayilan: gri arka plan */
        padding: 0.5rem 1rem;              /* Ic bosluk */
        border-radius: 20px;               /* Pill seklinde yuvarlaklok */
        display: inline-block;             /* Yan yana siralama */
        margin: 0.25rem;                   /* Dis bosluk */
        font-size: 0.9rem;                 /* Yazi boyutu */
    }
    
    /* Aktif adim: Mor arka plan, beyaz yazi */
    .step-active {
        background: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ANA BASLIK BOLUMU
# =============================================================================
# HTML ile ozel tasarimli baslik. Markdown icerisinde HTML kullaniyoruz.

st.markdown("""
<div class="main-header">
    <h1>Dosya Tarayici</h1>
    <p>Belge fotograflarinizi profesyonel PDF'lere donusturun</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# YAN PANEL (SIDEBAR) - AYARLAR MENUSU
# =============================================================================
# with st.sidebar: blogu icerisindeki tum elemanlar sol yan panelde gosterilir
# Sidebar, kullanici ayarlarini ana icerikten ayirmak icin idealdir

with st.sidebar:
    # Panel basligi
    st.header("Ayarlar")
    
    # -----------------------------------------
    # Goruntu Isleme Secenekleri
    # -----------------------------------------
    st.subheader("Goruntu Isleme")
    
    # Checkbox: Goruntu iyilestirme ozelligi
    # Bu secenek aktifken adaptif esikleme uygulanir (siyah-beyaz taranmis gorunum)
    iyilestirme_aktif = st.checkbox(
        "Goruntu Iyilestirme",        # Checkbox etiketi
        value=True,                    # Varsayilan olarak aktif
        help="Adaptif esikleme ile taranmis gorunum"  # Tooltip (hover bilgisi)
    )
    
    # Checkbox: Perspektif duzeltme ozelligi
    # Bu secenek aktifken egik cekilmis belgeler duzeltilir
    perspektif_aktif = st.checkbox(
        "Perspektif Duzeltme", 
        value=True,
        help="Egik belgeleri otomatik duzelt"
    )
    
    # Yatay ayirici cizgi
    st.divider()
    
    # -----------------------------------------
    # Kullanim Kilavuzu
    # -----------------------------------------
    st.subheader("Nasil Kullanilir?")
    st.markdown("""
    1. Belge fotografi yukleyin
    2. Onizlemeyi kontrol edin
    3. Gerekirse ayarlari degistirin
    4. PDF olarak indirin
    """)
    
    st.divider()
    
    # Alt bilgi (footer)
    st.caption("SAU Yapay Zeka & Bilgisayarli Goru")


# =============================================================================
# ANA ICERIK BOLUMU - DOSYA YUKLEME
# =============================================================================

# Dosya yukleme alaninin basligi
st.subheader("Belge Yukle")

# Streamlit file_uploader widget'i
# Bu widget kullanicinin bilgisayarindan dosya yuklemesini saglar
yuklenen_dosya = st.file_uploader(
    "Belge fotografini surukleyip birakin veya secin",  # Aciklama metni
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],         # Kabul edilen dosya turleri
    help="Desteklenen formatlar: JPG, PNG, BMP, WEBP"   # Tooltip bilgisi
)


# =============================================================================
# GORUNTU ISLEME VE GOSTERIM
# =============================================================================
# Dosya yuklendiyse isleme basliyoruz

if yuklenen_dosya is not None:
    # =========================================================================
    # ADIM 1: GORUNTUNUN OKUNMASI VE FORMAT DONUSUMU
    # =========================================================================
    
    # PIL (Pillow) ile goruntuyu oku
    # file_uploader'dan gelen veri bir BytesIO nesnesidir
    pil_goruntu = Image.open(yuklenen_dosya)
    
    # PIL goruntusu -> NumPy dizisine donustur
    # NumPy dizileri OpenCV ile uyumludur
    # Shape: (yukseklik, genislik, kanal_sayisi) - ornegin (1080, 1920, 3)
    goruntu_rgb = np.array(pil_goruntu)
    
    # RGB -> BGR donusumu (OpenCV icin)
    # PIL: RGB sirasi (Kirmizi, Yesil, Mavi)
    # OpenCV: BGR sirasi (Mavi, Yesil, Kirmizi)
    # Bu donusum OpenCV fonksiyonlarinin dogru calisması icin gereklidir
    if len(goruntu_rgb.shape) == 3 and goruntu_rgb.shape[2] == 3:
        # Renkli goruntu (3 kanal)
        goruntu = cv2.cvtColor(goruntu_rgb, cv2.COLOR_RGB2BGR)
    else:
        # Gri tonlama veya alpha kanalli goruntu
        goruntu = goruntu_rgb
    
    # =========================================================================
    # ADIM 2-5: BELGE ISLEME PIPELINE'I
    # =========================================================================
    
    # st.status: Islem durumunu gosteren genisletilebilir kutu
    # expanded=True: Baslangicta acik olarak goster
    with st.status("Belge isleniyor...", expanded=True) as status:
        
        # ---------------------------------------------------------------------
        # ADIM 2: KENAR TESPITI (Edge Detection)
        # ---------------------------------------------------------------------
        # Canny algoritmasi ile goruntudenki kenarlari tespit ediyoruz
        # Bu adim gorsellenshtirme icindir, asil belge tespiti farkli yapilir
        st.write("Kenar tespiti yapiliyor...")
        kenarlar = kenar_tespit(goruntu)
        
        # ---------------------------------------------------------------------
        # ADIM 3: BELGE BOLGESI TESPITI (Document Detection)
        # ---------------------------------------------------------------------
        # Belgenin 4 kosesini bulmaya calisiyoruz
        # Birden fazla yontem denenir:
        # 1. Canny + Morfolojik islemler
        # 2. Adaptif esikleme
        # 3. Farkli Canny parametreleri
        st.write("Belge bolgesi tespit ediliyor...")
        
        # Goruntu boyutlarini al
        # shape[0]: Yukseklik (satir sayisi)
        # shape[1]: Genislik (sutun sayisi)
        yukseklik, genislik = goruntu.shape[:2]
        
        if perspektif_aktif:
            # Belge konturunu bul (4 kose noktasi)
            kontur = belge_tespit(goruntu)
            
            if kontur is not None:
                # Koseler basariyla tespit edildi
                # Koseleri saat yonunde sirala: [sol-ust, sag-ust, sag-alt, sol-alt]
                koseler = koseler_sirala(kontur)
                kose_bulundu = True
                st.write("Belge basariyla tespit edildi!")
            else:
                # Koseler bulunamadiysa tum goruntuyu belge olarak kabul et
                # Goruntunun 4 kose noktasini manuel olarak belirliyoruz
                koseler = np.array([
                    [0, 0],                       # Sol-ust
                    [genislik - 1, 0],            # Sag-ust
                    [genislik - 1, yukseklik - 1], # Sag-alt
                    [0, yukseklik - 1]            # Sol-alt
                ], dtype=np.float32)
                kose_bulundu = False
                st.write("UYARI: Belge koseleri tespit edilemedi, tum goruntu kullanilacak.")
        else:
            # Perspektif duzeltme devre disi - tum goruntuyu kullan
            koseler = np.array([
                [0, 0],
                [genislik - 1, 0],
                [genislik - 1, yukseklik - 1],
                [0, yukseklik - 1]
            ], dtype=np.float32)
            kose_bulundu = False
        
        # ---------------------------------------------------------------------
        # ADIM 4: PERSPEKTIF DUZELTME (Homography Transformation)
        # ---------------------------------------------------------------------
        # Egik cekilmis belgeyi duz dikdortgen haline getiriyoruz
        # Bu islem "kus bakisi" (bird's eye view) olusturur
        if perspektif_aktif:
            st.write("Perspektif duzeltiliyor...")
            duzeltilmis = perspektif_duzelt(goruntu, koseler)
        else:
            # Perspektif duzeltme kapaliysa orijinal goruntuyu kopyala
            duzeltilmis = goruntu.copy()
        
        # ---------------------------------------------------------------------
        # ADIM 5: GORUNTU IYILESTIRME (Image Enhancement)
        # ---------------------------------------------------------------------
        # Adaptif esikleme ile taranmis belge gorunumu olusturuyoruz
        if iyilestirme_aktif:
            st.write("Goruntu iyilestiriliyor...")
            sonuc = goruntu_iyilestir(duzeltilmis)
        else:
            # Iyilestirme kapaliysa sadece duzeltilmis goruntuyu kullan
            sonuc = duzeltilmis
        
        # Islem tamamlandi bildirimi
        status.update(label="Islem tamamlandi!", state="complete")
    
    # =========================================================================
    # SONUCLARIN GOSTERIMI
    # =========================================================================
    
    st.subheader("Onizleme")
    
    # Iki sutunlu layout: Sol'da orijinal, sag'da sonuc
    col1, col2 = st.columns(2)
    
    # -----------------------------------------
    # Sol Sutun: Orijinal Goruntu
    # -----------------------------------------
    with col1:
        st.markdown("**Orijinal Goruntu**")
        
        # Goruntu uzerine kose noktalarini ciz (tespit edildiyse)
        goruntu_isaret = goruntu.copy()
        
        if kose_bulundu and perspektif_aktif:
            # Her kose noktasi icin yesil daire ve numara ciz
            for i, kose in enumerate(koseler):
                # Kose noktasinda yesil daire
                # circle(goruntu, merkez, yaricap, renk, kalinlik)
                # kalinlik=-1 icini dolu yapar
                cv2.circle(
                    goruntu_isaret, 
                    tuple(kose.astype(int)),  # Koordinatlari int'e cevir
                    15,                        # Yaricap: 15 piksel
                    (0, 255, 0),              # Renk: Yesil (BGR)
                    -1                         # Icini doldur
                )
                
                # Kose numarasini yaz (1, 2, 3, 4)
                # putText(goruntu, metin, konum, font, olcek, renk, kalinlik)
                cv2.putText(
                    goruntu_isaret, 
                    str(i+1),                              # Numara (1'den baslar)
                    tuple((kose + [5, 5]).astype(int)),    # Konumu biraz kaydir
                    cv2.FONT_HERSHEY_SIMPLEX,              # Font tipi
                    1.5,                                   # Font olcegi
                    (255, 0, 0),                           # Renk: Mavi (BGR)
                    3                                      # Kalinlik
                )
            
            # Koseleri birlestiren yesil cizgiler ciz
            # polylines kapalİ bir cokgen cizer (True = kapalİ)
            pts = koseler.astype(int).reshape((-1, 1, 2))
            cv2.polylines(goruntu_isaret, [pts], True, (0, 255, 0), 3)
        
        # BGR -> RGB donusumu (Streamlit icin)
        # Streamlit, goruntuleri RGB formatinda bekler
        goruntu_goster = cv2.cvtColor(goruntu_isaret, cv2.COLOR_BGR2RGB)
        
        # Goruntuyu goster
        # use_container_width=True: Container genisligine sigar
        st.image(goruntu_goster, use_container_width=True)
    
    # -----------------------------------------
    # Sag Sutun: Islenmis Sonuc
    # -----------------------------------------
    with col2:
        st.markdown("**Islenmis Sonuc**")
        
        # Sonuc goruntusu renkli mi yoksa gri tonlama mi kontrol et
        if len(sonuc.shape) == 3:
            # Renkli goruntu - BGR'den RGB'ye donustur
            sonuc_goster = cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB)
        else:
            # Gri tonlama goruntu - dogrudan kullan
            sonuc_goster = sonuc
        
        st.image(sonuc_goster, use_container_width=True)
    
    # =========================================================================
    # ISLEM ADIMLARINI GOSTERME (Genisletilebilir Panel)
    # =========================================================================
    # st.expander: Tiklandikginda acilan/kapanan panel
    # Detaylari gizleyerek arayuzu temiz tutmak icin kullanislidir
    
    with st.expander("Islem Adimlarini Gor"):
        # Uc sutunlu layout
        adim_col1, adim_col2, adim_col3 = st.columns(3)
        
        # Kenar Tespiti sonucu
        with adim_col1:
            st.markdown("**1. Kenar Tespiti**")
            st.caption("Canny Edge Detection")
            # Gri tonlama goruntuyu 3 kanala cevir (gosterim icin)
            kenarlar_goster = cv2.cvtColor(kenarlar, cv2.COLOR_GRAY2RGB)
            st.image(kenarlar_goster, use_container_width=True)
        
        # Perspektif Duzeltme sonucu
        with adim_col2:
            st.markdown("**2. Perspektif Duzeltme**")
            st.caption("Homography Transform")
            if len(duzeltilmis.shape) == 3:
                duzeltilmis_goster = cv2.cvtColor(duzeltilmis, cv2.COLOR_BGR2RGB)
            else:
                duzeltilmis_goster = duzeltilmis
            st.image(duzeltilmis_goster, use_container_width=True)
        
        # Goruntu Iyilestirme sonucu
        with adim_col3:
            st.markdown("**3. Goruntu Iyilestirme**")
            st.caption("Adaptive Thresholding")
            st.image(sonuc_goster, use_container_width=True)
    
    # =========================================================================
    # PDF INDIRME BOLUMU
    # =========================================================================
    
    st.subheader("PDF Indir")
    
    # Orijinal dosya adindan PDF adi olustur
    # os.path.splitext: Dosya adini ve uzantisini ayirir
    orijinal_ad = os.path.splitext(yuklenen_dosya.name)[0]
    varsayilan_ad = f"{orijinal_ad}_taranmis.pdf"
    
    # Kullanicinin PDF adini degistirebilmesi icin text input
    pdf_adi = st.text_input("PDF dosya adi:", value=varsayilan_ad)
    
    # PDF olusturma butonu
    # type="primary" butonu vurgulu gosterir
    if st.button("PDF Olustur ve Indir", type="primary", use_container_width=True):
        
        # Spinner: Islem sirasinda dongu animasyonu goster
        with st.spinner("PDF olusturuluyor..."):
            # Gecici dosya olustur (guvenli ve benzersiz isim)
            # suffix='.pdf': .pdf uzantili gecici dosya
            # delete=False: Hemen silinmesin, manuel silecegiz
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                temp_pdf_yol = tmp.name
            
            try:
                # PDF'i gecici dosyaya olustur
                pdf_olustur(sonuc, temp_pdf_yol)
                
                # PDF dosyasini binary modda oku
                with open(temp_pdf_yol, 'rb') as f:
                    pdf_bytes = f.read()
                
                # Streamlit download butonu
                # Bu buton kullaniciya dosya indirme imkani saglar
                st.download_button(
                    label="PDF'i Indir",
                    data=pdf_bytes,                                        # Dosya icerigi
                    file_name=pdf_adi if pdf_adi.endswith('.pdf') else f"{pdf_adi}.pdf",
                    mime="application/pdf",                                # MIME tipi
                    use_container_width=True
                )
                
                # Basari mesaji
                st.success("PDF basariyla olusturuldu! Yukaridaki butona tiklayarak indirebilirsiniz.")
                
            finally:
                # Gecici dosyayi temizle (her durumda calisir)
                # Finally blogu hata olsa bile calisir
                if os.path.exists(temp_pdf_yol):
                    os.remove(temp_pdf_yol)


else:
    # =========================================================================
    # DOSYA YUKLENMEMISSE - BILGILENDIRME EKRANI
    # =========================================================================
    
    # Bilgi mesaji
    st.info("Baslamak icin yukaridan bir belge fotografi yukleyin.")
    
    # Kullanim adimlari - 3 sutunlu kart gorunumu
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>1. Fotograf Cekin</h4>
            <p>Belgenizi duz bir yuzeye koyun ve fotografini cekin.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>2. Yukleyin</h4>
            <p>Fotografi bu uygulamaya surukleyip birakin.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>3. Indirin</h4>
            <p>Islenmis belgenizi PDF olarak indirin.</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SAYFA ALT BILGISI (Footer)
# =============================================================================

st.divider()

# st.caption: Kucuk, soluk renkli metin
st.caption("""
**Ipucu:** En iyi sonuc icin belgenin tum koselerinin gorunur oldugundan 
ve iyi aydinlatilmis oldugundan emin olun.
""")
