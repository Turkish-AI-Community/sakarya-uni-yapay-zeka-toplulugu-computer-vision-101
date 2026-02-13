"""
================================================================================
Yardimci Fonksiyonlar (Utility Functions) - Belge Tarama Islemleri
================================================================================

Bu modul, dosya tarayici uygulamasi icin gerekli goruntu isleme
fonksiyonlarini icerir. Her fonksiyon belge tarama pipeline'inin
bir adimini gerceklestirir.

BILGISAYARLI GORU KAVRAMLARI:
=============================

1. KENAR TESPITI (Edge Detection)
   ------------------------------
   Kenarlar, goruntudenki parlaklik degisimlerinin oldugu bolgelerdir.
   
   Canny Algoritmasi Adimlari:
   a) Gurultu Azaltma: Gaussian blur ile pürüzsüzlestirme
   b) Gradyan Hesaplama: Sobel operatorü ile x ve y gradyanlari
   c) Non-maximum Suppression: İnce kenarlar elde etme
   d) Hysteresis Thresholding: Kenar parcalarini birbirine baglama
   
   Canny(goruntu, alt_esik, ust_esik)
   - alt_esik: Bu degerden dusuk gradyanlar kenar degil
   - ust_esik: Bu degerden yuksek gradyanlar kesinlikle kenar
   - Aradaki degerler: Guclu kenarlara bagliysa kenar

2. MORFOLOJIK ISLEMLER (Morphological Operations)
   -----------------------------------------------
   Binary goruntulerdeki sekilleri degistiren islemler.
   
   Dilation (Genisletme):
   - Beyaz bölgeleri buyutur
   - Boşluklari kapatir
   - Nesneleri birlestirır
   
   Erosion (Aşındırma):
   - Beyaz bölgeleri küçültür
   - Gürültüyü temizler
   - Nesneleri ayirir
   
   Closing = Dilation + Erosion:
   - Küçük delikleri kapatir
   - Dis hatlari korur

3. KONTUR (Contour)
   -----------------
   Ayni renk/yogunluga sahip surekli noktalar zinciri.
   Nesne sinirlari olarak dusunulebilir.
   
   findContours():
   - RETR_EXTERNAL: Sadece dis konturlar
   - RETR_TREE: Tum hiyerarsi
   - CHAIN_APPROX_SIMPLE: Sadece kose noktaları

4. PERSPEKTIF DONUSUMU (Perspective Transform / Homography)
   ---------------------------------------------------------
   3D dunyadan 2D goruntuye geciste kaybolan derinlik bilgisini
   kullanarak goruntuyu donusturen islem.
   
   4 nokta gerektirir:
   - Kaynak noktalar (eğik dörtgen köşeleri)
   - Hedef noktalar (düz dikdörtgen köşeleri)
   
   Homografi Matrisi (3x3):
   [x']   [h11 h12 h13]   [x]
   [y'] = [h21 h22 h23] * [y]
   [w']   [h31 h32 h33]   [1]

5. ADAPTIF ESIKLEME (Adaptive Thresholding)
   -----------------------------------------
   Her piksel icin farkli esik degeri hesaplar.
   Degisen aydinlatma kosullarina dayanikli.
   
   cv2.adaptiveThreshold(goruntu, max_deger, metod, tip, blok_boyutu, C)
   - ADAPTIVE_THRESH_GAUSSIAN_C: Gaussian agirlikli ortalama
   - ADAPTIVE_THRESH_MEAN_C: Basit ortalama

Yazar: SAU Yapay Zeka & Bilgisayarli Goru Toplulugu
================================================================================
"""

# =============================================================================
# KUTUPHANELERIN ICE AKTARILMASI
# =============================================================================

# OpenCV: Acik kaynakli bilgisayarli goru kutuphanesi
# 2500+ algorit+ma icerir: goruntu isleme, makine ogrenimi, video analizi
import cv2

# NumPy: Python icin bilimsel hesaplama kutuphanesi
# N-boyutlu array islemleri icin kullanilir
# Goruntuleri (yukseklik, genislik, kanal) seklinde array olarak temsil eder
import numpy as np

# os: Isletim sistemi arayuzu
# Dosya yolu islemleri, dosya varlik kontrolu vb.
import os

# tempfile: Gecici dosya olusturma
# PDF olusturma sirasinda gecici JPEG dosyasi icin kullanilir
import tempfile


# =============================================================================
# ADIM 1: ON-ISLEME (PREPROCESSING)
# =============================================================================

def on_isleme(goruntu: np.ndarray) -> np.ndarray:
    """
    Goruntuyu kenar tespiti icin hazirlar.
    
    ON-ISLEME NEDEN GEREKLI?
    ------------------------
    Ham goruntulerdeki gurultu, kenar tespiti algoritmalarinin
    yanlis pozitifler (sahte kenarlar) uretmesine neden olur.
    On-isleme ile:
    1. Gurultu azaltilir
    2. Kontrast dengelenir
    3. Kenarlar belirginlestirilir
    
    UYGULANAN ISLEMLER:
    -------------------
    1. Gri Tonlama Donusumu (Grayscale Conversion):
       - 3 kanaldan (BGR) 1 kanala (Gray) gecis
       - Her piksel: 0-255 arasi tek deger
       - Formul: Gray = 0.299*R + 0.587*G + 0.114*B
       - Gozun yesile daha hassas olmasi nedeniyle yesil agirikli
    
    2. Gaussian Blur (Gaussian Bulaniklastirma):
       - Normal dagilim (Gauss cani) kernel'i ile konvolusyon
       - Merkeze yakin pikseller daha etkili
       - Kenarlari koruyarak gurultuyu azaltir
       - Kernel boyutu (5,5): 5x5 piksellik pencere
    
    PARAMETRELER:
    -------------
    goruntu : np.ndarray
        BGR formatinda giris goruntusu.
        Shape: (yukseklik, genislik, 3)
    
    DONDURUR:
    ---------
    np.ndarray
        Gri tonlama, bulaniklastirilmis goruntu.
        Shape: (yukseklik, genislik)
    
    ORNEK:
    ------
    >>> goruntu = cv2.imread("belge.jpg")
    >>> islenmis = on_isleme(goruntu)
    >>> print(islenmis.shape)  # (480, 640)
    """
    
    # -----------------------------------------
    # Adim 1: BGR -> Gri Tonlama Donusumu
    # -----------------------------------------
    # COLOR_BGR2GRAY: Blue-Green-Red -> Grayscale
    # 3 kanalli goruntu tek kanala indirgenir
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    
    # -----------------------------------------
    # Adim 2: Gaussian Blur
    # -----------------------------------------
    # GaussianBlur(kaynak, kernel_boyutu, sigmaX)
    # - (5, 5): 5x5 piksellik Gaussian kernel
    # - 0: SigmaX otomatik hesaplanir (kernel boyutundan)
    # Gurultuyu azaltir, kenar tespitini iyilestirir
    bulanik = cv2.GaussianBlur(gri, (5, 5), 0)
    
    return bulanik


# =============================================================================
# ADIM 2: KENAR TESPITI (EDGE DETECTION)
# =============================================================================

def kenar_tespit(goruntu: np.ndarray) -> np.ndarray:
    """
    Canny Edge Detection ve morfolojik islemler ile kenar tespiti.
    
    CANNY ALGORITMASI DETAYLARI:
    ----------------------------
    John F. Canny tarafindan 1986'da gelistirildi.
    "Optimal" kenar tespiti icin 3 kriteri karsilar:
    
    1. Dusuk Hata Orani:
       - Gercek kenarlarin cogu tespit edilmeli
       - Az sayida yanlis pozitif olmali
    
    2. Iyi Lokalizasyon:
       - Tespit edilen kenarlar gercek konuma yakin olmali
    
    3. Tek Yanit:
       - Her kenar icin tek bir tepki olmali
    
    ALGORITMA ADIMLARI:
    -------------------
    1. Gurultu Azaltma: 5x5 Gaussian filtre
    
    2. Gradyan Hesaplama:
       - Sobel operatorleri ile x ve y yonunde turevler
       - Gradyan buyuklugu: sqrt(Gx^2 + Gy^2)
       - Gradyan yonu: arctan(Gy/Gx)
    
    3. Non-Maximum Suppression:
       - Gradyan yonunde en buyuk degerleri koru
       - İnce, tek piksel genisliginde kenarlar olustur
    
    4. Hysteresis Thresholding:
       - Yuksek esik (ust): Kesinlikle kenar
       - Dusuk esik (alt): Guclu kenara bagliysa kenar
       - Bu sayede kenarlar kopuk kalmaz
    
    MORFOLOJIK ISLEMLER:
    --------------------
    Canny sonrasi kenarlari guclendirmek icin:
    
    1. Dilation (Genisletme):
       - Beyaz piksellerin komsularina yayilmasi
       - Kopuk kenarlari baglar
       - iterations=2: 2 kez tekrar
    
    2. Erosion (Asindirma):
       - Beyaz bolgeleri küçültme
       - Asiri kalinlasmis kenarlari inceltme
       - iterations=1: 1 kez uygula
    
    PARAMETRELER:
    -------------
    goruntu : np.ndarray
        BGR formatinda giris goruntusu
    
    DONDURUR:
    ---------
    np.ndarray
        Ikili (binary) kenar goruntusu.
        Kenarlar: 255 (beyaz)
        Arka plan: 0 (siyah)
    """
    
    # On-isleme (gri tonlama + blur)
    bulanik = on_isleme(goruntu)
    
    # -----------------------------------------
    # Canny Kenar Tespiti
    # -----------------------------------------
    # cv2.Canny(goruntu, alt_esik, ust_esik)
    # - alt_esik=50: Bu degerden dusuk gradyanlar kenar degil
    # - ust_esik=150: Bu degerden yuksek gradyanlar kesinlikle kenar
    # 
    # Esik orani genellikle 1:2 veya 1:3 secilir
    kenarlar = cv2.Canny(bulanik, 50, 150)
    
    # -----------------------------------------
    # Morfolojik Islemler
    # -----------------------------------------
    # Yapisal eleman (structuring element) olustur
    # MORPH_RECT: Dikdortgen seklinde kernel
    # (3, 3): 3x3 boyutunda
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Dilation: Kenarlari genislet, bosluklari kapat
    # iterations=2: 2 kez tekrar et
    kenarlar = cv2.dilate(kenarlar, kernel, iterations=2)
    
    # Erosion: Asiri kalin kenarlari incelt
    # iterations=1: 1 kez uygula
    kenarlar = cv2.erode(kenarlar, kernel, iterations=1)
    
    return kenarlar


def adaptif_esikleme(goruntu: np.ndarray) -> np.ndarray:
    """
    Adaptif esikleme ile belge bolgesi tespiti.
    
    NEDEN ADAPTIF ESIKLEME?
    -----------------------
    Normal (global) esikleme tek bir esik degeri kullanir.
    Ancak belge fotograflarinda:
    - Farkli bolgelerde farkli aydinlatma olabilir
    - Golge dusen yerler daha karanlik
    - Isik yansimasi olan yerler daha parlak
    
    Adaptif esikleme her piksel icin farkli esik hesaplar:
    - Komsu piksellerin ortalamasini kullanir
    - Lokal aydinlatma degisimlerine uyum saglar
    
    CANNY BASARISIZ OLDUGUNDA:
    --------------------------
    Canny bazi durumlarda belge kenarlarini bulamayabilir:
    - Dusuk kontrastli goruntulerde
    - Cok gurultulu goruntulerde
    - Belge ve arka plan renkleri benzer oldugunda
    
    Bu durumlarda adaptif esikleme alternatif olarak kullanilir.
    
    PARAMETRELER:
    -------------
    goruntu : np.ndarray
        BGR formatinda giris goruntusu
    
    DONDURUR:
    ---------
    np.ndarray
        Ikili (binary) goruntu.
        Belge bolgeleri: 255
        Arka plan: 0
    """
    
    # On-isleme
    bulanik = on_isleme(goruntu)
    
    # -----------------------------------------
    # Adaptif Esikleme
    # -----------------------------------------
    # cv2.adaptiveThreshold parametreleri:
    #
    # bulanik: Giris goruntusu (gri tonlama)
    # 255: Esik ustunde kalanlara atanacak deger (beyaz)
    # ADAPTIVE_THRESH_GAUSSIAN_C: Gaussian agirlikli ortalama
    #   - Merkeze yakin pikseller daha agirlikli
    #   - ADAPTIVE_THRESH_MEAN_C: Basit ortalama (alternatif)
    # THRESH_BINARY_INV: Tersine cevirme
    #   - Esik alti = beyaz, esik ustu = siyah
    #   - Koyu metin/kenarlar beyaz olur
    # 11: Blok boyutu (tek sayi olmali)
    #   - Her piksel icin 11x11 komsuluk kullanilir
    #   - Buyuk deger = daha yumusak gecisler
    # 2: C sabiti
    #   - Hesaplanan ortalamadan cikarilir
    #   - Hassasiyeti ayarlar
    esikli = cv2.adaptiveThreshold(
        bulanik,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # -----------------------------------------
    # Morfolojik Closing
    # -----------------------------------------
    # Closing = Dilation + Erosion
    # Kucuk delikleri kapatir, ana sekli korur
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kapali = cv2.morphologyEx(esikli, cv2.MORPH_CLOSE, kernel)
    
    return kapali


# =============================================================================
# ADIM 3: KONTUR BULMA VE FILTRELEME
# =============================================================================

def kontur_bul(kenar_goruntusu: np.ndarray, 
               orijinal_boyut: tuple = None) -> np.ndarray:
    """
    Kenar goruntusunden belge konturunu (4 kose) bulur.
    
    KONTUR NEDIR?
    -------------
    Kontur, ayni renk veya yogunluga sahip surekli noktalar dizisidir.
    Gorsel olarak nesnenin siniri veya cercevesidir.
    
    ALGORITMA:
    ----------
    1. Tum Konturlari Bul:
       - findContours ile binary goruntudeki tum konturlar
       - RETR_EXTERNAL: Sadece en dis konturlar (ic ice olanlar haric)
       - CHAIN_APPROX_SIMPLE: Sadece kose noktalarini sakla
    
    2. Alana Gore Sirala:
       - contourArea ile her konturun alani hesaplanir
       - Buyukten kucuge siralama
       - En buyuk kontur genellikle belge
    
    3. Dortgene Yaklastirma (Polygon Approximation):
       - Douglas-Peucker algoritmasi (approxPolyDP)
       - Konturlari daha az noktali cokgenlere indirger
       - epsilon: Tolerans degeri
       - Amac: 4 koseli sekil bulmak
    
    4. Gecerlilik Kontrolleri:
       - Konveks mi? (Dis bukey)
       - Yeterince buyuk mu? (Min. alan kontrolu)
       - 4 kose mi?
    
    DOUGLAS-PEUCKER ALGORITMASI:
    ----------------------------
    Bir egriyi daha az noktali hale getirme algoritmasi.
    
    1. Baslangic ve bitis noktalari arasina dogru ciz
    2. En uzak noktayi bul
    3. Uzaklik > epsilon ise o noktada bol ve tekrarla
    4. Uzaklik <= epsilon ise aradaki noktalari at
    
    PARAMETRELER:
    -------------
    kenar_goruntusu : np.ndarray
        Ikili kenar goruntusu (Canny veya adaptif esikleme ciktisi)
    
    orijinal_boyut : tuple, optional
        (yukseklik, genislik) - minimum alan kontrolu icin
        Belirtilmezse kenar goruntusunun boyutu kullanilir
    
    DONDURUR:
    ---------
    np.ndarray veya None
        4x2 boyutunda kose koordinatlari dizisi.
        Bulunamazsa None.
    """
    
    # -----------------------------------------
    # Konturlari Bul
    # -----------------------------------------
    # findContours parametreleri:
    # - RETR_EXTERNAL: Sadece en dis konturlar
    #   (RETR_TREE: Tum hiyerarsi, RETR_LIST: Hepsi duz liste)
    # - CHAIN_APPROX_SIMPLE: Gereksiz noktalari atla
    #   (Yatay/dikey/capraz cizgilerde sadece uc noktalar)
    #
    # Donus degerleri:
    # - konturlar: Kontur listesi (her biri numpy array)
    # - _: Hiyerarsi bilgisi (burada kullanilmiyor)
    konturlar, _ = cv2.findContours(
        kenar_goruntusu.copy(),  # Kopya kullan (orijinali degistirme)
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Kontur bulunamadiysa None don
    if len(konturlar) == 0:
        return None
    
    # -----------------------------------------
    # Konturlari Alana Gore Sirala
    # -----------------------------------------
    # cv2.contourArea: Konturun kapladigi alani hesaplar
    # reverse=True: Buyukten kucuge siralama
    # En buyuk kontur muhtemelen belge
    konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)
    
    # -----------------------------------------
    # Minimum Alan Hesapla
    # -----------------------------------------
    # Cok kucuk konturlar belge olamaz (gurultu olabilir)
    # Goruntu alaninin %5'inden kucuk konturlari reddet
    if orijinal_boyut:
        min_alan = orijinal_boyut[0] * orijinal_boyut[1] * 0.05
    else:
        min_alan = kenar_goruntusu.shape[0] * kenar_goruntusu.shape[1] * 0.05
    
    # -----------------------------------------
    # En Buyuk 10 Konturu Incele
    # -----------------------------------------
    # Sadece en buyuk 10 kontura bakmak yeterli
    # (kucuk konturlar belge olamaz)
    for kontur in konturlar[:10]:
        alan = cv2.contourArea(kontur)
        
        # Cok kucuk konturlari atla
        if alan < min_alan:
            continue
        
        # -----------------------------------------
        # Kontur Cevresini Hesapla
        # -----------------------------------------
        # arcLength: Kontur cevresinin uzunlugu
        # True: Kapali kontur (baslangic = bitis)
        cevre = cv2.arcLength(kontur, True)
        
        # -----------------------------------------
        # Dortgene Yaklastir (Polygon Approximation)
        # -----------------------------------------
        # Farkli epsilon degerleri dene (hassasiyet ayari)
        # epsilon = cevre * carpan
        # Dusuk carpan = daha hassas (daha fazla nokta)
        # Yuksek carpan = daha kaba (daha az nokta)
        for epsilon_carpan in [0.02, 0.03, 0.04, 0.05]:
            epsilon = epsilon_carpan * cevre
            
            # approxPolyDP: Douglas-Peucker algoritmasi
            # Konturu daha az noktali cokgene indirger
            yaklasik = cv2.approxPolyDP(kontur, epsilon, True)
            
            # -----------------------------------------
            # 4 Koseli Sekil Kontrolu
            # -----------------------------------------
            if len(yaklasik) == 4:
                # Konveks mi kontrol et
                # Konveks: Tum ic acilar 180 dereceden kucuk
                # Ic bukey sekiller belge olamaz
                if cv2.isContourConvex(yaklasik):
                    # Basarili! 4 koseyi dondur
                    # reshape(4, 2): 4 nokta, her biri (x, y)
                    return yaklasik.reshape(4, 2).astype(np.float32)
    
    # Hicbir uygun kontur bulunamadi
    return None


def belge_tespit(goruntu: np.ndarray) -> np.ndarray:
    """
    Ana belge tespiti fonksiyonu - birden fazla yontemi dener.
    
    STRATEJI:
    ---------
    Tek bir yontem her zaman calismaz. Bu fonksiyon farkli
    yontemleri sirayla dener ve ilk basarili sonucu dondurur.
    
    DENENEN YONTEMLER:
    ------------------
    1. Canny Edge Detection + Morfoloji:
       - En yaygin ve genellikle basarili yontem
       - Iyi aydinlatilmis, net goruntuler icin ideal
    
    2. Adaptif Esikleme:
       - Degisen aydinlatma kosullarinda etkili
       - Dusuk kontrastli goruntulerde faydali
    
    3. Farkli Canny Parametreleri:
       - Alt/ust esik degerleri degistirilerek
       - Farkli kontrast seviyeleri icin
       - [(30, 100), (75, 200), (100, 250)]
    
    PARAMETRELER:
    -------------
    goruntu : np.ndarray
        BGR formatinda giris goruntusu
    
    DONDURUR:
    ---------
    np.ndarray veya None
        4x2 boyutunda kose koordinatlari.
        Tum yontemler basarisiz olursa None.
    """
    
    yukseklik, genislik = goruntu.shape[:2]
    
    # -----------------------------------------
    # Yontem 1: Canny Edge Detection
    # -----------------------------------------
    # Varsayilan parametrelerle kenar tespiti
    kenarlar = kenar_tespit(goruntu)
    kontur = kontur_bul(kenarlar, (yukseklik, genislik))
    
    if kontur is not None:
        return kontur  # Basarili!
    
    # -----------------------------------------
    # Yontem 2: Adaptif Esikleme
    # -----------------------------------------
    # Canny basarisiz olduysa adaptif esikleme dene
    esikli = adaptif_esikleme(goruntu)
    kontur = kontur_bul(esikli, (yukseklik, genislik))
    
    if kontur is not None:
        return kontur  # Basarili!
    
    # -----------------------------------------
    # Yontem 3: Farkli Canny Parametreleri
    # -----------------------------------------
    # Farkli esik degerlerini dene
    gri = on_isleme(goruntu)
    
    # Her parametre cifti icin dene
    for alt_esik, ust_esik in [(30, 100), (75, 200), (100, 250)]:
        # Farkli esiklerle Canny
        kenarlar = cv2.Canny(gri, alt_esik, ust_esik)
        
        # Morfolojik guclendirme
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kenarlar = cv2.dilate(kenarlar, kernel, iterations=2)
        
        kontur = kontur_bul(kenarlar, (yukseklik, genislik))
        
        if kontur is not None:
            return kontur  # Basarili!
    
    # Hicbir yontem basarili olmadi
    return None


# =============================================================================
# ADIM 4: KOSE SIRALAMA
# =============================================================================

def koseler_sirala(koseler: np.ndarray) -> np.ndarray:
    """
    4 koseyi saat yonunde siralar: [sol-ust, sag-ust, sag-alt, sol-alt]
    
    NEDEN SIRALAMA GEREKLI?
    -----------------------
    Perspektif donusumu icin kaynak ve hedef noktalarinin
    ayni sırada olması gerekir. findContours rastgele sirada dondurur.
    
    SIRALAMA MANTIĞI:
    -----------------
    Bir goruntude koordinat sistemi:
    
        (0,0) -----> X (genislik)
          |
          |
          V
          Y (yukseklik)
    
    Bu sisteme gore:
    
    1. Sol-Ust Kose:
       - En kucuk x + y toplami
       - Ornegin: (10, 20) -> toplam = 30
    
    2. Sag-Alt Kose:
       - En buyuk x + y toplami
       - Ornegin: (200, 300) -> toplam = 500
    
    3. Sag-Ust Kose:
       - En kucuk y - x farki (y kucuk, x buyuk)
       - Ornegin: (200, 20) -> fark = 20 - 200 = -180
    
    4. Sol-Alt Kose:
       - En buyuk y - x farki (y buyuk, x kucuk)
       - Ornegin: (10, 300) -> fark = 300 - 10 = 290
    
    GORSEL ACIKLAMA:
    ----------------
       0(Sol-Ust)--------1(Sag-Ust)
            |                |
            |                |
            |                |
       3(Sol-Alt)--------2(Sag-Alt)
    
    PARAMETRELER:
    -------------
    koseler : np.ndarray
        4x2 boyutunda kose koordinatlari.
        Herhangi bir sirada olabilir.
    
    DONDURUR:
    ---------
    np.ndarray
        4x2 boyutunda siralanmis kose koordinatlari.
        Sira: [sol-ust, sag-ust, sag-alt, sol-alt]
        Tip: float32 (perspektif donusumu icin gerekli)
    """
    
    # Sonuc dizisini olustur (4 nokta, her biri 2 koordinat)
    siralanmis = np.zeros((4, 2), dtype=np.float32)
    
    # -----------------------------------------
    # Toplam Hesapla (x + y)
    # -----------------------------------------
    # axis=1: Her satir icin toplam (her nokta icin x+y)
    # Ornek: [[10,20], [200,20], [200,300], [10,300]]
    #        -> [30, 220, 500, 310]
    toplam = koseler.sum(axis=1)
    
    # -----------------------------------------
    # Fark Hesapla (y - x)
    # -----------------------------------------
    # np.diff: Ardisik elemanlar arasindaki fark
    # axis=1: Satir boyunca (y - x hesaplar)
    # flatten(): 2D -> 1D
    fark = np.diff(koseler, axis=1).flatten()
    
    # -----------------------------------------
    # Koseleri Yerlesitir
    # -----------------------------------------
    
    # Sol-Ust: x+y toplami en kucuk
    # argmin: En kucuk degerin indeksini verir
    siralanmis[0] = koseler[np.argmin(toplam)]
    
    # Sag-Alt: x+y toplami en buyuk
    # argmax: En buyuk degerin indeksini verir
    siralanmis[2] = koseler[np.argmax(toplam)]
    
    # Sag-Ust: y-x farki en kucuk (y kucuk, x buyuk = negatif fark)
    siralanmis[1] = koseler[np.argmin(fark)]
    
    # Sol-Alt: y-x farki en buyuk (y buyuk, x kucuk = pozitif fark)
    siralanmis[3] = koseler[np.argmax(fark)]
    
    return siralanmis


# =============================================================================
# ADIM 5: PERSPEKTIF DUZELTME (HOMOGRAPHY)
# =============================================================================

def perspektif_duzelt(goruntu: np.ndarray, koseler: np.ndarray) -> np.ndarray:
    """
    Perspektif donusumu (Homography) uygulayarak belgeyi duzlestirir.
    
    HOMOGRAFI NEDIR?
    ----------------
    Bir duzlemdeki noktalari baska bir duzleme eslestiren 3x3'luk matris.
    
    Gercek Hayat Ornegi:
    - Bir poster egik acidan fotograflandiginda yamuk gorunur
    - Homografi ile posterin duz haline donusturulur
    - "Kus bakisi" (bird's eye view) olusturulur
    
    MATEMATIKSEL TEMEL:
    -------------------
    Homojen koordinatlarda donusum:
    
    [x']   [h11 h12 h13]   [x]
    [y'] = [h21 h22 h23] * [y]
    [w']   [h31 h32 h33]   [1]
    
    Kartezyen koordinatlar: (x'/w', y'/w')
    
    4 NOKTA YETERLILIGI:
    --------------------
    - Her nokta 2 denklem saglar (x ve y icin)
    - 4 nokta = 8 denklem
    - Homografi matrisi 8 bilinmeyen icerir (h33=1 normalize)
    - 4 nokta tam cozum icin yeterli
    
    CIKTI BOYUTU HESABI:
    --------------------
    Cikti goruntusunun boyutlari, belgenin gercek oranlarini korur:
    
    1. Ust kenar genisligi: sqrt((sag_ust - sol_ust)^2)
    2. Alt kenar genisligi: sqrt((sag_alt - sol_alt)^2)
    3. Genislik = max(ust_genislik, alt_genislik)
    
    4. Sol kenar yuksekligi: sqrt((sol_alt - sol_ust)^2)
    5. Sag kenar yuksekligi: sqrt((sag_alt - sag_ust)^2)
    6. Yukseklik = max(sol_yukseklik, sag_yukseklik)
    
    PARAMETRELER:
    -------------
    goruntu : np.ndarray
        Orijinal BGR goruntusu
    
    koseler : np.ndarray
        Siralanmis 4x2 kose koordinatlari.
        Sira: [sol-ust, sag-ust, sag-alt, sol-alt]
    
    DONDURUR:
    ---------
    np.ndarray
        Perspektifi duzeltilmis goruntu.
        Boyut: (hesaplanan_yukseklik, hesaplanan_genislik, 3)
    """
    
    # Koseleri ayri degiskenlere ata (okunabilirlik icin)
    sol_ust, sag_ust, sag_alt, sol_alt = koseler
    
    # =========================================================================
    # CIKTI BOYUTLARINI HESAPLA
    # =========================================================================
    # Euclidean mesafe (iki nokta arasi duz cizgi mesafesi)
    # Formul: sqrt((x2-x1)^2 + (y2-y1)^2)
    
    # -----------------------------------------
    # Genislik Hesabi
    # -----------------------------------------
    # Ust kenar genisligi
    ust_genislik = np.sqrt(
        (sag_ust[0] - sol_ust[0])**2 + 
        (sag_ust[1] - sol_ust[1])**2
    )
    
    # Alt kenar genisligi
    alt_genislik = np.sqrt(
        (sag_alt[0] - sol_alt[0])**2 + 
        (sag_alt[1] - sol_alt[1])**2
    )
    
    # Maksimum genislik (daha genis olan kenari kullan)
    genislik = max(int(ust_genislik), int(alt_genislik))
    
    # -----------------------------------------
    # Yukseklik Hesabi
    # -----------------------------------------
    # Sol kenar yuksekligi
    sol_yukseklik = np.sqrt(
        (sol_alt[0] - sol_ust[0])**2 + 
        (sol_alt[1] - sol_ust[1])**2
    )
    
    # Sag kenar yuksekligi
    sag_yukseklik = np.sqrt(
        (sag_alt[0] - sag_ust[0])**2 + 
        (sag_alt[1] - sag_ust[1])**2
    )
    
    # Maksimum yukseklik
    yukseklik = max(int(sol_yukseklik), int(sag_yukseklik))
    
    # =========================================================================
    # HEDEF NOKTALARI TANIMLA
    # =========================================================================
    # Cikti goruntusu duz bir dikdortgen olacak
    # Kose siralama: [sol-ust, sag-ust, sag-alt, sol-alt]
    hedef = np.array([
        [0, 0],                        # Sol-ust: (0, 0)
        [genislik - 1, 0],             # Sag-ust: (genislik-1, 0)
        [genislik - 1, yukseklik - 1], # Sag-alt: (genislik-1, yukseklik-1)
        [0, yukseklik - 1]             # Sol-alt: (0, yukseklik-1)
    ], dtype=np.float32)
    
    # =========================================================================
    # HOMOGRAFI MATRISINI HESAPLA
    # =========================================================================
    # getPerspectiveTransform: 4 kaynak ve 4 hedef noktasindan
    # 3x3 homografi matrisini hesaplar
    #
    # Kaynak: Orijinal goruntudeki belge koseleri (egik/yamuk)
    # Hedef: Cikti goruntusundeki koseler (duz dikdortgen)
    matris = cv2.getPerspectiveTransform(koseler, hedef)
    
    # =========================================================================
    # DONUSUMU UYGULA
    # =========================================================================
    # warpPerspective: Her pikseli homografi matrisine gore
    # yeni konumuna tasir
    #
    # Parametreler:
    # - goruntu: Kaynak goruntu
    # - matris: 3x3 donusum matrisi
    # - (genislik, yukseklik): Cikti boyutu
    duzeltilmis = cv2.warpPerspective(goruntu, matris, (genislik, yukseklik))
    
    return duzeltilmis


# =============================================================================
# ADIM 6: GORUNTU IYILESTIRME (IMAGE ENHANCEMENT)
# =============================================================================

def goruntu_iyilestir(goruntu: np.ndarray, 
                      siyah_beyaz: bool = True) -> np.ndarray:
    """
    Goruntuyu taranmis belge gorunumune kavusturur.
    
    AMAC:
    -----
    Belge fotografini masaustu tarayiciyla taranmis gibi gostermek.
    
    SIYAH-BEYAZ MOD (Varsayilan):
    -----------------------------
    Tipik taranmis belge gorunumu. Metin ve cizgiler net, arka plan temiz.
    
    1. Gri Tonlama Donusumu:
       - 3 kanal (BGR) -> 1 kanal (Gray)
    
    2. Adaptif Esikleme:
       - Her piksel icin lokal esik
       - ADAPTIVE_THRESH_GAUSSIAN_C: Gaussian agirlikli ortalama
       - THRESH_BINARY: Esik ustu beyaz, alti siyah
       - Blok boyutu 11: 11x11 komsuluk
       - C=2: Ortalamadan cikartilan sabit
    
    3. Median Blur:
       - Tuz-biber gurultusunu temizler
       - Kernel boyutu 3: 3x3 pencere
       - Kenar koruyucu yumusatma
    
    RENKLI MOD (siyah_beyaz=False):
    --------------------------------
    Orijinal renkleri koruyarak kontrasti iyilestirir.
    
    1. LAB Renk Uzayina Donustur:
       - L: Lightness (parlaklik)
       - A: Yesil-Kirmizi ekseni
       - B: Mavi-Sari ekseni
    
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization):
       - Sadece L kanalina uygula
       - Lokal kontrast iyilestirme
       - clipLimit: Kontrast siniri (asiri amplifikasyonu onler)
       - tileGridSize: Parcalama boyutu
    
    3. BGR'ye Geri Donustur
    
    PARAMETRELER:
    -------------
    goruntu : np.ndarray
        Giris goruntusu (BGR formati)
    
    siyah_beyaz : bool, default=True
        True: Siyah-beyaz taranmis gorunum
        False: Renkli kontrast iyilestirmesi
    
    DONDURUR:
    ---------
    np.ndarray
        Iyilestirilmis goruntu (BGR formati)
    """
    
    if not siyah_beyaz:
        # -----------------------------------------
        # RENKLI MOD: CLAHE ile Kontrast Iyilestirme
        # -----------------------------------------
        
        # BGR -> LAB donusumu
        # LAB uzayi parlaklik ve rengi ayirir
        lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
        
        # Kanallari ayir
        l, a, b = cv2.split(lab)
        
        # CLAHE nesnesi olustur
        # clipLimit=2.0: Histogram kutusu basina maksimum yukseklik
        # tileGridSize=(8,8): Goruntu 8x8 parcaya bolunur
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # CLAHE'yi sadece L (parlaklik) kanalina uygula
        # Renkler etkilenmez
        l = clahe.apply(l)
        
        # Kanallari birlestir
        lab = cv2.merge([l, a, b])
        
        # LAB -> BGR donusumu
        sonuc = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return sonuc
    
    # -----------------------------------------
    # SIYAH-BEYAZ MOD: Adaptif Esikleme
    # -----------------------------------------
    
    # Gri tonlamaya donustur
    if len(goruntu.shape) == 3:
        # Renkli goruntu
        gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    else:
        # Zaten gri tonlama
        gri = goruntu.copy()
    
    # Adaptif esikleme
    # - 255: Esik ustunde beyaz
    # - ADAPTIVE_THRESH_GAUSSIAN_C: Gaussian agirlikli lokal ortalama
    # - THRESH_BINARY: Normal esikleme (BINARY_INV degil)
    # - 11: 11x11 komsuluk
    # - 2: C sabiti
    esikli = cv2.adaptiveThreshold(
        gri,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Median filtresi ile gurultu azalt
    # Tuz-biber (salt-and-pepper) gurultusune karsi etkili
    # 3: 3x3 kernel
    temiz = cv2.medianBlur(esikli, 3)
    
    # Gri tonlamayi BGR'ye donustur (tutarlilik icin)
    # Tek kanaldan 3 kanala (ayni deger 3 kez)
    sonuc = cv2.cvtColor(temiz, cv2.COLOR_GRAY2BGR)
    
    return sonuc


# =============================================================================
# PDF OLUSTURMA
# =============================================================================

def pdf_olustur(goruntu: np.ndarray, dosya_yolu: str) -> str:
    """
    Goruntuyu PDF dosyasi olarak kaydeder.
    
    PDF OLUSTURMA SURECI:
    ---------------------
    1. Dosya uzantisi kontrolu (.pdf ekleme)
    2. Gecici JPEG dosyasi olusturma
    3. PDF'e donusturme (img2pdf veya Pillow)
    4. Gecici dosyayi temizleme
    
    NEDEN GECICI DOSYA?
    -------------------
    img2pdf kutuphanesi dosya yolu bekler, numpy array degil.
    Bu nedenle once goruntuyu gecici JPEG'e kaydediyoruz.
    
    KUTUPHANE SECIMI:
    -----------------
    1. img2pdf (tercih edilen):
       - A4 boyutunda profesyonel PDF
       - Kayipsiz goruntu kalitesi
       - Sayfa boyutu kontrolu
    
    2. Pillow (yedek):
       - img2pdf yoksa kullanilir
       - Daha basit ama calisir
    
    PARAMETRELER:
    -------------
    goruntu : np.ndarray
        Kaydedilecek goruntu (BGR formati)
    
    dosya_yolu : str
        Cikti PDF dosyasinin yolu
    
    DONDURUR:
    ---------
    str
        Olusturulan PDF dosyasinin tam yolu
    """
    
    # Dosya uzantisi kontrolu
    if not dosya_yolu.lower().endswith('.pdf'):
        dosya_yolu += '.pdf'
    
    # -----------------------------------------
    # Gecici JPEG Dosyasi Olustur
    # -----------------------------------------
    # NamedTemporaryFile: Benzersiz isimli gecici dosya
    # suffix='.jpg': .jpg uzantili
    # delete=False: Otomatik silme kapali (manuel silecegiz)
    temp_dosya = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_yol = temp_dosya.name
    temp_dosya.close()  # Dosyayi kapat (Windows uyumlulugu)
    
    try:
        # -----------------------------------------
        # Goruntuyu JPEG Olarak Kaydet
        # -----------------------------------------
        # IMWRITE_JPEG_QUALITY: JPEG kalitesi (0-100)
        # 95: Yuksek kalite, makul dosya boyutu
        cv2.imwrite(temp_yol, goruntu, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # -----------------------------------------
        # PDF'e Donustur
        # -----------------------------------------
        try:
            # img2pdf kutuphanesini dene
            import img2pdf
            
            with open(dosya_yolu, 'wb') as f:
                # A4 boyutunda PDF olustur
                # A4: 210mm x 297mm
                # mm_to_pt: Milimetreyi point'e cevir (1 pt = 1/72 inch)
                a4_boyut = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
                
                # Layout fonksiyonu: Goruntu boyutunu ayarlar
                layout = img2pdf.get_layout_fun(a4_boyut)
                
                # PDF icerigini yaz
                f.write(img2pdf.convert(temp_yol, layout_fun=layout))
                
        except ImportError:
            # img2pdf yoksa Pillow kullan
            from PIL import Image
            
            # JPEG'i ac
            img = Image.open(temp_yol)
            
            # RGB moduna donustur (PDF icin gerekli)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # PDF olarak kaydet
            # resolution: DPI (dots per inch)
            img.save(dosya_yolu, 'PDF', resolution=100.0)
            
    finally:
        # -----------------------------------------
        # Gecici Dosyayi Temizle
        # -----------------------------------------
        # finally blogu hata olsa bile calisir
        if os.path.exists(temp_yol):
            os.remove(temp_yol)
    
    return dosya_yolu


def coklu_sayfa_pdf(goruntuler: list, dosya_yolu: str) -> str:
    """
    Birden fazla goruntuyu tek bir PDF dosyasina birlestirir.
    
    COK SAYFALI PDF:
    ----------------
    Birden fazla belge sayfasini tek PDF'de birlestirmek icin.
    Her goruntu ayri bir sayfa olur.
    
    PARAMETRELER:
    -------------
    goruntuler : list
        Goruntu dizileri listesi.
        Her eleman: np.ndarray (BGR formati)
    
    dosya_yolu : str
        Cikti PDF dosyasinin yolu
    
    DONDURUR:
    ---------
    str
        Olusturulan PDF dosyasinin tam yolu
    
    ORNEK:
    ------
    >>> sayfalar = [sayfa1, sayfa2, sayfa3]
    >>> pdf_yolu = coklu_sayfa_pdf(sayfalar, "belge.pdf")
    """
    
    from PIL import Image
    
    # Dosya uzantisi kontrolu
    if not dosya_yolu.lower().endswith('.pdf'):
        dosya_yolu += '.pdf'
    
    # Tum goruntuleri PIL Image'e donustur
    pil_goruntuler = []
    
    for goruntu in goruntuler:
        # BGR -> RGB donusumu
        if len(goruntu.shape) == 3:
            rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
        else:
            # Gri tonlama -> RGB
            rgb = cv2.cvtColor(goruntu, cv2.COLOR_GRAY2RGB)
        
        # NumPy array -> PIL Image
        pil_img = Image.fromarray(rgb)
        pil_goruntuler.append(pil_img)
    
    # PDF olustur
    if pil_goruntuler:
        # Ilk sayfa
        ilk_sayfa = pil_goruntuler[0]
        
        # Diger sayfalar (varsa)
        diger_sayfalar = pil_goruntuler[1:] if len(pil_goruntuler) > 1 else []
        
        # Cok sayfali PDF olarak kaydet
        ilk_sayfa.save(
            dosya_yolu, 
            'PDF', 
            resolution=100.0,      # DPI
            save_all=True,         # Tum sayfalari kaydet
            append_images=diger_sayfalar  # Ek sayfalar
        )
    
    return dosya_yolu


# =============================================================================
# DEBUG / GORSELESTIRME FONKSIYONLARI
# =============================================================================

def debug_goruntule(goruntu: np.ndarray, 
                    koseler: np.ndarray = None,
                    baslik: str = "Debug") -> np.ndarray:
    """
    Debug amacli goruntu uzerinde koseleri isaretler.
    
    Bu fonksiyon gelistirme ve hata ayiklama sirasinda
    kullanilmak uzerine tasarlanmistir.
    
    ISARETLEME DETAYLARI:
    ---------------------
    - Her kose icin yesil daire (yaricap 10px)
    - Her kosenin yaninda numara (1-4)
    - Koseleri birlestiren yesil cizgiler
    
    PARAMETRELER:
    -------------
    goruntu : np.ndarray
        Isaretlenecek giris goruntusu
    
    koseler : np.ndarray, optional
        4x2 boyutunda kose koordinatlari.
        None ise sadece goruntu kopyasi doner.
    
    baslik : str, default="Debug"
        cv2.imshow kullanilirsa pencere basligi.
        (Bu fonksiyon icinde kullanilmiyor ama gelecekte eklenebilir)
    
    DONDURUR:
    ---------
    np.ndarray
        Isaretlenmis goruntunun kopyasi.
        Orijinal goruntu degistirilmez.
    """
    
    # Orijinali korumak icin kopya olustur
    sonuc = goruntu.copy()
    
    if koseler is not None:
        # Her kose icin
        for i, kose in enumerate(koseler):
            # Koordinatlari tam sayiya cevir
            x, y = int(kose[0]), int(kose[1])
            
            # -----------------------------------------
            # Yesil Daire Ciz
            # -----------------------------------------
            # circle(goruntu, merkez, yaricap, renk, kalinlik)
            # kalinlik=-1: Icini doldur
            cv2.circle(
                sonuc, 
                (x, y),           # Merkez noktasi
                10,               # Yaricap (piksel)
                (0, 255, 0),      # Renk: Yesil (BGR)
                -1                # Icini doldur
            )
            
            # -----------------------------------------
            # Numara Yaz
            # -----------------------------------------
            # putText(goruntu, metin, konum, font, olcek, renk, kalinlik)
            cv2.putText(
                sonuc, 
                str(i + 1),                     # Numara (1'den baslar)
                (x + 15, y + 5),                # Konum (biraz saga kaydir)
                cv2.FONT_HERSHEY_SIMPLEX,       # Font tipi
                0.8,                            # Font olcegi
                (0, 0, 255),                    # Renk: Kirmizi (BGR)
                2                               # Kalinlik
            )
        
        # -----------------------------------------
        # Koseleri Birlestiren Cizgiler
        # -----------------------------------------
        # Koselerden kapalı poligon olustur
        pts = koseler.astype(int).reshape((-1, 1, 2))
        
        # polylines(goruntu, [noktalar], kapali_mi, renk, kalinlik)
        cv2.polylines(
            sonuc, 
            [pts],        # Nokta dizisi
            True,         # Kapali poligon (son nokta ilke baglanir)
            (0, 255, 0),  # Renk: Yesil
            2             # Kalinlik
        )
    
    return sonuc
