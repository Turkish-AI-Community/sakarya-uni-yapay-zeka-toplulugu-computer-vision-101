"""
================================================================================
Dosya Tarayici (Document Scanner) - Komut Satiri Uygulamasi
================================================================================

Bu dosya, belge tarama uygulamasinin komut satiri arayuzunu saglar.
Terminalden dogrudan kullanilabilir.

BELGE TARAMA NEDIR?
-------------------
Belge tarama, bir kagit belgenin fotografini alip onu "taranmis" gibi
duz ve okunakli bir gorunume kavusturma islemidir.

Ornegin:
- Telefon ile egik cekilmis bir belge fotografini
- Sanki masaustu tarayicidan gecirmis gibi duz ve temiz hale getirir

TARAMA PIPELINE'I (Islem Adimlari):
-----------------------------------
1. Goruntu Yukleme
   - Dosyadan BGR formatinda okuma
   
2. Belge Tespiti (Document Detection)
   - Kenar tespiti (Canny Edge Detection)
   - Kontur bulma (Contour Detection)
   - 4 koseli sekil filtreleme
   
3. Kose Siralama
   - Bulunan 4 koseyi saat yonunde siralama
   - [sol-ust, sag-ust, sag-alt, sol-alt] sirasi
   
4. Perspektif Duzeltme (Homography)
   - 4 nokta kullanarak 3x3 donusum matrisi hesaplama
   - Egik goruntuyu duz dikdortgene donusturme
   - "Kus bakisi" (bird's eye view) olusturma
   
5. Goruntu Iyilestirme
   - Adaptif esikleme (taranmis gorunum)
   - Gurultu azaltma
   
6. PDF Olusturma
   - A4 boyutunda PDF ciktisi

KULLANIM ORNEKLERI:
-------------------
# Basit kullanim - varsayilan PDF adi
$ python scanner.py belge.jpg

# Cikti dosya adi belirtme
$ python scanner.py belge.jpg -o taranmis_belge.pdf

# Islem adimlarini gorsellestirme (egitim amacli)
$ python scanner.py belge.jpg --onizleme

GEREKSINIMLER:
--------------
- Python 3.8+
- OpenCV (cv2)
- NumPy
- img2pdf veya Pillow (PDF icin)

Yazar: SAU Yapay Zeka & Bilgisayarli Goru Toplulugu
================================================================================
"""

# =============================================================================
# KUTUPHANELERIN ICE AKTARILMASI
# =============================================================================

# OpenCV: Goruntu isleme kutuphanesi
# - Goruntu okuma/yazma
# - Renk donusumleri
# - Kenar tespiti, morfolojik islemler
# - Perspektif donusumu
import cv2

# NumPy: Sayisal hesaplama
# - Goruntu verileri N-boyutlu array olarak temsil edilir
# - Matematiksel operasyonlar icin kullanilir
import numpy as np

# Yerel modullerden goruntu isleme fonksiyonlarini ice aktar
from utils import (
    kenar_tespit,       # Canny + morfoloji ile kenar tespiti
    belge_tespit,       # Belge konturunun (4 kose) tespiti
    koseler_sirala,     # Koseleri saat yonunde siralama
    perspektif_duzelt,  # Homografi ile perspektif duzeltme
    goruntu_iyilestir,  # Adaptif esikleme ile taranmis gorunum
    pdf_olustur,        # PDF formatinda kaydetme
    debug_goruntule     # Debug amacli isaretleme
)

# argparse: Komut satiri argumanlari islemek icin standart Python modulu
# Kullanici terminal'de calistirirken parametre vermesini saglar
import argparse

# os: Isletim sistemi islemleri
# - Dosya varlik kontrolu
# - Dosya yolu islemleri
import os


# =============================================================================
# ANA TARAMA FONKSIYONU
# =============================================================================

def belge_tara(goruntu_yolu: str, cikti_yolu: str = None) -> str:
    """
    Belge fotografini taranmis PDF'e donusturur.
    
    Bu fonksiyon, belge tarama isleminin tum pipeline'ini yurutur.
    Her adimda ne yapildigini konsola yazdirir.
    
    PIPELINE ADIMLARI:
    ------------------
    1. Goruntu Yukleme:
       - cv2.imread() ile BGR formatinda yukleme
       - Dosya varlik kontrolu
       
    2. Belge Tespiti:
       - Birden fazla kenar tespiti yontemi denenir
       - 4 koseli konveks sekil aranir
       - Bulunamazsa tum goruntu kullanilir
       
    3. Kose Siralama:
       - Toplam ve fark metoduyla siralama
       - Perspektif donusumu icin kritik
       
    4. Perspektif Duzeltme:
       - getPerspectiveTransform ile 3x3 matris
       - warpPerspective ile donusum uygulama
       
    5. Goruntu Iyilestirme:
       - adaptiveThreshold ile siyah-beyaz
       - medianBlur ile gurultu azaltma
       
    6. PDF Olusturma:
       - img2pdf veya Pillow kullanarak
       - A4 boyutunda cikti
    
    PARAMETRELER:
    -------------
    goruntu_yolu : str
        Taranacak belgenin dosya yolu.
        Desteklenen formatlar: JPG, PNG, BMP, TIFF, vb.
        
    cikti_yolu : str, optional
        Olusturulacak PDF dosyasinin yolu.
        Belirtilmezse: "{orijinal_adi}_taranmis.pdf" olarak olusturulur.
    
    DONDURUR:
    ---------
    str
        Olusturulan PDF dosyasinin tam yolu.
    
    HATALAR:
    --------
    FileNotFoundError
        Goruntu dosyasi bulunamazsa.
    
    ORNEK KULLANIM:
    ---------------
    >>> pdf_yolu = belge_tara("belge.jpg")
    >>> print(f"PDF olusturuldu: {pdf_yolu}")
    
    >>> pdf_yolu = belge_tara("belge.jpg", "cikti.pdf")
    """
    
    # =========================================================================
    # ADIM 1: GORUNTUYU YUKLE
    # =========================================================================
    # cv2.imread() fonksiyonu:
    # - Goruntu dosyasini diskten okur
    # - BGR (Blue, Green, Red) renk siralamasinda dondurur
    # - Basarisiz olursa None dondurur
    
    print(f"[1/6] Goruntu yukleniyor: {goruntu_yolu}")
    orijinal = cv2.imread(goruntu_yolu)
    
    # Dosya bulunamadi veya okunamadi kontrolu
    if orijinal is None:
        raise FileNotFoundError(f"Goruntu dosyasi bulunamadi: {goruntu_yolu}")
    
    # Goruntu boyutlarini al
    # shape: (yukseklik, genislik, kanal_sayisi) formatinda tuple
    # shape[:2]: Sadece yukseklik ve genislik (kanal sayisini atla)
    yukseklik, genislik = orijinal.shape[:2]
    print(f"      Goruntu boyutu: {genislik}x{yukseklik} piksel")
    
    # =========================================================================
    # ADIM 2: BELGE TESPITI (Document Detection)
    # =========================================================================
    # Bu adimda goruntudenki belgenin sinirlarini (4 kose) bulmaya calisiyoruz.
    # 
    # KULLANILAN YONTEMLER (sirasi ile denenir):
    # 1. Canny Edge Detection + Morfolojik islemler
    #    - Kenarlar guclendirilerek kontur bulunur
    # 
    # 2. Adaptif Esikleme
    #    - Dusuk kontrastli goruntulerde daha etkili
    # 
    # 3. Farkli Canny parametreleri
    #    - Alt/ust esik degerleri degistirilerek denenir
    
    print("[2/6] Belge tespit ediliyor...")
    print("      - Kenar tespiti (Canny + Morfoloji)")
    print("      - Kontur analizi")
    print("      - Dortgen filtresi")
    
    # belge_tespit fonksiyonu birden fazla yontemi dener
    # Basarili olursa 4 kose koordinati, basarisizsa None doner
    kontur = belge_tespit(orijinal)
    
    if kontur is None:
        # Belge koseleri bulunamadi
        # Fallback: Tum goruntuyu belge olarak kabul et
        print("      UYARI: Belge koseleri tespit edilemedi!")
        print("      Tum goruntu kullanilacak.")
        
        # Goruntunun 4 kosesini manuel belirle
        kontur = np.array([
            [0, 0],                       # Sol-ust kose
            [genislik - 1, 0],            # Sag-ust kose
            [genislik - 1, yukseklik - 1], # Sag-alt kose
            [0, yukseklik - 1]            # Sol-alt kose
        ], dtype=np.float32)
    else:
        print("      Belge basariyla tespit edildi!")
    
    # =========================================================================
    # ADIM 3: KOSE SIRALAMA
    # =========================================================================
    # Perspektif donusumu icin koselerin belirli bir sirada olmasi gerekir.
    # Siralama: [sol-ust, sag-ust, sag-alt, sol-alt] (saat yonunde)
    # 
    # SIRALAMA ALGORITMASI:
    # - Sol-ust: x+y toplami en kucuk
    # - Sag-alt: x+y toplami en buyuk
    # - Sag-ust: y-x farki en kucuk (y kucuk, x buyuk)
    # - Sol-alt: y-x farki en buyuk (y buyuk, x kucuk)
    
    print("[3/6] Koseler siralaniyor...")
    koseler = koseler_sirala(kontur)
    
    # Kose koordinatlarini yazdir (debug icin faydali)
    print(f"      Sol-Ust:  ({koseler[0][0]:.0f}, {koseler[0][1]:.0f})")
    print(f"      Sag-Ust:  ({koseler[1][0]:.0f}, {koseler[1][1]:.0f})")
    print(f"      Sag-Alt:  ({koseler[2][0]:.0f}, {koseler[2][1]:.0f})")
    print(f"      Sol-Alt:  ({koseler[3][0]:.0f}, {koseler[3][1]:.0f})")
    
    # =========================================================================
    # ADIM 4: PERSPEKTIF DUZELTME (Homography)
    # =========================================================================
    # HOMOGRAFI NEDIR?
    # ----------------
    # Homografi, bir duzlemdeki noktalari baska bir duzleme eslestiren
    # 3x3 boyutunda bir donusum matrisidir.
    # 
    # Matematiksel olarak:
    #   [x']   [h11 h12 h13]   [x]
    #   [y'] = [h21 h22 h23] * [y]
    #   [w']   [h31 h32 h33]   [1]
    # 
    # Burada (x,y) kaynak nokta, (x'/w', y'/w') hedef noktadir.
    # 
    # KULLANIM AMACI:
    # - Egik cekilmis belgeyi duz dikdortgene donusturme
    # - "Kus bakisi" (bird's eye view) olusturma
    # - En az 4 nokta gerektirir (4 kose = 4 nokta)
    
    print("[4/6] Perspektif duzeltiliyor (Homography)...")
    duzeltilmis = perspektif_duzelt(orijinal, koseler)
    
    # Duzeltilmis goruntunun boyutlarini yazdir
    duz_yuk, duz_gen = duzeltilmis.shape[:2]
    print(f"      Cikti boyutu: {duz_gen}x{duz_yuk} piksel")
    
    # =========================================================================
    # ADIM 5: GORUNTU IYILESTIRME (Image Enhancement)
    # =========================================================================
    # Bu adimda goruntuyu taranmis belge gibi gostermek icin islemler yapilir.
    # 
    # ADAPTIF ESIKLEME (Adaptive Thresholding):
    # - Her piksel icin lokal esik degeri hesaplanir
    # - Komsu piksellerin ortalamasina gore karar verilir
    # - Degisen aydinlatma kosullarina dayanikli
    # 
    # GURULTU AZALTMA (Noise Reduction):
    # - Median filtresi ile tuz-biber gurultusu giderilir
    # - Kenarlari koruyarak yumusatma yapar
    
    print("[5/6] Goruntu iyilestiriliyor...")
    print("      - Adaptif esikleme")
    print("      - Gurultu azaltma")
    iyilestirilmis = goruntu_iyilestir(duzeltilmis)
    
    # =========================================================================
    # ADIM 6: PDF OLUSTURMA
    # =========================================================================
    # Son adimda islenmis goruntuyu PDF formatinda kaydediyoruz.
    # 
    # PDF OLUSTURMA SURECI:
    # 1. Goruntu gecici JPEG dosyasina kaydedilir
    # 2. img2pdf kutuphanesi ile A4 boyutunda PDF olusturulur
    # 3. img2pdf yoksa Pillow (PIL) kullanilir
    # 4. Gecici JPEG dosyasi silinir
    
    print("[6/6] PDF olusturuluyor...")
    
    # Cikti yolu belirlenmemisse otomatik olustur
    if cikti_yolu is None:
        # Orijinal dosya adindan PDF adi turet
        dosya_adi = os.path.splitext(os.path.basename(goruntu_yolu))[0]
        cikti_yolu = f"{dosya_adi}_taranmis.pdf"
    
    # PDF'i kaydet
    pdf_yolu = pdf_olustur(iyilestirilmis, cikti_yolu)
    print(f"[OK] PDF basariyla olusturuldu: {pdf_yolu}")
    
    return pdf_yolu


# =============================================================================
# ONIZLEME/DEBUG FONKSIYONU
# =============================================================================

def onizleme_goster(goruntu_yolu: str):
    """
    Tarama isleminin her adimini gorsel olarak gosterir.
    
    Bu fonksiyon egitim amaclidir. Belge tarama pipeline'inin
    her adiminin sonucunu 2x2 grid seklinde gosterir.
    
    GOSTERILEN ADIMLAR:
    -------------------
    +-------------------+-------------------+
    | 1. Orijinal       | 2. Kenar Tespiti  |
    |    Goruntu        |    (Canny)        |
    +-------------------+-------------------+
    | 3. Kose Tespiti   | 4. Sonuc          |
    |    (Isaretli)     |    (Iyilestir.)   |
    +-------------------+-------------------+
    
    PARAMETRELER:
    -------------
    goruntu_yolu : str
        Gosterilecek goruntu dosyasinin yolu.
    
    NOTLAR:
    -------
    - ESC tusu veya herhangi bir tusa basilinca pencere kapanir
    - Goruntu buyukse otomatik olarak kucultulur
    - OpenCV GUI kullanir (cv2.imshow)
    """
    
    # Goruntuyu yukle
    orijinal = cv2.imread(goruntu_yolu)
    
    if orijinal is None:
        raise FileNotFoundError(f"Goruntu bulunamadi: {goruntu_yolu}")
    
    # -------------------------------------------------------------------------
    # Goruntu Boyutunu Ayarla
    # -------------------------------------------------------------------------
    # Cok buyuk goruntuleri ekrana sigdirmak icin kucultme
    max_boyut = 600  # Maksimum kenar uzunlugu (piksel)
    yukseklik, genislik = orijinal.shape[:2]
    
    # En buyuk kenar max_boyut'u asiyorsa kucult
    if max(yukseklik, genislik) > max_boyut:
        olcek = max_boyut / max(yukseklik, genislik)
    else:
        olcek = 1.0  # Kucultme gerekmiyor
    
    # Kucultulmus goruntu
    kucuk = cv2.resize(orijinal, None, fx=olcek, fy=olcek)
    
    # -------------------------------------------------------------------------
    # Pipeline Adimlarini Uygula
    # -------------------------------------------------------------------------
    
    # Kenar tespiti (Canny + Morfoloji)
    kenarlar = kenar_tespit(kucuk)
    
    # Belge tespiti (4 kose bul)
    kontur = belge_tespit(kucuk)
    
    # Konturu goruntu uzerine ciz
    konturlu = kucuk.copy()
    
    if kontur is not None:
        # Koseler bulundu - isaretle
        koseler = koseler_sirala(kontur)
        
        # Her kose icin daire ve numara ciz
        for i, kose in enumerate(koseler):
            x, y = int(kose[0]), int(kose[1])
            
            # Yesil daire
            cv2.circle(konturlu, (x, y), 8, (0, 255, 0), -1)
            
            # Kirmizi numara
            cv2.putText(
                konturlu, 
                str(i+1), 
                (x+10, y+5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 255),  # Kirmizi (BGR)
                2
            )
        
        # Koseleri birlestiren cizgiler
        pts = koseler.astype(int).reshape((-1, 1, 2))
        cv2.polylines(konturlu, [pts], True, (0, 255, 0), 2)
        
        # Orijinal boyuta donustur ve perspektif duzelt
        # (Kucuk goruntudeki koseler orijinale gore olceklenir)
        koseler_orijinal = koseler / olcek
        duzeltilmis = perspektif_duzelt(orijinal, koseler_orijinal)
        
        # Sonucu gosterim boyutuna kucult
        duz_kucuk = cv2.resize(duzeltilmis, (kucuk.shape[1], kucuk.shape[0]))
        
        # Goruntu iyilestirme
        iyilestirilmis = goruntu_iyilestir(duz_kucuk)
    else:
        # Koseler bulunamadi - sadece iyilestirme uygula
        iyilestirilmis = goruntu_iyilestir(kucuk)
    
    # -------------------------------------------------------------------------
    # Goruntuleri 2x2 Grid Seklinde Birlestir
    # -------------------------------------------------------------------------
    
    # Kenar goruntusu gri tonlama - 3 kanala cevir (gorselestirme icin)
    kenarlar_renkli = cv2.cvtColor(kenarlar, cv2.COLOR_GRAY2BGR)
    
    # Tum goruntuleri ayni boyuta getir
    hedef_boyut = (kucuk.shape[1], kucuk.shape[0])
    kenarlar_renkli = cv2.resize(kenarlar_renkli, hedef_boyut)
    iyilestirilmis = cv2.resize(iyilestirilmis, hedef_boyut)
    
    # Yatay birlestirme: [Orijinal | Kenarlar]
    ust = np.hstack([kucuk, kenarlar_renkli])
    
    # Yatay birlestirme: [Konturlu | Iyilestirilmis]
    alt = np.hstack([konturlu, iyilestirilmis])
    
    # Dikey birlestirme: Ust + Alt
    birlesik = np.vstack([ust, alt])
    
    # -------------------------------------------------------------------------
    # Basliklari Ekle
    # -------------------------------------------------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    renk = (0, 255, 255)  # Sari (BGR)
    
    # Her karenin sol ustune baslik yaz
    cv2.putText(birlesik, "1. Orijinal", (10, 25), font, 0.6, renk, 2)
    cv2.putText(birlesik, "2. Kenar Tespiti", (kucuk.shape[1] + 10, 25), font, 0.6, renk, 2)
    cv2.putText(birlesik, "3. Kose Tespiti", (10, kucuk.shape[0] + 25), font, 0.6, renk, 2)
    cv2.putText(birlesik, "4. Sonuc", (kucuk.shape[1] + 10, kucuk.shape[0] + 25), font, 0.6, renk, 2)
    
    # -------------------------------------------------------------------------
    # Pencereyi Goster ve Bekle
    # -------------------------------------------------------------------------
    cv2.imshow("Dosya Tarayici - Islem Adimlari", birlesik)
    print("Pencereyi kapatmak icin herhangi bir tusa basin...")
    
    # Herhangi bir tusa basilana kadar bekle
    cv2.waitKey(0)
    
    # Pencereyi kapat
    cv2.destroyAllWindows()


# =============================================================================
# KOMUT SATIRI ARAYUZU (Command Line Interface)
# =============================================================================

def main():
    """
    Komut satiri arayuzu icin ana fonksiyon.
    
    ARGPARSE KULLANIMI:
    -------------------
    argparse, Python'un standart komut satiri islemci modulu.
    
    Sunlari saglar:
    - Argumanlar icin --help otomatik olusturma
    - Zorunlu ve opsiyonel arguman desteği
    - Tip denetimi ve varsayilan degerler
    - Kullanim mesajlari
    
    DESTEKLENEN ARGUMANLAR:
    -----------------------
    goruntu (zorunlu): Taranacak dosya yolu
    -o, --output: Cikti PDF yolu (opsiyonel)
    --onizleme: Debug/onizleme modu (flag)
    
    CIKIS KODLARI:
    --------------
    0: Basarili
    1: Hata
    """
    
    # Arguman parser'i olustur
    # description: Yardim metninde gosterilecek aciklama
    # formatter_class: Formatlama stili (ornegin satir sonlarini koru)
    parser = argparse.ArgumentParser(
        description="Dosya Tarayici - Belge fotograflarini PDF'e donusturur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python scanner.py belge.jpg                    # Basit kullanim
  python scanner.py belge.jpg -o cikti.pdf       # Cikti dosyasi belirt
  python scanner.py belge.jpg --onizleme         # Adimlari gorsel goster
        """
    )
    
    # Zorunlu arguman: Goruntu dosyasi
    parser.add_argument(
        "goruntu",
        help="Taranacak belgenin goruntu dosyasi (jpg, png, vb.)"
    )
    
    # Opsiyonel arguman: Cikti PDF yolu
    parser.add_argument(
        "-o", "--output",
        help="Cikti PDF dosyasinin yolu",
        default=None
    )
    
    # Flag arguman: Onizleme modu
    # action="store_true": Verilirse True, verilmezse False
    parser.add_argument(
        "--onizleme",
        action="store_true",
        help="Tarama adimlarini gorsel olarak goster"
    )
    
    # Argumanlari parse et
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Dosya Varlik Kontrolu
    # -------------------------------------------------------------------------
    if not os.path.exists(args.goruntu):
        print(f"HATA: Dosya bulunamadi: {args.goruntu}")
        return 1  # Hata kodu
    
    # -------------------------------------------------------------------------
    # Tarama Islemini Calistir
    # -------------------------------------------------------------------------
    try:
        # Onizleme modu istendiyse once gorsellestir
        if args.onizleme:
            onizleme_goster(args.goruntu)
        
        # Ana tarama islemini gerceklestir
        pdf_yolu = belge_tara(args.goruntu, args.output)
        print(f"\nIslem tamamlandi! PDF dosyasi: {pdf_yolu}")
        return 0  # Basari kodu
        
    except Exception as e:
        # Hata durumunda mesaj yazdir
        print(f"HATA: {str(e)}")
        return 1  # Hata kodu


# =============================================================================
# SCRIPT GIRIS NOKTASI
# =============================================================================
# Bu dosya dogrudan calistirildiginda (import edilmeden) main() fonksiyonu calisir
# Python'da __name__ degiskeni dosya dogrudan calistirildiginda "__main__" olur

if __name__ == "__main__":
    # main() fonksiyonunun donush kodunu program cikis kodu olarak kullan
    exit(main())
