# Dosya Tarayici (File Scanner)

Bir kagit belgenin fotografini alarak taranmis gibi duzgun bir PDF dosyasina donusturen basit bir bilgisayarli goru uygulamasi.

## Ne Ise Yarar?

Bu uygulama, telefonunuzla veya kameranizla cektiginiz belge fotograflarini profesyonel gorunumlu PDF dosyalarina donusturur. Tipki bir tarayici (scanner) gibi calisir:

- Egik cekilmis belgeleri duzeltir
- Perspektif bozukluklarini giderir
- Kontrasti artirarak okunabilirligi iyilestirir
- A4/A5 boyutunda PDF ciktisi olusturur

## Uygulama Akis Diyagrami

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOSYA TARAYICI PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────┐
                              │  BASLANGIC    │
                              │ Goruntu Yukle │
                              └───────┬───────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ADIM 1: ON-ISLEME (Preprocessing)                                           │
│ ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────┐          │
│ │ BGR Goruntu │───▶│ Gri Tonlama    │───▶│ Gaussian Blur (5x5) │          │
│ │ (Orijinal)  │    │ (Grayscale)    │    │ (Gurultu Azaltma)   │          │
│ └─────────────┘    └─────────────────┘    └──────────┬──────────┘          │
└──────────────────────────────────────────────────────┼──────────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ADIM 2: KENAR TESPITI (Edge Detection)                                      │
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│   │Canny Edge   │───▶│ Dilation    │───▶│  Erosion    │───▶│   Kenar     │ │
│   │Detection    │    │ (Genislet)  │    │ (Incelt)    │    │   Haritasi  │ │
│   │(50,150)     │    │ 2 iterasyon │    │ 1 iterasyon │    │   (Binary)  │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘ │
└──────────────────────────────────────────────────────────────────┼──────────┘
                                                                   │
                                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ADIM 3: BELGE TESPITI (Document Detection)                                  │
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐│
│   │ findContours()  │───▶│ Alana Gore     │───▶│ Douglas-Peucker         ││
│   │ (Konturlari Bul)│    │ Siralama       │    │ Algoritmasi (4 Kose)    ││
│   └─────────────────┘    └─────────────────┘    └────────────┬────────────┘│
│                                                              │             │
│                     ┌────────────────────────────────────────┘             │
│                     │                                                      │
│                     ▼                                                      │
│          ┌────────────────────┐                                            │
│          │ Konvekslik         │                                            │
│          │ Kontrolu           │                                            │
│          │ (4 koseli sekil?)  │                                            │
│          └─────────┬──────────┘                                            │
│                    │                                                       │
│        ┌───────────┴───────────┐                                           │
│        │                       │                                           │
│    EVET ▼                  HAYIR ▼                                         │
│   ┌──────────┐           ┌──────────────────┐                              │
│   │ 4 Kose   │           │ Alternatif Yontem│                              │
│   │ Bulundu  │           │ (Adaptif Esikleme│                              │
│   └────┬─────┘           │  veya Farkli     │                              │
│        │                 │  Canny Param.)   │                              │
│        │                 └────────┬─────────┘                              │
│        │                          │                                        │
│        └──────────┬───────────────┘                                        │
│                   ▼                                                        │
└───────────────────┼─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ADIM 4: KOSE SIRALAMA (Corner Ordering)                                     │
│                                                                             │
│              Sol-Ust (0)──────────────────────Sag-Ust (1)                   │
│                  │  x+y min             x-y min  │                          │
│                  │                               │                          │
│                  │         BELGE                 │                          │
│                  │                               │                          │
│                  │  x-y max             x+y max  │                          │
│              Sol-Alt (3)──────────────────────Sag-Alt (2)                   │
│                                                                             │
│   Siralama: [sol-ust, sag-ust, sag-alt, sol-alt] (Saat Yonunde)            │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ADIM 5: PERSPEKTIF DUZELTME (Homography Transform)                          │
│                                                                             │
│   KAYNAK (Egik Goruntu)           HEDEF (Duz Dikdortgen)                    │
│   ┌─────────────────────┐         ┌─────────────────────┐                   │
│   │    /          \     │         │                     │                   │
│   │   /            \    │  ────▶  │                     │                   │
│   │  /   BELGE      \   │ 3x3     │      BELGE          │                   │
│   │ /                \  │ Matris  │                     │                   │
│   │/                  \ │         │                     │                   │
│   └─────────────────────┘         └─────────────────────┘                   │
│                                                                             │
│   getPerspectiveTransform() ──▶ warpPerspective()                           │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ADIM 6: GORUNTU IYILESTIRME (Image Enhancement)                             │
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐│
│   │ Gri Tonlama     │───▶│ Adaptif Esikleme   │───▶│ Median Blur (3x3)   ││
│   │ Donusumu        │    │ (Gaussian, 11x11)  │    │ (Gurultu Temizleme) ││
│   └─────────────────┘    └─────────────────────┘    └──────────┬──────────┘│
│                                                                │           │
│                           TARANMIS GORUNUM                     │           │
│                           (Siyah-Beyaz)                        │           │
└────────────────────────────────────────────────────────────────┼───────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ADIM 7: PDF OLUSTURMA                                                       │
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐│
│   │ JPEG Olarak     │───▶│ img2pdf ile     │───▶│ A4 Boyutunda PDF       ││
│   │ Gecici Kayit    │    │ Donustur        │    │ Dosyasi                ││
│   └─────────────────┘    └─────────────────┘    └─────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                              ┌───────────┐
                              │   BITIS   │
                              │ PDF Hazir │
                              └───────────┘
```

## Kullanilan Teknolojiler

| Teknoloji | Aciklama |
|-----------|----------|
| **Python 3.8+** | Programlama dili |
| **OpenCV** | Goruntu isleme kutuphanesi |
| **NumPy** | Matematiksel hesaplamalar |
| **Pillow** | Goruntu formati donusumleri |
| **img2pdf** | PDF olusturma |
| **Streamlit** | Web arayuzu framework'u |

## Kullanilan Bilgisayarli Goru Teknikleri

### 1. On-Isleme (Preprocessing)
Goruntunun belge tespiti icin hazirlanmasi:
- BGR -> Grayscale donusumu
- Gaussian Blur ile gurultu azaltma
- Morfolojik islemler (dilation, erosion)

### 2. Kenar Tespiti (Edge Detection)
**Canny Edge Detection** algoritmasi + morfolojik guclandirme:
- Gaussian bulaniklastirma ile gurultu azaltma
- Sobel operatoru ile gradyan hesaplama
- Non-maximum suppression ile ince kenarlar
- Hysteresis esikleme ile kenar baglama
- Dilation ile kenarlari kalinlastirma
- Erosion ile ince ayar

### 3. Belge Tespiti (Document Detection / ROI)
Birden fazla yontem denenerek en iyi sonuc alinir:
- Canny + Morfolojik islemler (birincil yontem)
- Adaptif esikleme (alternatif yontem)
- Farkli Canny parametreleri ile deneme

### 4. Kontur Analizi (Contour Analysis)
Kenarlardan belge bolgesini cikarma:
- `cv2.findContours()` ile kontur cikarma
- Alana gore siralama (en buyuk = belge)
- Douglas-Peucker algoritmasi ile dortgen yaklasimi
- Konvekslik kontrolu
- Minimum alan filtreleme

### 5. Kose Siralama (Corner Ordering)
4 koseyi saat yonunde siralama:
- Sol-ust: x+y toplami en kucuk
- Sag-alt: x+y toplami en buyuk
- Sag-ust: y-x farki en kucuk
- Sol-alt: y-x farki en buyuk

### 6. Perspektif Donusumu (Homography)
Egik belgeyi duz hale getirme (kus bakisi):
- 4 kaynak noktasi (tespit edilen koseler)
- 4 hedef noktasi (duz dikdortgen)
- `cv2.getPerspectiveTransform()` ile 3x3 homografi matrisi
- `cv2.warpPerspective()` ile donusum uygulama

### 7. Goruntu Iyilestirme (Enhancement)
Taranmis belge gorunumu olusturma:
- Adaptif esikleme (her piksel icin dinamik esik)
- Gaussian agirlikli yerel ortalama
- Median filtresi ile gurultu azaltma

## Kurulum

```bash
# Proje klasorune gidin
cd File-Scanner

# Sanal ortam olusturun (onerilir)
python -m venv venv

# Sanal ortami aktiflestirin
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Bagimliliklari yukleyin
pip install -r requirements.txt
```

## Kullanim

### Web Arayuzu (Streamlit)

En kolay kullanim yontemi! Tarayicida calisir:

```bash
streamlit run app.py
```

Tarayicinizda `http://localhost:8501` adresinde acilir.

**Ozellikler:**
- Surukle-birak dosya yukleme
- Anlik onizleme
- Ayarlanabilir secenekler
- Tek tikla PDF indirme
- Islem adimlarini gorsel inceleme

### Komut Satiri Kullanimi

#### Basit Kullanim

```bash
python scanner.py belge_fotografi.jpg
```

Bu komut, `belge_fotografi_taranmis.pdf` dosyasini olusturur.

#### Cikti Dosyasi Belirleme

```bash
python scanner.py belge.jpg -o odevim.pdf
```

#### Adimlari Gorsel Olarak Inceleme

```bash
python scanner.py belge.jpg --onizleme
```

Bu mod, her islem adimini gorsel olarak gosterir (egitim amacli):

```
+------------------+------------------+
|   1. Orijinal    |  2. Kenarlar     |
+------------------+------------------+
|   3. Koseler     |  4. Sonuc        |
+------------------+------------------+
```

#### Yardim

```bash
python scanner.py --help
```

## Proje Yapisi

```
File-Scanner/
│
├── app.py          # Streamlit web arayuzu
│                   # - Surukle-birak dosya yukleme
│                   # - Anlik onizleme
│                   # - PDF indirme
│
├── scanner.py      # Komut satiri uygulamasi
│                   # - Terminal arayuzu
│                   # - Ana tarama is akisi
│                   # - Onizleme modu
│
├── utils.py        # Yardimci fonksiyonlar
│                   # - kenar_tespit(): Canny algoritmasi
│                   # - kontur_bul(): Belge sinirlari
│                   # - koseler_sirala(): Kose siralamasi
│                   # - perspektif_duzelt(): Goruntu duzeltme
│                   # - goruntu_iyilestir(): Kontrast/temizlik
│                   # - pdf_olustur(): PDF donusumu
│
├── img/            # Ornek gorseller
│   ├── front.jpeg
│   └── page.jpeg
│
├── requirements.txt # Bagimliliklar listesi
│
└── README.md        # Bu dosya
```

## Ornek Kullanim Senaryosu

```python
# Python icinde kullanim
from scanner import belge_tara

# Tek bir belgeyi tara
pdf_yolu = belge_tara("belge.jpg", "cikti.pdf")
print(f"PDF olusturuldu: {pdf_yolu}")
```

```python
# Birden fazla sayfayi tek PDF'e birlestir
from utils import coklu_sayfa_pdf, perspektif_duzelt, goruntu_iyilestir
import cv2

sayfalar = []
for dosya in ["sayfa1.jpg", "sayfa2.jpg", "sayfa3.jpg"]:
    img = cv2.imread(dosya)
    # ... islemler ...
    sayfalar.append(iyilestirilmis)

coklu_sayfa_pdf(sayfalar, "kitap.pdf")
```

## Ogrenme Noktalari

Bu proje ile asagidaki konulari ogrenebilirsiniz:

1. **Goruntu Okuma/Yazma**: `cv2.imread()`, `cv2.imwrite()`
2. **Renk Donusumleri**: BGR - Grayscale - RGB
3. **Filtreleme**: Gaussian Blur, Median Blur
4. **Kenar Tespiti**: Canny algoritmasi
5. **Kontur Islemleri**: findContours, approxPolyDP
6. **Geometrik Donusumler**: Perspektif donusumu
7. **Esikleme**: Adaptif esikleme teknikleri

## Bilinen Sinirlamalar

- Belge koseleri net gorunmuyorsa tespit edilemeyebilir
- Cok bulanik goruntulerde sonuc kalitesi dusebilir
- Karmasik arka planlarda belge tespiti zorlasabilir

## Gelistirme Onerileri

Projeyi gelistirmek isteyenler icin fikirler:

1. **OCR Entegrasyonu**: Tesseract ile metin tanima ekleyin
2. **GUI Arayuzu**: Tkinter veya PyQt ile masaustu uygulamasi
3. **Otomatik Dondurme**: Belge yonunu otomatik algilama
4. **Toplu Islem**: Klasordeki tum goruntuleri isleme

## Lisans

Bu proje egitim amacli hazirlanmistir. Ozgurce kullanabilir ve gelistirebilirsiniz.

---

*SAU Yapay Zeka & Bilgisayarli Goru Egitimi - Gun 3*
