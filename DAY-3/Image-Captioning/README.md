# CLIP Image Analyzer

OpenAI CLIP modeli kullanarak goruntu analizi yapan bir web uygulamasi. Zero-shot siniflandirma ile goruntuler ve metinler arasindaki anlami karsilastirir.

## Ne Ise Yarar?

Bu uygulama, yukleyeceginiz herhangi bir goruntuyu (meme, fotograf, cizim vb.) analiz eder ve:

- Goruntuyu siniflandirir (kedi, kopek, araba, manzara vb.)
- Goruntunun turunu tespit eder (meme, fotograf, cizim vb.)
- Duygu analizi yapar (mutlu, uzgun, komik vb.)
- Ozel etiketlerle eslestirme yapar
- Goruntu-metin benzerlik skoru hesaplar

## Uygulama Akis Diyagrami

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLIP IMAGE ANALYZER PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────┐
                              │  BASLANGIC    │
                              │ Goruntu Yukle │
                              └───────┬───────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│       IMAGE ENCODER               │   │       TEXT ENCODER                │
│       (Vision Transformer)        │   │       (Transformer)               │
├───────────────────────────────────┤   ├───────────────────────────────────┤
│                                   │   │                                   │
│  ┌─────────────────────────────┐  │   │  ┌─────────────────────────────┐  │
│  │     Goruntu (RGB)           │  │   │  │     Etiketler/Metinler     │  │
│  │     224 x 224 x 3           │  │   │  │     ["cat", "dog", ...]    │  │
│  └──────────────┬──────────────┘  │   │  └──────────────┬──────────────┘  │
│                 │                 │   │                 │                 │
│                 ▼                 │   │                 ▼                 │
│  ┌─────────────────────────────┐  │   │  ┌─────────────────────────────┐  │
│  │   Patch Bolme (32x32)       │  │   │  │   Prompt Template           │  │
│  │   224/32 = 7x7 = 49 patch   │  │   │  │   "a photo of {label}"     │  │
│  └──────────────┬──────────────┘  │   │  └──────────────┬──────────────┘  │
│                 │                 │   │                 │                 │
│                 ▼                 │   │                 ▼                 │
│  ┌─────────────────────────────┐  │   │  ┌─────────────────────────────┐  │
│  │   Lineer Projeksiyon        │  │   │  │   BPE Tokenization          │  │
│  │   + Position Encoding       │  │   │  │   (Byte-Pair Encoding)      │  │
│  └──────────────┬──────────────┘  │   │  └──────────────┬──────────────┘  │
│                 │                 │   │                 │                 │
│                 ▼                 │   │                 ▼                 │
│  ┌─────────────────────────────┐  │   │  ┌─────────────────────────────┐  │
│  │   Transformer Encoder       │  │   │  │   Transformer Encoder       │  │
│  │   (12 Katman, ViT-B/32)    │  │   │  │   (12 Katman)               │  │
│  └──────────────┬──────────────┘  │   │  └──────────────┬──────────────┘  │
│                 │                 │   │                 │                 │
│                 ▼                 │   │                 ▼                 │
│  ┌─────────────────────────────┐  │   │  ┌─────────────────────────────┐  │
│  │   CLS Token Ciktisi         │  │   │  │   EOS Token Ciktisi         │  │
│  │   (512 boyutlu vektor)      │  │   │  │   (512 boyutlu vektor)      │  │
│  └──────────────┬──────────────┘  │   │  └──────────────┬──────────────┘  │
│                 │                 │   │                 │                 │
│                 ▼                 │   │                 ▼                 │
│  ┌─────────────────────────────┐  │   │  ┌─────────────────────────────┐  │
│  │   L2 Normalizasyon          │  │   │  │   L2 Normalizasyon          │  │
│  │   ||v|| = 1                 │  │   │  │   ||v|| = 1                 │  │
│  └──────────────┬──────────────┘  │   │  └──────────────┬──────────────┘  │
│                 │                 │   │                 │                 │
└─────────────────┼─────────────────┘   └─────────────────┼─────────────────┘
                  │                                       │
                  │      IMAGE EMBEDDING                  │      TEXT EMBEDDINGS
                  │      (1 x 512)                        │      (N x 512)
                  │                                       │
                  └───────────────────┬───────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BENZERLIK HESAPLAMA                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     COSINE SIMILARITY                               │   │
│   │                                                                     │   │
│   │              Image_Emb · Text_Emb_i                                 │   │
│   │   sim(i) = ─────────────────────────                                │   │
│   │             ||Image_Emb|| × ||Text_Emb_i||                          │   │
│   │                                                                     │   │
│   │   (L2 normalize edilmis oldugu icin: sim = dot product)            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Goruntu ──────▶ [0.89, 0.12, 0.05, 0.23, ...] ◀────── Benzerlik Skorlari │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        SOFTMAX                                      │   │
│   │                                                                     │   │
│   │              exp(sim_i × temperature)                               │   │
│   │   P(i) = ──────────────────────────────                             │   │
│   │           Σ exp(sim_j × temperature)                                │   │
│   │                                                                     │   │
│   │   temperature ≈ 100 (ogrenilmis parametre)                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SONUC                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Etiket          │  Benzerlik Skoru  │  Olasilik (%)                      │
│   ────────────────┼───────────────────┼──────────────────                  │
│   cat             │      0.89         │     85.3%        ◄── En Yuksek     │
│   animal          │      0.45         │     10.2%                          │
│   dog             │      0.23         │      3.1%                          │
│   car             │      0.12         │      1.4%                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────┐
                              │   BITIS   │
                              │  Sonuclar │
                              │  Gosterim │
                              └───────────┘
```

## CLIP Nedir?

**CLIP (Contrastive Language-Image Pre-training)**, OpenAI tarafindan 2021 yilinda gelistirilen cok modlu (multimodal) bir yapay zeka modelidir.

### Temel Ozellikler

| Ozellik | Aciklama |
|---------|----------|
| **Egitim Verisi** | 400 milyon goruntu-metin cifti (internet'ten toplanan) |
| **Mimari** | Vision Transformer (ViT) + Text Transformer |
| **Zero-shot** | Hic gormedigi kategorileri tanimlar |
| **Multimodal** | Goruntu ve metni ayni uzayda temsil eder |

### CLIP Nasil Calisir?

CLIP modeli iki ana bilesenden olusur:

1. **Image Encoder (Goruntu Kodlayici)**
   - Goruntuyu alir
   - Vision Transformer (ViT) veya ResNet kullanarak isler
   - 512 boyutlu bir vektor (embedding) uretir

2. **Text Encoder (Metin Kodlayici)**
   - Metni alir
   - Transformer tabanlı model ile isler
   - 512 boyutlu bir vektor (embedding) uretir

3. **Benzerlik Hesaplama**
   - Goruntu ve metin vektorleri arasinda cosine similarity hesaplanir
   - En yuksek benzerlige sahip metin, goruntunun aciklamasi olur

### Contrastive Learning (Zitlikli Ogrenme)

CLIP, **contrastive learning** yontemi ile egitilir:

- Eslesen goruntu-metin ciftleri birbirine yakin olmali (pozitif)
- Eslesmeyen ciftler birbirinden uzak olmali (negatif)
- Model, dogru eslesmeleri bulmayı ogrenir

```
Goruntu: [kedi fotografi]  <--> Metin: "a photo of a cat"     (YAKIN)
Goruntu: [kedi fotografi]  <--> Metin: "a photo of a dog"     (UZAK)
```

### CLIP Egitim Matrisi

```
                    Metin_1   Metin_2   Metin_3   ...   Metin_N
                   ┌─────────┬─────────┬─────────┬─────┬─────────┐
     Goruntu_1     │  ✓ +++ │   ---   │   ---   │ ... │   ---   │
                   ├─────────┼─────────┼─────────┼─────┼─────────┤
     Goruntu_2     │   ---   │  ✓ +++ │   ---   │ ... │   ---   │
                   ├─────────┼─────────┼─────────┼─────┼─────────┤
     Goruntu_3     │   ---   │   ---   │  ✓ +++ │ ... │   ---   │
                   ├─────────┼─────────┼─────────┼─────┼─────────┤
        ...        │   ...   │   ...   │   ...   │ ... │   ...   │
                   ├─────────┼─────────┼─────────┼─────┼─────────┤
     Goruntu_N     │   ---   │   ---   │   ---   │ ... │  ✓ +++ │
                   └─────────┴─────────┴─────────┴─────┴─────────┘

     ✓ +++ = Pozitif cift (yakinlastir)
       --- = Negatif cift (uzaklastir)
```

### Zero-Shot Siniflandirma

CLIP'in en onemli ozelligi **zero-shot** siniflandirma yapabilmesidir:

- Model, siniflandirma icin onceden egitilmemis olsa bile
- Sadece sinif isimlerini metin olarak vererek
- Goruntuyu siniflandirir

**Ornek:**
```python
labels = ["cat", "dog", "bird", "car"]
# Model bu etiketleri hic gormemis olsa bile
# Goruntuyu dogru etiketle eslestirir
```

### CLIP vs Geleneksel Siniflandirma

| Geleneksel CNN | CLIP |
|----------------|------|
| Sabit sinif sayisi | Sinirsiz sinif |
| Her sinif icin egitim gerekli | Zero-shot |
| Sadece goruntu | Goruntu + metin |
| Yeni sinif = yeniden egitim | Yeni sinif = sadece metin ekle |

## Kullanilan Teknolojiler

| Teknoloji | Aciklama |
|-----------|----------|
| **Python 3.8+** | Programlama dili |
| **PyTorch** | Derin ogrenme kutuphanesi |
| **Hugging Face Transformers** | Model yukleyici |
| **OpenAI CLIP** | Goruntu-metin modeli |
| **Streamlit** | Web arayuzu |
| **Pillow** | Goruntu isleme |
| **NumPy** | Sayisal hesaplamalar |

## Kurulum

```bash
# Proje klasorune gidin
cd Image-Captioning

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
pip install torch torchvision transformers streamlit pillow numpy
```

**Not:** GPU kullanmak icin uygun PyTorch versiyonunu yukleyin: https://pytorch.org/get-started/locally/

## Kullanim

### Web Arayuzu (Streamlit)

```bash
streamlit run app.py
```

Tarayicinizda `http://localhost:8501` adresinde acilir.

**Ozellikler:**
- Surukle-birak goruntu yukleme
- 5 farkli analiz modu
- Ozel etiket tanimlama
- Metin-goruntu karsilastirma
- Detayli sonuc gosterimi

### Python ile Kullanim

```python
from clip_model import ClipAnalyzer

# Modeli yukle (Hugging Face'den)
# Varsayilan: "openai/clip-vit-base-patch32"
analyzer = ClipAnalyzer()

# Goruntuyu analiz et
from PIL import Image
image = Image.open("meme.jpg")

# Siniflandirma
labels = ["funny meme", "sad meme", "cute animal", "text post"]
results = analyzer.analyze(image, labels)

for label, score in results:
    print(f"{label}: {score:.1f}%")
```

### Goruntu-Metin Karsilastirma

```python
# Dogrudan metin karsilastirma
texts = [
    "a cat sitting on a computer",
    "a dog playing in the park",
    "a funny internet meme"
]

scores = analyzer.calculate_similarity(image, texts)
for text, score in zip(texts, scores):
    print(f"{text}: {score:.1f}%")
```

## Proje Yapisi

```
Image-Captioning/
│
├── app.py           # Streamlit web arayuzu
│                    # - Goruntu yukleme
│                    # - Analiz modlari
│                    # - Sonuc gosterimi
│
├── clip_model.py    # CLIP model fonksiyonlari
│                    # - ClipAnalyzer sinifi
│                    # - encode_image()
│                    # - encode_text()
│                    # - calculate_similarity()
│                    # - analyze()
│
├── img/             # Ornek gorseller
│   ├── cat.jpeg
│   ├── cat-meme.jpeg
│   ├── joey.jpeg
│   ├── landscape.jpeg
│   ├── old-women.jpeg
│   └── Windows-95.jpeg
│
├── requirements.txt # Bagimliliklar
│
└── README.md        # Bu dosya
```

## Model Secenekleri

CLIP modelleri Hugging Face uzerinden erisilebilir:

| Hugging Face Model ID | Parametre | Hiz | Kalite |
|-----------------------|-----------|-----|--------|
| openai/clip-vit-base-patch32 | 151M | Hizli | Iyi |
| openai/clip-vit-base-patch16 | 151M | Orta | Daha iyi |
| openai/clip-vit-large-patch14 | 428M | Yavas | En iyi |

Varsayilan olarak **openai/clip-vit-base-patch32** kullanilir (hiz ve kalite dengesi).

```python
# Farkli model kullanmak icin:
analyzer = ClipAnalyzer(model_name="openai/clip-vit-large-patch14")
```

## Ogrenme Noktalari

Bu proje ile asagidaki konulari ogrenebilirsiniz:

1. **Multimodal Ogrenme**: Farkli modaliteleri (goruntu, metin) birlestirme
2. **Contrastive Learning**: Zitlikli ogrenme teknigi
3. **Vision Transformer (ViT)**: Transformer mimarisinin goruntuye uygulanmasi
4. **Zero-shot Learning**: Egitim olmadan siniflandirma
5. **Embedding Uzayi**: Vektorel temsiller ve benzerlik
6. **Transfer Learning**: Onceden egitilmis modellerin kullanimi

## Sinirlamalar

- CLIP Ingilizce metinlerle daha iyi calisir
- Cok ozel/teknik alanlarda performans dusebilir
- Buyuk modeller daha fazla GPU bellegi gerektirir
- Ince detaylari ayirt etmekte zorlanabilir

## Kaynaklar

- [CLIP Paper (arXiv)](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- [OpenAI Blog Post](https://openai.com/research/clip)

## Lisans

Bu proje egitim amacli hazirlanmistir. CLIP modeli OpenAI tarafindan MIT lisansi ile sunulmaktadir.

---

*SAU Yapay Zeka & Bilgisayarli Goru Egitimi - Gun 3*
