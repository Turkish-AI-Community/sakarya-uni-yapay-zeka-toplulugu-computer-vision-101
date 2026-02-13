"""
================================================================================
CLIP Model Fonksiyonlari (Hugging Face Transformers)
================================================================================

Bu modul, OpenAI CLIP modelini Hugging Face uzerinden kullanarak
goruntu analizi yapar.

================================================================================
CLIP DETAYLI ACIKLAMA
================================================================================

CLIP (Contrastive Language-Image Pre-training):
-----------------------------------------------
OpenAI tarafindan 2021'de yayinlanan cok modlu (multimodal) bir yapay zeka
modeli. Goruntuleri ve metinleri ayni vektorel uzayda temsil ederek
aralarindaki anlami karsilastirir.

EGITIM YONTEMI:
---------------
Contrastive Learning (Karsilastirmali Ogrenme):

1. Egitim verisi: (goruntu, metin) ciftleri
   - Internet'ten toplanan 400 milyon cifti
   - Goruntu: JPG/PNG dosyalari
   - Metin: Alt yazılar, aciklamalar

2. Pozitif ve Negatif Ornekler:
   - Pozitif: Eslesen goruntu-metin cifti
   - Negatif: Eslesmeyen ciftler (batch icindeki diger ciftler)

3. Kayip Fonksiyonu:
   - InfoNCE loss (Noise Contrastive Estimation)
   - Pozitif ciftleri yaklastir
   - Negatif ciftleri uzaklastir

4. Matris Formati:
   Batch icindeki N goruntu ve N metin icin:
   
   Metin1  Metin2  Metin3  ...  MetinN
   +-------+-------+-------+---+-------+
   |  +++  |  ---  |  ---  |...|  ---  |  Goruntu1
   +-------+-------+-------+---+-------+
   |  ---  |  +++  |  ---  |...|  ---  |  Goruntu2
   +-------+-------+-------+---+-------+
   |  ---  |  ---  |  +++  |...|  ---  |  Goruntu3
   +-------+-------+-------+---+-------+
   |  ...  |  ...  |  ...  |...|  ...  |  ...
   +-------+-------+-------+---+-------+
   |  ---  |  ---  |  ---  |...|  +++  |  GoruntuN
   +-------+-------+-------+---+-------+
   
   +++ = yuksek benzerlik (pozitif, diyagonal)
   --- = dusuk benzerlik (negatif)

ARSITEKTUR:
-----------
1. Image Encoder (Goruntu Kodlayici):
   - Vision Transformer (ViT) veya ResNet
   - Goruntu -> 512 boyutlu vektor
   
   ViT-B/32 (Vision Transformer Base, patch size 32):
   - Goruntuyu 32x32 piksellik parcalara (patch) boler
   - Her parcayi flatten edip lineer proyeksiyondan gecirir
   - Transformer encoder ile isle
   - CLS token ciktisi = goruntu temsili

2. Text Encoder (Metin Kodlayici):
   - Transformer tabanlı
   - Metin -> 512 boyutlu vektor
   - BPE tokenization kullanir
   - [EOS] token ciktisi = metin temsili

3. Ortak Uzay (Joint Embedding Space):
   - Her iki encoder da 512 boyutlu cikti verir
   - Bu ciktilar normalize edilir
   - Ayni uzayda benzer anlamlar yakin, farkli anlamlar uzak

ZERO-SHOT SINIFLANDIRMA:
------------------------
CLIP'in en guclu ozelligi zero-shot ogrenme yetenegi:

1. Sinif isimleri metin olarak verilir
   Ornek: ["cat", "dog", "bird"]

2. Prompt template uygulanir
   Ornek: ["a photo of cat", "a photo of dog", "a photo of bird"]

3. Goruntu ve metinler encode edilir

4. Cosine similarity hesaplanir:
   similarity(img, text) = img_emb · text_emb / (|img_emb| * |text_emb|)

5. Softmax ile olasiliklara cevrilir:
   P(class_i) = exp(sim_i / T) / sum(exp(sim_j / T))
   T = sicaklik parametresi (CLIP'te 0.07 gibi)

6. En yuksek olasilikli sinif secilir

KULLANIM ORNEKLERI:
-------------------
# Model yukle
analyzer = ClipAnalyzer()

# Goruntu analizi
results = analyzer.analyze(image, ["cat", "dog", "bird"])
# [(cat, 85.3), (bird, 10.2), (dog, 4.5)]

# Metin benzerligi
scores = analyzer.calculate_similarity(image, ["a happy cat", "a sad dog"])
# [92.1, 7.9]

Yazar: SAU Yapay Zeka & Bilgisayarli Goru Toplulugu
================================================================================
"""

# =============================================================================
# KUTUPHANELERIN ICE AKTARILMASI
# =============================================================================

# PyTorch: Derin ogrenme framework'u
# - Tensor islemleri
# - GPU hizlandirma
# - Otomatik turev alma (autograd)
import torch

# PIL: Python Imaging Library
# - Goruntu dosyalarini okuma/yazma
# - Format donusumleri
from PIL import Image

# NumPy: Sayisal hesaplama
# - Array islemleri
# - Sonuclari Python listelerine donusturme
import numpy as np

# Typing: Python tip ipuclari
# - Fonksiyon parametre ve donus tiplerini belirtme
# - IDE destegi ve kod okunabilirligi
from typing import List, Tuple, Union

# Hugging Face Transformers: NLP ve Computer Vision modelleri
# - Onceden egitilmis model ve processor'lar
# - Kolay model yukleme ve kullanma
from transformers import CLIPProcessor, CLIPModel


# =============================================================================
# CLIP ANALYZER SINIFI
# =============================================================================

class ClipAnalyzer:
    """
    CLIP modeli ile goruntu analizi yapan sinif.
    
    Bu sinif, CLIP modelini Hugging Face'den yukler ve
    goruntu-metin karsilastirma islemlerini gerceklestirir.
    
    SINIF OZELLIKLERI:
    ------------------
    model : CLIPModel
        Yuklu CLIP modeli
        
    processor : CLIPProcessor
        Goruntu ve metin on-islemcisi
        
    device : str
        Kullanilan cihaz ("cuda" veya "cpu")
    
    TEMEL METODLAR:
    ---------------
    encode_image(image) -> tensor
        Goruntuyu 512-boyutlu vektore donusturur
        
    encode_text(texts) -> tensor
        Metin listesini vektorlere donusturur
        
    calculate_similarity(image, texts) -> array
        Goruntu-metin benzerlik skorlarini hesaplar
        
    analyze(image, labels, template) -> list
        Zero-shot siniflandirma yapar
    
    KULLANIM:
    ---------
    >>> analyzer = ClipAnalyzer()
    >>> results = analyzer.analyze(image, ["cat", "dog"])
    >>> print(results)
    [("cat", 85.3), ("dog", 14.7)]
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        CLIP modelini Hugging Face'den yukler.
        
        MODEL SECENEKLERI:
        ------------------
        Farkli model boyutlari farkli performans/kalite dengesi sunar:
        
        1. "openai/clip-vit-base-patch32" (Varsayilan):
           - ViT-B/32 arsitekturu
           - En hizli, en hafif
           - ~300MB model boyutu
           - Iyi genel performans
           
        2. "openai/clip-vit-base-patch16":
           - ViT-B/16 arsitekturu
           - Daha kucuk patch = daha fazla detay
           - Biraz daha yavas
           - Daha iyi kalite
           
        3. "openai/clip-vit-large-patch14":
           - ViT-L/14 arsitekturu
           - En buyuk model
           - En yuksek kalite
           - En yavas, en fazla bellek
        
        PATCH SIZE NEDIR?
        -----------------
        Vision Transformer goruntuyu patch'lere boler.
        - patch32: 224x224 goruntu = 7x7 = 49 patch
        - patch16: 224x224 goruntu = 14x14 = 196 patch
        - patch14: 224x224 goruntu = 16x16 = 256 patch
        
        Kucuk patch = Daha fazla detay, daha yavas islem
        
        PARAMETRELER:
        -------------
        model_name : str
            Hugging Face model ismi
        """
        
        # -----------------------------------------
        # Cihaz Secimi
        # -----------------------------------------
        # CUDA (GPU) varsa kullan, yoksa CPU
        # GPU, tensor islemlerini paralel yaparak 10-100x hizlandirir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"CLIP modeli yukleniyor: {model_name}")
        print(f"Cihaz: {self.device}")
        
        # -----------------------------------------
        # Processor ve Model Yukleme
        # -----------------------------------------
        # CLIPProcessor:
        # - Goruntuler icin: Boyutlandirma, normalizasyon, tensor donusumu
        # - Metinler icin: Tokenizasyon, padding, attention mask
        #
        # CLIPModel:
        # - Image encoder (ViT veya ResNet)
        # - Text encoder (Transformer)
        # - Ortak embedding uzayi
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        
        # -----------------------------------------
        # Modeli Degerlendirme Moduna Al
        # -----------------------------------------
        # eval() modu:
        # - Dropout katmanlarini devre disi birakir
        # - BatchNorm istatistiklerini dondurur
        # - Inference icin gerekli
        self.model.eval()
        
        print("Model basariyla yuklendi!")
    
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Goruntuyu CLIP embedding vektorune donusturur.
        
        EMBEDDING NEDIR?
        ----------------
        Embedding, yuksek boyutlu veriyi (goruntu) dusuk boyutlu,
        anlamsal olarak zengin bir vektore donusturme islemidir.
        
        CLIP'te:
        - Giris: 224x224x3 = 150,528 deger
        - Cikis: 512 deger
        
        Bu 512 deger, goruntuun "anlamini" temsil eder.
        Benzer goruntulerin embedding'leri birbirine yakindir.
        
        ISLEM ADIMLARI:
        ---------------
        1. Goruntu On-Isleme (Processor):
           - Boyutlandirma: 224x224
           - Normalizasyon: mean=[0.48145466, 0.4578275, 0.40821073]
                           std=[0.26862954, 0.26130258, 0.27577711]
           - Tensor donusumu: (H,W,C) -> (C,H,W)
        
        2. Image Encoder (ViT):
           - Goruntu -> Patch'ler
           - Patch'ler -> Lineer projeksiyon
           - Position encoding ekleme
           - Transformer encoder katmanlari
           - CLS token ciktisi
        
        3. Normalizasyon:
           - L2 norm = 1 yapilir
           - Cosine similarity hesabi icin gerekli
        
        PARAMETRELER:
        -------------
        image : PIL.Image veya np.ndarray
            Giris goruntusu (herhangi bir boyutta)
        
        DONDURUR:
        ---------
        torch.Tensor
            Shape: (1, 512) - 1 goruntu, 512 boyutlu embedding
            L2 normalized
        """
        
        # NumPy array ise PIL Image'e donustur
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # CLIP RGB bekler, diger modlari donustur
        # RGBA, L (grayscale), P (palette) vb.
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # -----------------------------------------
        # Processor ile On-Isleme
        # -----------------------------------------
        # return_tensors="pt": PyTorch tensor formatinda dondur
        # Cikti: {"pixel_values": tensor(1, 3, 224, 224)}
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Tensorleri GPU'ya tasι (GPU kullaniliyorsa)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # -----------------------------------------
        # Embedding Hesapla
        # -----------------------------------------
        # torch.no_grad(): Gradyan hesaplamasi yapma
        # - Bellek tasarrufu (egitim yapmiyoruz)
        # - Hiz artisi
        with torch.no_grad():
            # get_image_features: Sadece goruntu embedding'ini dondurur
            image_features = self.model.get_image_features(**inputs)
        
        # -----------------------------------------
        # L2 Normalizasyon
        # -----------------------------------------
        # Her vektoru unit uzunluguna getir
        # Bu sayede dot product = cosine similarity olur
        # norm(dim=-1, keepdim=True): Son boyut uzerinde norm, boyutu koru
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Metin listesini CLIP embedding vektorlerine donusturur.
        
        TEXT ENCODING SURECI:
        ---------------------
        1. Tokenizasyon (BPE):
           - Byte-Pair Encoding algoritmasi
           - Metni alt-kelimelere ayirir
           - Vocab size: 49,152
           - Ornek: "photograph" -> ["photo", "graph"]
        
        2. Token -> ID Donusumu:
           - Her token bir tam sayiya eslesir
           - Ornek: ["a", "photo", "of", "cat"] -> [320, 1125, 539, 2368]
        
        3. Padding:
           - Farkli uzunluktaki metinleri ayni boyuta getir
           - Kisa metinler [PAD] tokenlari ile doldurulur
           - Maksimum uzunluk: 77 token
        
        4. Text Encoder (Transformer):
           - Token embeddings + Position embeddings
           - Self-attention katmanlari
           - [EOS] token ciktisi = metin temsili
        
        5. L2 Normalizasyon:
           - Unit uzunluga getir
        
        PARAMETRELER:
        -------------
        texts : List[str]
            Encode edilecek metin listesi
            Ornek: ["a photo of cat", "a photo of dog"]
        
        DONDURUR:
        ---------
        torch.Tensor
            Shape: (N, 512) - N metin, her biri 512 boyutlu
            L2 normalized
        """
        
        # -----------------------------------------
        # Tokenizasyon ve On-Isleme
        # -----------------------------------------
        # text=texts: Metin listesi
        # return_tensors="pt": PyTorch tensor
        # padding=True: Kisa metinleri doldur
        #
        # Cikti:
        # - input_ids: Token ID'leri (N, max_len)
        # - attention_mask: Hangi tokenlar gercek, hangileri padding (N, max_len)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        
        # GPU'ya tasi
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # -----------------------------------------
        # Embedding Hesapla
        # -----------------------------------------
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # L2 normalizasyon
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def calculate_similarity(self, 
                            image: Union[Image.Image, np.ndarray],
                            texts: List[str]) -> np.ndarray:
        """
        Goruntu ile metinler arasindaki benzerlik skorlarini hesaplar.
        
        BENZERLIK HESABI:
        -----------------
        CLIP, cosine similarity kullanir:
        
        cosine_sim(A, B) = (A · B) / (|A| * |B|)
        
        Vektorler L2 normalize edildiginde:
        |A| = |B| = 1
        
        Dolayisiyla:
        cosine_sim(A, B) = A · B (dot product)
        
        SOFTMAX DONUSUMU:
        -----------------
        Ham benzerlik skorlari [-1, 1] araligindadir.
        Bunlari [0, 1] araliginda olasiliklara ceviriyoruz:
        
        P(text_i | image) = exp(sim_i * logit_scale) / sum(exp(sim_j * logit_scale))
        
        logit_scale: Ogrenilen sicaklik parametresi (~100)
        
        PARAMETRELER:
        -------------
        image : PIL.Image veya np.ndarray
            Karsilastirilacak goruntu
            
        texts : List[str]
            Karsilastirilacak metin listesi
        
        DONDURUR:
        ---------
        np.ndarray
            Shape: (N,) - Her metin icin benzerlik skoru (0-100 arasi)
            Toplam = 100 (olasilik dagilimi)
        """
        
        # Goruntu format kontrolu
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # -----------------------------------------
        # Goruntu ve Metinleri Birlikte Isle
        # -----------------------------------------
        # Processor her ikisini de ayni anda isleyebilir
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # -----------------------------------------
        # Model Ciktisini Al
        # -----------------------------------------
        with torch.no_grad():
            # Model goruntu ve metin embedding'lerini dondurur
            # Ayrica logits_per_image ve logits_per_text hesaplar
            outputs = self.model(**inputs)
        
        # -----------------------------------------
        # Logits -> Olasilik Donusumu
        # -----------------------------------------
        # logits_per_image: (1, N) boyutunda
        # Her metin icin benzerlik skoru
        logits = outputs.logits_per_image
        
        # Softmax: Skorlari olasiliklara cevir
        # dim=-1: Son boyut uzerinde (metinler)
        # Toplam = 1.0
        probs = logits.softmax(dim=-1)
        
        # CPU'ya tasi ve NumPy'a donustur
        # [0]: Batch boyutunu kaldir
        # * 100: Yuzdelik degere cevir
        return probs.cpu().numpy()[0] * 100
    
    def analyze(self, 
                image: Union[Image.Image, np.ndarray],
                labels: List[str],
                prompt_template: str = "a photo of {}") -> List[Tuple[str, float]]:
        """
        Goruntuyu analiz eder ve etiketlerle eslestirir.
        
        ZERO-SHOT SINIFLANDIRMA:
        ------------------------
        Bu metod, CLIP'in en guclu ozelligi olan zero-shot siniflandirmayi yapar.
        
        Zero-shot: Model bu etiketleri hic gormemis olsa bile siniflandirabilir.
        
        PROMPT ENGINEERING:
        -------------------
        Sadece "cat" yerine "a photo of cat" kullanmak sonuclari iyilestirir.
        
        Neden?
        - CLIP, dogal dil aciklamalari ile egitilmis
        - "cat" tek bir kelime, context yok
        - "a photo of cat" daha dogal, egitim verisine benzer
        
        Bazi etkili prompt template'leri:
        - "a photo of {}"
        - "a photograph of {}"
        - "an image of {}"
        - "a picture of a {}"
        - "a {} in the wild"
        
        ISLEM ADIMLARI:
        ---------------
        1. Her etiket icin prompt olustur
           ["cat", "dog"] -> ["a photo of cat", "a photo of dog"]
        
        2. Goruntuyu encode et
        
        3. Prompt'lari encode et
        
        4. Cosine similarity hesapla
        
        5. Softmax ile olasiliklara cevir
        
        6. Skora gore sirala
        
        PARAMETRELER:
        -------------
        image : PIL.Image veya np.ndarray
            Analiz edilecek goruntu
            
        labels : List[str]
            Olasi etiket listesi
            Ornek: ["cat", "dog", "bird"]
            
        prompt_template : str
            Prompt sablonu. {} yerine etiket yerlestirilir.
            Varsayilan: "a photo of {}"
        
        DONDURUR:
        ---------
        List[Tuple[str, float]]
            (etiket, skor) ciftleri, skora gore azalan sirada
            Ornek: [("cat", 85.3), ("bird", 10.2), ("dog", 4.5)]
        """
        
        # -----------------------------------------
        # Prompt'lari Olustur
        # -----------------------------------------
        # Template'e her etiketi yerlestir
        # "a photo of {}" + "cat" -> "a photo of cat"
        prompts = [prompt_template.format(label) for label in labels]
        
        # -----------------------------------------
        # Benzerlik Skorlarini Hesapla
        # -----------------------------------------
        scores = self.calculate_similarity(image, prompts)
        
        # -----------------------------------------
        # Sonuclari Formatla ve Sirala
        # -----------------------------------------
        # Etiket-skor ciftleri olustur
        results = list(zip(labels, scores))
        
        # Skora gore azalan sirada sirala
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def describe_image(self, 
                       image: Union[Image.Image, np.ndarray],
                       candidate_descriptions: List[str] = None) -> List[Tuple[str, float]]:
        """
        Goruntu icin en uygun aciklamayi bulur.
        
        Bu metod, onceden tanimlanmis aciklama adaylari arasindan
        goruntuye en uygun olanini secer.
        
        FARK: analyze() vs describe_image()
        -----------------------------------
        analyze():
        - Tek kelimelik etiketler
        - Prompt template kullanir
        - Siniflandirma amacli
        
        describe_image():
        - Tam cumle aciklamalar
        - Template kullanmaz
        - Aciklama secme amacli
        
        PARAMETRELER:
        -------------
        image : PIL.Image veya np.ndarray
            Aciklanacak goruntu
            
        candidate_descriptions : List[str], optional
            Aday aciklama listesi
            Belirtilmezse varsayilan genel aciklamalar kullanilir
        
        DONDURUR:
        ---------
        List[Tuple[str, float]]
            (aciklama, skor) ciftleri, skora gore sirali
        """
        
        # Varsayilan genel aciklamalar
        if candidate_descriptions is None:
            candidate_descriptions = [
                "a photograph of a person",        # Insan
                "a photograph of an animal",       # Hayvan
                "a photograph of food",            # Yemek
                "a photograph of a landscape",     # Manzara
                "a photograph of a building",      # Bina
                "a photograph of a vehicle",       # Arac
                "a photograph of text or document", # Belge
                "a photograph of art or painting", # Sanat
                "a screenshot of a computer or phone", # Ekran goruntusu
                "a meme or funny image"            # Meme
            ]
        
        # Dogrudan aciklamalari kullan (prompt template yok)
        scores = self.calculate_similarity(image, candidate_descriptions)
        
        # Sonuclari formatla ve sirala
        results = list(zip(candidate_descriptions, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


# =============================================================================
# VARSAYILAN ETIKET LISTELERI
# =============================================================================
# Farkli analiz modlari icin onceden tanimlanmis etiketler

DEFAULT_LABELS = {
    # Genel kategoriler - her turlu icerik icin
    "genel": [
        "person",      # Insan
        "animal",      # Hayvan
        "food",        # Yemek
        "vehicle",     # Arac
        "building",    # Bina
        "nature",      # Doga
        "object",      # Nesne
        "text",        # Metin
        "art",         # Sanat
        "technology"   # Teknoloji
    ],
    
    # Duygu analizi - goruntuun duygusal icerigi
    "duygular": [
        "happy",       # Mutlu
        "sad",         # Uzgun
        "angry",       # Kizgin
        "surprised",   # Sasirmis
        "neutral",     # Notr
        "funny",       # Komik
        "serious",     # Ciddi
        "romantic",    # Romantik
        "scary",       # Korkutucu
        "peaceful"     # Huzurlu
    ],
    
    # Renk analizi - baskın renkler
    "renkler": [
        "red",         # Kirmizi
        "blue",        # Mavi
        "green",       # Yesil
        "yellow",      # Sari
        "orange",      # Turuncu
        "purple",      # Mor
        "pink",        # Pembe
        "black",       # Siyah
        "white",       # Beyaz
        "colorful"     # Renkli
    ],
    
    # Stil analizi - gorsel stil ve teknik
    "stil": [
        "photograph",     # Fotograf
        "drawing",        # Cizim
        "painting",       # Boyama
        "cartoon",        # Karikatur
        "3D render",      # 3D gorsellestirme
        "sketch",         # Eskiz
        "digital art",    # Dijital sanat
        "vintage photo"   # Eski fotograf
    ],
    
    # Meme turleri - internet icerigi
    "meme_turleri": [
        "reaction meme",   # Tepki memesi
        "text meme",       # Metin memesi
        "animal meme",     # Hayvan memesi
        "movie scene",     # Film sahnesi
        "comic",           # Cizgi roman
        "screenshot",      # Ekran goruntusu
        "wholesome meme",  # Tatli/sirin meme
        "sarcastic meme",  # Alayli meme
        "political meme",  # Politik meme
        "gaming meme"      # Oyun memesi
    ]
}


# =============================================================================
# YARDIMCI FONKSIYONLAR
# =============================================================================

def get_device_info() -> dict:
    """
    Sistem GPU/CPU bilgilerini dondurur.
    
    Bu fonksiyon, kullaniciya hangi donanim uzerinde calistıklarini
    gostermek icin kullanilir.
    
    CUDA NEDIR?
    -----------
    CUDA (Compute Unified Device Architecture), NVIDIA'nin
    GPU programlama platformudur. Derin ogrenme modellerini
    GPU uzerinde calistirmak icin gereklidir.
    
    GPU vs CPU:
    - CPU: 4-16 cekirdek, sirali islem
    - GPU: 1000+ cekirdek, paralel islem
    - Tensor islemleri GPU'da 10-100x daha hizli
    
    DONDURUR:
    ---------
    dict:
        - cuda_available: bool - CUDA kullanilabilir mi?
        - device: str - "cuda" veya "cpu"
        - gpu_name: str veya None - GPU model ismi
        - gpu_memory: str veya None - GPU bellek miktari
    """
    
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    if torch.cuda.is_available():
        # GPU adi
        # Ornek: "NVIDIA GeForce RTX 3090"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        
        # GPU bellek miktari
        # total_memory bytes cinsinden, GB'a cevir
        # Ornek: "24.0 GB"
        memory_bytes = torch.cuda.get_device_properties(0).total_memory
        info["gpu_memory"] = f"{memory_bytes / 1e9:.1f} GB"
    
    return info
