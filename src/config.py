import os

class Config:
    # Proje kök dizini
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Veri dizinleri
    DATA_DIR = os.path.join(ROOT_DIR, "data", "resources", "decisions")
    TERMS_FILE = os.path.join(ROOT_DIR, "data", "resources", "terms", "legal_terms.txt")
    OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "processed", "train_data_wwm.jsonl")
    STATE_FILE = os.path.join(ROOT_DIR, "data", "processed", "masking_state.json")

    # Model ayarları
    MODEL_NAME = "FacebookAI/xlm-roberta-base"  # RoBERTa modeli

    # Tokenizasyon parametreleri
    MAX_SEQ_LEN = 512  # Modelin maksimum giriş uzunluğu
    WINDOW_STRIDE = 256  # Kayar pencere adımı (%50 Overlap)

    # Maskeleme parametreleri
    MASK_PROB = 0.20  # Toplam maskeleme oranı
    LEGAL_FOCUS_RATIO = 0.60  # Maskeleme bütçesinin ne kadarı hukuk terimlerine gidecek?

    # Seed (Tekrarlanabilirlik için)
    SEED = 42