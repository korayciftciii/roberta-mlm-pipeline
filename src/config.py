import os

class Config:
    # Proje kök dizini
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Veri dizinleri
    DATA_DIR = os.path.join(ROOT_DIR, "data", "resources", "decisions")
    TERMS_FILE = os.path.join(ROOT_DIR, "data", "resources", "terms", "legal_terms.txt")
    OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "processed", "train_data_wwm.jsonl")

    # Model ayarları
    MODEL_NAME = "FacebookAI/xlm-roberta-base"

    # Tokenizasyon parametreleri
    MAX_SEQ_LEN = 512
    WINDOW_STRIDE = 256

    # Maskeleme parametreleri (Dinamik - bunlar maksimum değerler)
    BASE_MASK_PROB = 0.20
    BASE_LEGAL_RATIO = 0.60
    
    # Dinamik maskeleme sınırları
    MIN_MASK_PROB = 0.12
    MAX_MASK_PROB = 0.28
    MIN_LEGAL_RATIO = 0.35
    MAX_LEGAL_RATIO = 0.80

    # Seed
    BASE_SEED = 42
    
    # Performance
    USE_MULTIPROCESSING = True