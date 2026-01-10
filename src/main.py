import os
import ujson
from tqdm import tqdm
from transformers import AutoTokenizer
from config import Config
from masker import LegalMasker
from utils import aggressive_legal_cleaner
from multiprocessing import Pool, cpu_count

# Global worker variables
worker_tokenizer = None
worker_masker = None

def init_worker():
    """Her çekirdek (process) başladığında bir kere çalışır."""
    global worker_tokenizer, worker_masker
    # Fast Tokenizer kullandığımızdan emin oluyoruz (add_prefix_space gerekebilir)
    worker_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)
    worker_masker = LegalMasker(
        tokenizer=worker_tokenizer,
        terms_file_path=Config.TERMS_FILE,
        mask_prob=Config.MASK_PROB,
        legal_ratio=Config.LEGAL_FOCUS_RATIO
    )

def process_file(filename):
    file_path = os.path.join(Config.DATA_DIR, filename)
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f_in:
            raw_text = f_in.read()

        # Metin temizliği
        text = aggressive_legal_cleaner(raw_text)
        if len(text) < 50: return []

        # 1. Tüm metni tokenize et (Offset ve Word IDs al)
        encoded = worker_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        
        all_ids = encoded['input_ids']
        all_offsets = encoded['offset_mapping']
        all_word_ids = encoded.word_ids() # [0, 0, 1, 2, 2...]
        
        # Word IDs None kontrolü (Nadir durumlarda None gelebilir, düzeltelim)
        # XLM-R Fast tokenizer genelde None döndürmez ama garanti olsun
        cleaned_word_ids = []
        current_word_id = -1
        for wid in all_word_ids:
            if wid is not None:
                current_word_id = wid
                cleaned_word_ids.append(wid)
            else:
                # Eğer başta None varsa 0 say, yoksa öncekini devam ettir
                cleaned_word_ids.append(max(0, current_word_id))
        all_word_ids = cleaned_word_ids

        # Legal Bitmap (Tüm metin için 1 kere hesapla)
        full_legal_bitmap = worker_masker.get_legal_bitmap(text, all_offsets)

        # 2. Kayar Pencere (Sliding Window)
        step = Config.WINDOW_STRIDE # Config'de 256 veya 512 yapmanı öneririm
        window_size = Config.MAX_SEQ_LEN - 2 # CLS ve SEP için yer ayır

        for i in range(0, len(all_ids), step):
            chunk_ids = all_ids[i : i + window_size]
            chunk_word_ids = all_word_ids[i : i + window_size]
            chunk_bitmap = full_legal_bitmap[i : i + window_size]

            if len(chunk_ids) < 10: continue

            # Special Tokens Ekleme ([CLS] ... [SEP])
            final_input_ids = [worker_tokenizer.cls_token_id] + chunk_ids + [worker_tokenizer.sep_token_id]
            
            # Word ID hizalaması: Special tokenlar için None koyuyoruz
            final_word_ids = [None] + chunk_word_ids + [None]
            
            # Bitmap hizalaması
            final_bitmap = [False] + chunk_bitmap + [False]

            # Maskeleme
            masked_input, labels = worker_masker.mask_segment(
                final_input_ids, 
                final_word_ids, 
                final_bitmap
            )

            record = {
                "input_ids": masked_input,
                "labels": labels
            }
            results.append(ujson.dumps(record))

    except Exception as e:
        print(f"Dosya hatası ({filename}): {e}")
        return []

    return results

def main():
    os.makedirs(os.path.dirname(Config.OUTPUT_FILE), exist_ok=True)
    
    files = [f for f in os.listdir(Config.DATA_DIR) if f.endswith(".txt")]
    # Test için dosya sayısını kısıtlayabilirsin: files = files[:100]
    
    num_cores = max(1, cpu_count() - 1)
    print(f"Model: {Config.MODEL_NAME}")
    print(f"İşlem başlıyor: {len(files)} dosya, {num_cores} çekirdek ile işlenecek.")
    
    total_samples = 0
    with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        with Pool(processes=num_cores, initializer=init_worker) as pool:
            for result_list in tqdm(pool.imap_unordered(process_file, files), total=len(files)):
                if result_list:
                    for line in result_list:
                        f_out.write(line + "\n")
                        total_samples += 1

    print(f"\n--- TAMAMLANDI ---")
    print(f"Toplam Üretilen Örnek: {total_samples}")

if __name__ == "__main__":
    main()