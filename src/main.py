import os
import ujson
from tqdm import tqdm
from transformers import AutoTokenizer
from config import Config
from masker import LegalMasker
from multiprocessing import Pool, cpu_count

# Global workers
worker_tokenizer = None
worker_masker = None

def init_worker():
    """Process başlangıcında modelleri yükle"""
    global worker_tokenizer, worker_masker
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
            for line in f_in:
                line = line.strip()
                if not line: continue
                
                try:
                    # JSON parse et
                    data = ujson.loads(line)
                    # Sadece text alanını al, cleaner YOK
                    text = data.get("text", "")
                except ValueError:
                    continue 

                if len(text) < 50: continue

                # 1. Tokenizasyon
                encoded = worker_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                all_ids = encoded['input_ids']
                all_offsets = encoded['offset_mapping']
                all_word_ids = encoded.word_ids()
                
                # Word ID temizleme (None fix)
                cleaned_word_ids = []
                current_word_id = -1
                for wid in all_word_ids:
                    if wid is not None:
                        current_word_id = wid
                        cleaned_word_ids.append(wid)
                    else:
                        cleaned_word_ids.append(max(0, current_word_id))
                all_word_ids = cleaned_word_ids

                # 2. Legal Bitmap (Tüm metin için)
                full_legal_bitmap = worker_masker.get_legal_bitmap(text, all_offsets)

                # 3. Sliding Window
                step = Config.WINDOW_STRIDE
                window_size = Config.MAX_SEQ_LEN - 2 

                for i in range(0, len(all_ids), step):
                    chunk_ids = all_ids[i : i + window_size]
                    chunk_word_ids = all_word_ids[i : i + window_size]
                    chunk_bitmap = full_legal_bitmap[i : i + window_size]

                    if len(chunk_ids) < 10: continue

                    final_input_ids = [worker_tokenizer.cls_token_id] + chunk_ids + [worker_tokenizer.sep_token_id]
                    final_word_ids = [None] + chunk_word_ids + [None]
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
        print(f"HATA ({filename}): {e}")
        return (filename, [])

    return (filename, results)

def main():
    os.makedirs(os.path.dirname(Config.OUTPUT_FILE), exist_ok=True)
    
    # 1. Mevcut State'i Yükle
    processed_files = set()
    if os.path.exists(Config.STATE_FILE):
        try:
            with open(Config.STATE_FILE, "r", encoding="utf-8") as f:
                state_data = ujson.load(f)
                processed_files = set(state_data.get("processed_files", []))
            print(f"State Yüklendi (JSON): {len(processed_files)} dosya işlenmiş.")
        except Exception as e:
            print(f"State yüklenirken hata oluştu: {e}. Sıfırdan başlanıyor.")

    # 2. İşlenecek Dosyaları Belirle
    all_files = [f for f in os.listdir(Config.DATA_DIR) if f.endswith(".jsonl")]
    files_to_process = [f for f in all_files if f not in processed_files]
    
    if not files_to_process:
        print("İşlenecek yeni dosya yok. Tüm dosyalar tamamlanmış.")
        return

    # CPU count - 2 (Sistemi kilitlememesi için)
    num_cores = max(1, cpu_count() - 2)
    
    print(f"--- BAŞLIYOR ---")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Toplam Dosya: {len(all_files)}")
    print(f"İşlenecek: {len(files_to_process)}")
    print(f"Atlanan: {len(processed_files)}")
    print(f"Çekirdek Sayısı: {num_cores}")
    
    total_samples = 0
    
    # Dosya modu: Eğer state varsa 'a' (append), yoksa 'w' (write)
    file_mode = 'a' if processed_files else 'w'
    
    with open(Config.OUTPUT_FILE, file_mode, encoding='utf-8') as f_out:
        with Pool(processes=num_cores, initializer=init_worker) as pool:
            # imap_unordered sonuçları geldikçe işleyelim
            for (filename, result_list) in tqdm(pool.imap_unordered(process_file, files_to_process), total=len(files_to_process)):
                
                # Sonuçları yaz
                if result_list:
                    for line in result_list:
                        f_out.write(line + "\n")
                        total_samples += 1
                
                # State güncelle (RAM'de ve Dosyada)
                processed_files.add(filename)
                
                # Her dosya bitiminde state JSON'ı güncelle (Crash-safe)
                try:
                    with open(Config.STATE_FILE, "w", encoding="utf-8") as f_state:
                        ujson.dump({"processed_files": list(processed_files)}, f_state, indent=2)
                except Exception as e:
                    print(f"State kaydedilemedi: {e}")

    print(f"\n--- BİTTİ ---")
    print(f"Bu oturumda üretilen örnek: {total_samples}")
    print(f"Kaydedildi: {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()