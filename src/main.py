import os
import ujson
from tqdm import tqdm
from transformers import AutoTokenizer
from config import Config
from masker import SmartLegalMasker
from multiprocessing import Pool, cpu_count
import random
import numpy as np

# Global worker state
worker_tokenizer = None
worker_masker = None

def init_worker():
    """Worker başlangıç"""
    global worker_tokenizer, worker_masker
    
    # Worker başına unique seed
    worker_pid = os.getpid()
    random.seed(Config.BASE_SEED + worker_pid)
    np.random.seed(Config.BASE_SEED + worker_pid)
    
    worker_tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME, 
        use_fast=True,
        model_max_length=10**6
    )
    
    worker_masker = SmartLegalMasker(
        tokenizer=worker_tokenizer,
        terms_file_path=Config.TERMS_FILE
    )

def process_file(filename):
    """Tek dosya işleme - generator"""
    input_path = os.path.join(Config.DATA_DIR, filename)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = ujson.loads(line)
                    text = data.get("text", "")
                except (ValueError, KeyError):
                    continue
                
                if len(text) < 50:
                    continue

                # Tokenizasyon
                encoded = worker_tokenizer(
                    text, 
                    add_special_tokens=False, 
                    return_offsets_mapping=True
                )
                all_ids = encoded['input_ids']
                all_offsets = encoded['offset_mapping']
                all_word_ids = encoded.word_ids()

                # Word ID temizleme (-1 = special/ignore)
                cleaned_word_ids = []
                last_real_wid = -1
                for wid in all_word_ids:
                    if wid is not None:
                        last_real_wid = wid
                        cleaned_word_ids.append(wid)
                    else:
                        cleaned_word_ids.append(-1)

                # Legal map - YENİ FONKSİYON ADI
                legal_map, phrase_spans = worker_masker.get_legal_word_map(
                    text, all_offsets, cleaned_word_ids
                )

                # Sliding window
                step = Config.WINDOW_STRIDE
                window_size = Config.MAX_SEQ_LEN - 2

                for i in range(0, len(all_ids), step):
                    chunk_ids = all_ids[i : i + window_size]
                    chunk_word_ids = cleaned_word_ids[i : i + window_size]
                    
                    if len(chunk_ids) < 10:
                        continue

                    # CLS/SEP ekle
                    final_input_ids = [worker_tokenizer.cls_token_id] + chunk_ids + [worker_tokenizer.sep_token_id]
                    final_word_ids = [-1] + chunk_word_ids + [-1]

                    # Maskeleme
                    masked_input, labels = worker_masker.mask_segment_optimized(
                        final_input_ids, 
                        final_word_ids, 
                        legal_map,
                        phrase_spans
                    )

                    record = {
                        "input_ids": masked_input,
                        "labels": labels
                    }
                    
                    yield ujson.dumps(record)

    except Exception as e:
        print(f"HATA ({filename}): {e}")
        import traceback
        traceback.print_exc()  # Detaylı hata için
        return

def _process_file_wrapper(filename):
    """Multiprocessing wrapper"""
    return list(process_file(filename))

def main():
    os.makedirs(os.path.dirname(Config.OUTPUT_FILE), exist_ok=True)
    
    files = [f for f in os.listdir(Config.DATA_DIR) if f.endswith(".jsonl")]
    num_cores = max(1, cpu_count() - 1)
    
    print(f"--- SMART LEGAL MASKING ---")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Terms: {Config.TERMS_FILE}")
    print(f"Files: {len(files)} | Workers: {num_cores}")
    print(f"Config: MASK={Config.BASE_MASK_PROB}, LEGAL_RATIO={Config.BASE_LEGAL_RATIO}, PHRASE_PRIO={Config.PHRASE_PRIORITY}")
    print("-" * 50)
    
    total_samples = 0
    
    with open(Config.OUTPUT_FILE, 'w', encoding='utf-8', buffering=8192) as f_out:
        
        if Config.USE_MULTIPROCESSING:
            with Pool(processes=num_cores, initializer=init_worker) as pool:
                for result_list in tqdm(
                    pool.imap_unordered(_process_file_wrapper, files, chunksize=1),
                    total=len(files)
                ):
                    for line in result_list:
                        f_out.write(line + "\n")
                        total_samples += 1
        else:
            # Single process (debug)
            init_worker()
            for filename in tqdm(files):
                for line in process_file(filename):
                    f_out.write(line + "\n")
                    total_samples += 1

    print(f"\n--- TAMAMLANDI ---")
    print(f"Toplam: {total_samples:,} samples")
    print(f"Output: {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()