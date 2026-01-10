import json
import os
from transformers import AutoTokenizer
import colorama
from colorama import Fore, Style
from config import Config

#
colorama.init(autoreset=True)


PROJECT_ROOT = Config.ROOT_DIR
MODEL_NAME = Config.MODEL_NAME
INPUT_FILE = Config.OUTPUT_FILE  #
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "test")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reverse_masking_report.txt")

def reverse_masking():
    print(f"{Fore.CYAN}Tokenizer yükleniyor: {MODEL_NAME}...{Style.RESET_ALL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"{Fore.CYAN}Veri okunuyor: {INPUT_FILE}{Style.RESET_ALL}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        
        lines_to_process = lines[:50]
        
        print(f"Toplam {len(lines)} satır veri var. İlk {len(lines_to_process)} örnek analiz ediliyor.")
        print("=" * 100)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            # Rapor Başlığı
            out_f.write(f"REVERSE MASKING & WWM ANALİZ RAPORU\n")
            out_f.write(f"Model: {MODEL_NAME}\n")
            out_f.write(f"Kaynak Dosya: {INPUT_FILE}\n")
            out_f.write("=" * 100 + "\n\n")

            for i, line in enumerate(lines_to_process):
                try:
                    data = json.loads(line)
                    input_ids = data.get("input_ids", [])
                    labels = data.get("labels", [])

                    if not input_ids or not labels:
                        continue

                   
                    # Labels -100 değilse orjinal tokenı al, -100 ise input'tan al
                    reconstructed_ids = [
                        lbl if lbl != -100 else inp 
                        for inp, lbl in zip(input_ids, labels)
                    ]

                    masked_text = tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)
                    original_text = tokenizer.decode(reconstructed_ids, clean_up_tokenization_spaces=False)
                    
                    # Sadece maskelenmiş kelimeleri listele
                    target_words = [tokenizer.decode([lbl]) for lbl in labels if lbl != -100]

                 
                    # Tokenları ham haliyle (raw) alıyoruz. XLM-Rda (sentencepiece) görünür.
                    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    
                    visual_tokens_console = [] # Renkli (Ekrana)
                    visual_tokens_file = []    # Düz (Dosyaya)

                    for idx, token_str in enumerate(raw_tokens):
                        # Eğer bu token maskelendiyse (label != -100)
                        if labels[idx] != -100:
                            # Kırmızı ve belirgin göster
                            visual_tokens_console.append(f"{Fore.RED}[{token_str}]{Style.RESET_ALL}")
                            visual_tokens_file.append(f"[MASK:{token_str}]")
                        else:
                            visual_tokens_console.append(token_str)
                            visual_tokens_file.append(token_str)

                    
                    visual_raw_str = " ".join(visual_tokens_console)
                    file_raw_str = " ".join(visual_tokens_file)

                    
                    # EKRAN ÇIKTISI
                    print(f"{Fore.CYAN}ÖRNEK #{i + 1}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Maskeli Input:{Style.RESET_ALL} {masked_text}")
                    print(f"{Fore.BLUE}WWM Analizi:  {Style.RESET_ALL} {visual_raw_str}") 
                    print(f"{Fore.GREEN}Orijinal Text:{Style.RESET_ALL} {original_text}")
                    print(f"{Fore.MAGENTA}Hedefler:     {Style.RESET_ALL} {target_words}")
                    print("-" * 100)

                    # DOSYA ÇIKTISI
                    out_f.write(f"ÖRNEK #{i + 1}\n")
                    out_f.write(f"Maskeli Input: {masked_text}\n")
                    out_f.write(f"WWM Analizi:   {file_raw_str}\n")
                    out_f.write(f"Orijinal Text: {original_text}\n")
                    out_f.write(f"Hedefler:      {target_words}\n")
                    out_f.write("-" * 100 + "\n")

                except Exception as inner_e:
                    print(f"Satır {i} işlenirken hata: {inner_e}")
                    continue

        print(f"\n{Fore.GREEN}✔ İşlem Tamamlandı!{Style.RESET_ALL}")
        print(f"Rapor dosyası oluşturuldu: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"{Fore.RED}Hata: Dosya bulunamadı -> {INPUT_FILE}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Genel Hata: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    reverse_masking()