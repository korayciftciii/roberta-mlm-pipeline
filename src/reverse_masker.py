import os
import ujson
from transformers import AutoTokenizer
import colorama
from colorama import Fore, Style, Back
from config import Config

# Renklendirme başlat
colorama.init(autoreset=True)

PROJECT_ROOT = Config.ROOT_DIR
MODEL_NAME = Config.MODEL_NAME
INPUT_FILE = Config.OUTPUT_FILE  
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "test_reports")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reverse_masking_report.txt")

def format_token(token):
    """
    SentencePiece (XLM-R) tokenlarını okunabilir hale getirir.
    U+2581 ( ) karakterini alt tire (_) ile değiştirir.
    """
    return token.replace(" ", "_")

def reverse_masking():
    print(f"{Fore.CYAN}Tokenizer yükleniyor: {MODEL_NAME}...{Style.RESET_ALL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"{Fore.RED}Model yüklenemedi: {e}{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}Veri okunuyor: {INPUT_FILE}{Style.RESET_ALL}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Dosyayı satır satır okuyacağız (Listeye atmadan)
        lines_to_process = []
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for _ in range(50): # İlk 50 örneği al
                line = f.readline()
                if not line: break
                lines_to_process.append(line)
        
        print(f"İlk {len(lines_to_process)} örnek analiz ediliyor.")
        print("=" * 100)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            # Rapor Başlığı
            header = f"REVERSE MASKING & WWM ANALİZ RAPORU\nModel: {MODEL_NAME}\nKaynak: {INPUT_FILE}\n" + "=" * 100 + "\n\n"
            out_f.write(header)

            for i, line in enumerate(lines_to_process):
                try:
                    data = ujson.loads(line)
                    input_ids = data.get("input_ids", [])
                    labels = data.get("labels", [])

                    if not input_ids or not labels: continue

                    # 1. ORİJİNAL METNİ GERİ OLUŞTURMA (RECONSTRUCTION)
                    # Label -100 ise input_ids'i al, değilse (maskeliyse) label'ı al.
                    reconstructed_ids = [
                        lbl if lbl != -100 else inp 
                        for inp, lbl in zip(input_ids, labels)
                    ]

                    # Decode işlemleri
                    masked_text = tokenizer.decode(input_ids, skip_special_tokens=False)
                    original_text = tokenizer.decode(reconstructed_ids, skip_special_tokens=True) # Temiz metin

                    # 2. DETAYLI WWM ANALİZİ (TOKEN BAZLI)
                    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    
                    visual_tokens_console = [] 
                    visual_tokens_file = []
                    target_words_list = []

                    # Maskelenmiş bir bloğu takip etmek için flag
                    current_mask_block = [] 
                    
                    for idx, (token, lbl_id, inp_id) in enumerate(zip(raw_tokens, labels, input_ids)):
                        token_str = format_token(token)

                        # EĞER BU TOKEN MASKELENMİŞSE (Label != -100)
                        if lbl_id != -100:
                            # Hangi tokenın gizlendiğini bul
                            hidden_token = tokenizer.decode([lbl_id])
                            target_words_list.append(hidden_token)

                            # Konsol: Kırmızı Arkaplan + Beyaz Yazı
                            # Dosya: [MASKED_TOKEN] formatı
                            
                            # Input tarafında ne görüyoruz? (MASK token mı, Random token mı?)
                            input_token_str = format_token(tokenizer.convert_ids_to_tokens(inp_id))
                            
                            # Görselleştirme: [Input(Gizli)] şeklinde gösterelim
                            if inp_id == tokenizer.mask_token_id:
                                vis_str = f"[{token_str}]" # Normal maske
                            else:
                                vis_str = f"[{token_str}*]" # Random veya Same replacement

                            visual_tokens_console.append(f"{Back.RED}{Fore.WHITE}{vis_str}{Style.RESET_ALL}")
                            visual_tokens_file.append(vis_str)
                        else:
                            visual_tokens_console.append(f"{Fore.LIGHTBLACK_EX}{token_str}{Style.RESET_ALL}")
                            visual_tokens_file.append(token_str)

                    # Çıktı Stringleri
                    visual_raw_str = " ".join(visual_tokens_console)
                    file_raw_str = " ".join(visual_tokens_file)
                    targets_str = ", ".join(target_words_list)

                    # --- EKRAN ÇIKTISI ---
                    print(f"{Fore.CYAN}ÖRNEK #{i + 1}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Maskeli Input (Text):{Style.RESET_ALL} {masked_text}")
                    print(f"{Fore.BLUE}WWM Token Analizi:   {Style.RESET_ALL} {visual_raw_str}") 
                    print(f"{Fore.GREEN}Orijinal Text:       {Style.RESET_ALL} {original_text}")
                    print(f"{Fore.MAGENTA}Maskelenen Kelimeler:{Style.RESET_ALL} {targets_str}")
                    print("-" * 100)

                    # --- DOSYA ÇIKTISI ---
                    out_f.write(f"ÖRNEK #{i + 1}\n")
                    out_f.write(f"Maskeli Input: {masked_text}\n")
                    out_f.write(f"WWM Analizi:   {file_raw_str}\n")
                    out_f.write(f"Orijinal Text: {original_text}\n")
                    out_f.write(f"Hedefler:      {targets_str}\n")
                    out_f.write("-" * 100 + "\n")

                except Exception as inner_e:
                    print(f"Satır {i} işlenirken hata: {inner_e}")
                    continue

        print(f"\n{Fore.GREEN}✓ İşlem Tamamlandı!{Style.RESET_ALL}")
        print(f"Detaylı rapor şuraya kaydedildi: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"{Fore.RED}Hata: Dosya bulunamadı -> {INPUT_FILE}{Style.RESET_ALL}")
        print("Lütfen önce 'main.py' dosyasını çalıştırarak veriyi üretin.")
    except Exception as e:
        print(f"{Fore.RED}Genel Hata: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    reverse_masking()