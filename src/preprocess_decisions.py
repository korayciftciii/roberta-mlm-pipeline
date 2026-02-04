import os
import ujson as json  # Hız için ujson kullandım
import re
import sys
from config import Config

# --- CONFIGURATION ---
# Ham verilerin olduğu yer
INPUT_DIR = os.path.join(Config.ROOT_DIR, "data", "raw", "jsonl")
# Temizlenmiş verilerin gideceği yer (Masker burayı okuyacak)
OUTPUT_DIR = os.path.join(Config.ROOT_DIR, "data", "resources", "decisions")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fix_broken_spacing(text):
    """
    Yargıtay kararlarındaki kronik boşluk sorunlarını çözer.
    """
    # 1. T. C. -> T.C. (Nokta sonrası tek harf boşluklarını sil)
    # (?<=\.) : Öncesinde nokta var mı?
    # \s+     : Boşluk
    # (?=[A-ZİĞÜŞÖÇ]\.) : Sonrasında Harf+Nokta var mı?
    text = re.sub(r'(?<=\.)\s+(?=[A-ZİĞÜŞÖÇ]\.)', '', text)
    
    # 2. T U T U K L U -> TUTUKLU (Ayrık yazılan kelimeleri birleştir)
    # Mantık: En az 3 harf boyunca "Harf+Boşluk" örüntüsü varsa yakala ve birleştir.
    def replacer(match):
        return match.group(0).replace(" ", "")

    # Regex: Kelime sınırı -> (Harf + Boşluk) x 2 veya daha fazla -> Harf -> Kelime sınırı
    # Hem BÜYÜK harfleri hem küçük harfleri kapsar.
    pattern = r'\b(?:[A-ZİĞÜŞÖÇa-zıüğşöç]\s+){2,}[A-ZİĞÜŞÖÇa-zıüğşöç]\b'
    text = re.sub(pattern, replacer, text)

    return text

def clean_text(text):
    """
    Cleans the decision text for WWM.
    """
    if not text:
        return None
    
    # 1. Extraction Strategy (Case Insensitive)
    # "İçtihat Metni" ibaresinden öncesini (başlıkları) at.
    parts = re.split(r"İçtihat Metni", text, maxsplit=1, flags=re.IGNORECASE)
    
    if len(parts) > 1:
        content = parts[1]
    else:
        content = text

    # 2. Whitespace Normalization
    content = content.replace("\n", " ").replace("\t", " ")
    
    # 3. FIX BROKEN SPACING (En Önemli Kısım Burası!)
    content = fix_broken_spacing(content)
    
    # 4. Collapse multiple spaces (Çift boşlukları teke indir)
    content = re.sub(r'\s+', ' ', content)
    
    # 5. Tırnak temizliği
    content = content.strip().strip('"').strip("'").strip()

    return content

def process_file(filename):
    """
    Processes a single JSONL file.
    """
    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = f"{filename}" # Aynı isimle kaydedebiliriz veya prefix ekleyebilirsin
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"\n Processing: {filename}...")
    print(f"   Target: {output_path}")

    total_read = 0
    total_saved = 0
    total_skipped = 0
    
    try:
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                
                total_read += 1
                try:
                    record = json.loads(line)
                    raw_text = record.get("text", "")
                    doc_id = record.get("id")
                    
                    cleaned_text = clean_text(raw_text)
                    
                    # 50 karakterden kısa kararlar gürültüdür, at gitsin.
                    if cleaned_text and len(cleaned_text) > 50: 
                        new_record = {
                            "id": doc_id,
                            "text": cleaned_text
                        }
                        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                        total_saved += 1
                    else:
                        total_skipped += 1
                        
                except ValueError: # json decode error
                    print(f"   ⚠️ Warning: Failed to decode JSON at line {total_read}")
                    continue
                    
    except Exception as e:
        print(f"  Error processing file: {e}")
        return

    print(f" Complete.")
    print(f"      - Read: {total_read}")
    print(f"      - Saved: {total_saved}")
    print(f"      - Skipped: {total_skipped} (Empty/Short)")

def main():
    print("=== Data Preprocessor for WWM Started ===")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create Output Dir if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all .jsonl files
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    
    if not files:
        print("No JSONL files found in input directory.")
        return

    print(f"Found {len(files)} files in queue.\n")
    
    for i, filename in enumerate(files):
        process_file(filename)
    
    print("\n✅ All files processed successfully! Now run main.py")

if __name__ == "__main__":
    main()