import os
import ujson as json
import re
import unicodedata
from config import Config
from multiprocessing import Pool, cpu_count

# ============ TÜRKÇE KARAKTER DESTEĞİ ============
TURKISH_UPPER = "A-ZÇĞİÖŞÜ"
TURKISH_LOWER = "a-zçğıöşü"
TURKISH_ALL = TURKISH_UPPER + TURKISH_LOWER

# ============ YAPILANDIRMA ============
INPUT_DIR = os.path.join(Config.ROOT_DIR, "data", "raw", "jsonl")
OUTPUT_DIR = os.path.join(Config.ROOT_DIR, "data", "resources", "decisions")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ NORMALIZASYON ============

def normalize_turkish_text(text):
    """
    Türkçe karakterleri koruyarak normalize et
    """
    if not text:
        return ""
    
    # Unicode normalize (NFKC - compatibility decomposition)
    # Bu özel karakterleri standart forma getirir ama Türkçe karakterleri korur
    text = unicodedata.normalize('NFKC', text)
    
    # Kaçış dizilerini gerçek karakterlere çevir
    escape_map = {
        '\\n': '\n',
        '\\t': '\t',
        '\\r': '\r',
        '\\\\': '\\',
        '\\/': '/',  # \/ -> /
        '\\"': '"',
        "\\'": "'"
    }
    
    for escaped, real in escape_map.items():
        text = text.replace(escaped, real)
    
    return text

def remove_control_chars(text):
    """
    Kontrol karakterlerini temizle (ama Türkçe karakterleri koru)
    """
    # Sadece gerçek kontrol karakterlerini kaldır (0-31 arası, tab hariç)
    # ve 127+ kontrol karakterleri
    result = []
    for char in text:
        code = ord(char)
        # Tab (9), LF (10), CR (13) hariç 0-31 arası kontrol karakterleri
        if code < 32 and code not in (9, 10, 13):
            continue
        # 127+ kontrol karakterleri (DEL ve sonrası)
        if code == 127:
            continue
        # Zero-width ve diğer görünmez karakterler
        if unicodedata.category(char) in ('Cc', 'Cf', 'Cs', 'Co', 'Cn'):
            if code not in (0x200C, 0x200D):  # Zero-width non-joiner/joiner (Türkçe için gerekli olabilir)
                continue
        result.append(char)
    
    return ''.join(result)

def fix_broken_spacing_v2(text):
    """
    Gelişmiş boşluk düzeltme - Türkçe karakterleri koruyarak
    """
    if not text:
        return ""
    
    # 1. T. C. -> T.C., T.C. -> T.C. (noktalı kısaltmalar)
    # Türkçe karakterleri de kapsa
    text = re.sub(
        rf'(?<=\.)\s+(?=[{TURKISH_UPPER}]\.)', 
        '', 
        text
    )
    
    # 2. Ayrık harfleri birleştir: T U T U K L U -> TUTUKLU
    # En az 3 harf ayrık yazılmışsa
    def replacer(match):
        return match.group(0).replace(" ", "").replace("\t", "")
    
    # Büyük harfli ayrık kelimeler (T U T U K L U)
    pattern_upper = rf'\b(?:[{TURKISH_UPPER}]\s+){{2,}}[{TURKISH_UPPER}]\b'
    text = re.sub(pattern_upper, replacer, text)
    
    # Küçük harfli ayrık kelimeler (t u t u k l u)
    pattern_lower = rf'\b(?:[{TURKISH_LOWER}]\s+){{2,}}[{TURKISH_LOWER}]\b'
    text = re.sub(pattern_lower, replacer, text)
    
    # 3. Rakam-harf ayrıklarını düzelt: 5 2 3 7 -> 5237
    text = re.sub(r'\b(?:\d\s+){2,}\d\b', replacer, text)
    
    # 4. Karışık durumlar (büyük-küçük karışık ayrık)
    pattern_mixed = rf'\b(?:[{TURKISH_ALL}]\s+){{2,}}[{TURKISH_ALL}]\b'
    text = re.sub(pattern_mixed, replacer, text)
    
    return text

def fix_common_ocr_errors(text):
    """
    Yaygın OCR/dijitalleştirme hatalarını düzelt
    """
    if not text:
        return ""
    
    # Kanun numaralarındaki boşluklar: 5 2 3 7 -> 5237
    text = re.sub(r'(\d)\s+(?=\d)', r'\1', text)
    
    # Tarih formatları: 2 0 2 4 -> 2024
    text = re.sub(r'(\d{1})\s+(\d{1})\s+(\d{1})\s+(\d{1})', r'\1\2\3\4', text)
    
    # Madde/fıkra referansları: m. 1 2 3 -> m.123
    text = re.sub(r'(m\.?|madde)\s*(\d)\s+(\d)\s+(\d)', r'\1\2\3\4', text, flags=re.IGNORECASE)
    
    # Kanun kısaltmaları: T C K -> TCK, T M K -> TMK
    text = re.sub(r'\bT\s*C\s*K\b', 'TCK', text, flags=re.IGNORECASE)
    text = re.sub(r'\bT\s*M\s*K\b', 'TMK', text, flags=re.IGNORECASE)
    text = re.sub(r'\bC\s*M\s*K\b', 'CMK', text, flags=re.IGNORECASE)
    text = re.sub(r'\bH\s*M\s*K\b', 'HMK', text, flags=re.IGNORECASE)
    
    # Yargıtay/Danıştay referansları
    text = re.sub(r'\bY\s*2\s*\.\s*H\s*D\b', 'Y.2.HD.', text, flags=re.IGNORECASE)
    text = re.sub(r'\bY\s*\d+\s*H\s*D\b', lambda m: m.group(0).replace(' ', ''), text, flags=re.IGNORECASE)
    
    return text

def clean_legal_references(text):
    """
    Hukuki referansları standardize et
    """
    if not text:
        return ""
    
    # \/ -> / (JSON escape düzeltmesi)
    text = text.replace(r'\/', '/')
    
    # Kanun numaralarını standardize et: 5237sayılı -> 5237 sayılı
    text = re.sub(r'(\d{3,4})(sayılı|sayili|Sayılı)', r'\1 sayılı', text, flags=re.IGNORECASE)
    
    # Madde/fıkra standardizasyonu
    text = re.sub(r'(\d+)\s*\.\s*maddes?\.?', r'\1. maddesi', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*\.\s*fıkras?\.?', r'\1. fıkrası', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*\.\s*bend?\.?', r'\1. bendi', text, flags=re.IGNORECASE)
    
    # Tarih standardizasyonu: 05.03.2013 -> 05.03.2013 (boşlukları temizle)
    text = re.sub(r'(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})', r'\1.\2.\3', text)
    
    return text

def extract_content(text):
    """
    Kararın esas içeriğini çıkar
    """
    if not text:
        return None
    
    # Normalize önce
    text = normalize_turkish_text(text)
    
    # "İçtihat Metni" bölümünü bul
    patterns = [
        r'[İi]çtihat\s*[Mm]etni\s*:?\s*(.*)',
        r'[Kk]arar\s*:?\s*(.*)',
        r'[Hh]üküm\s*:?\s*(.*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Eğer içerik çok kısaysa, tüm metni al
            if len(content) > 100:
                return content
    
    # Hiçbiri bulunamazsa, başlık kısımlarını at
    # İlk 500 karakteri at (genellikle meta veri)
    if len(text) > 600:
        return text[500:].strip()
    
    return text.strip()

def final_cleanup(text):
    """
    Son temizlik adımları
    """
    if not text:
        return None
    
    # Satır sonlarını boşluğa çevir
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Baş-son boşlukları temizle
    text = text.strip()
    
    # Tırnak ve özel karakter temizliği
    text = text.strip('"').strip("'").strip('`')
    
    return text

def clean_text_enhanced(text):
    """
    Ana temizleme fonksiyonu - Tüm adımlar
    """
    if not text:
        return None
    
    # Adım 1: Normalize ve kontrol karakterleri
    text = normalize_turkish_text(text)
    text = remove_control_chars(text)
    
    # Adım 2: İçerik çıkarma
    text = extract_content(text)
    if not text:
        return None
    
    # Adım 3: OCR hataları
    text = fix_common_ocr_errors(text)
    
    # Adım 4: Boşluk düzeltme
    text = fix_broken_spacing_v2(text)
    
    # Adım 5: Hukuki referanslar
    text = clean_legal_references(text)
    
    # Adım 6: Son temizlik
    text = final_cleanup(text)
    
    # Adım 7: Kalite kontrol
    if len(text) < 50:  # Çok kısa
        return None
    
    # Türkçe karakter oranı kontrolü
    turkish_chars = len(re.findall(f'[{TURKISH_LOWER}]', text, re.IGNORECASE))
    total_chars = len(re.findall(r'[a-zA-Z]', text))
    if total_chars > 0 and turkish_chars / total_chars < 0.5:
        # Türkçe karakter oranı düşükse uyarı (ama yine de kaydet)
        pass
    
    return text

# ============ DOSYA İŞLEME ============

def process_file_enhanced(filename):
    """
    Tek dosyayı işle (multiprocessing için)
    """
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    stats = {'read': 0, 'saved': 0, 'skipped': 0, 'errors': 0}
    
    try:
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                
                stats['read'] += 1
                
                try:
                    record = json.loads(line)
                    raw_text = record.get("text", "")
                    doc_id = record.get("id", f"unknown_{stats['read']}")
                    
                    # ENHANCED CLEANING
                    cleaned_text = clean_text_enhanced(raw_text)
                    
                    if cleaned_text:
                        new_record = {
                            "id": doc_id,
                            "text": cleaned_text,
                            "original_length": len(raw_text),
                            "cleaned_length": len(cleaned_text)
                        }
                        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                        stats['saved'] += 1
                    else:
                        stats['skipped'] += 1
                        
                except Exception as e:
                    stats['errors'] += 1
                    if stats['errors'] <= 5:  # İlk 5 hatayı göster
                        print(f"  ⚠️  Hata (satır {stats['read']}): {e}")
                    continue
                    
    except Exception as e:
        print(f"  ❌ Dosya hatası ({filename}): {e}")
        return filename, stats
    
    return filename, stats

def main():
    print("=" * 60)
    print("ENHANCED TURKISH LEGAL DATA CLEANER")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("-" * 60)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    
    if not files:
        print("❌ JSONL dosyası bulunamadı!")
        return
    
    print(f"📁 {len(files)} dosya bulundu\n")
    
    # Multiprocessing
    num_workers = max(1, cpu_count() - 1)
    print(f"⚙️  {num_workers} worker ile işleniyor...\n")
    
    total_stats = {'read': 0, 'saved': 0, 'skipped': 0, 'errors': 0}
    
    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(process_file_enhanced, files)
        
        for filename, stats in results:
            print(f"✅ {filename:40s} | "
                  f"Okunan: {stats['read']:5d} | "
                  f"Kaydedilen: {stats['saved']:5d} | "
                  f"Atlanan: {stats['skipped']:4d}")
            
            for key in total_stats:
                total_stats[key] += stats[key]
    
    print("\n" + "=" * 60)
    print("GENEL İSTATISTIK")
    print("=" * 60)
    print(f"📖 Toplam okunan:    {total_stats['read']:,}")
    print(f"💾 Toplam kaydedilen: {total_stats['saved']:,}")
    print(f"⏭️  Toplam atlanan:    {total_stats['skipped']:,}")
    print(f"❌ Toplam hata:       {total_stats['errors']:,}")
    
    if total_stats['read'] > 0:
        retention = total_stats['saved'] / total_stats['read'] * 100
        print(f"📊 Retention oranı:   %{retention:.1f}")
    
    print("\n✅ Temizlik tamamlandı! Masker'ı çalıştırabilirsiniz.")

if __name__ == "__main__":
    main()