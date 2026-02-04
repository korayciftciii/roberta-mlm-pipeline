import unicodedata
import re

def normalize_text_for_search(text):
  
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.replace("İ", "i").replace("I", "ı").lower()
    
    replacements = {"â": "a", "î": "i", "û": "u", "Â": "a", "Î": "i", "Û": "u"}
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    return text

def fix_broken_spacing(text):
    """
    'T U T U K L U' -> 'TUTUKLU'
    'T. C.' -> 'T.C.'
    Dönüşümlerini yapar.
    """
    if not text: return ""

    # 1. Satırları düzelt
    text = text.replace("\n", " ").replace("\r", " ")

    # 2. T. C. -> T.C. (Nokta sonrası tek harf boşluklarını sil)
    # (?<=\.) : Öncesinde nokta var mı?
    # \s+     : Boşluk
    # (?=[A-ZİĞÜŞÖÇ]\.) : Sonrasında Harf+Nokta var mı?
    text = re.sub(r'(?<=\.)\s+(?=[A-ZİĞÜŞÖÇ]\.)', '', text)
    
    # 3. A Y R I K  H A R F L E R İ  B İ R L E Ş T İ R
    # En az 3 harflik (H A R) zincirleri yakalar.
    def replacer(match):
        return match.group(0).replace(" ", "")

    # Regex: Kelime sınırı -> (Harf + Boşluk) x 2 veya daha fazla -> Harf -> Kelime sınırı
    pattern = r'\b(?:[A-ZİĞÜŞÖÇa-zıüğşöç]\s+){2,}[A-ZİĞÜŞÖÇa-zıüğşöç]\b'
    text = re.sub(pattern, replacer, text)

    # 4. Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def aggressive_legal_cleaner(text):
    # Ana temizleyici fonksiyonumuz artık bu
    return fix_broken_spacing(text)