
import re
import os
import unicodedata

def normalize_term(text):
    text = text.strip()
    # Remove leading/trailing punctuation
    text = text.strip('.,;()[]"\'')
    # Unicode normalize
    text = unicodedata.normalize('NFKC', text)
    return text

def check_raw_file(input_path):
    print(f"Checking raw file: {input_path}")
    found = False
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if "üküm" in line: # Partial match
                print(f"  Line {i+1}: {repr(line)}")
                found = True
                if i > 800: break # Sadece ilk bulduklarını yaz
    if not found:
        print("Raw fileda 'üküm' bile yok!")

def clean_legal_terms(input_path, output_path):
    print(f"Processing {input_path}...")
    
    new_terms = set()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Unicode fix
            line = unicodedata.normalize('NFKC', line)
            
            # 1. Mevcut kısa terimleri koru
            if len(line.split()) < 10:
                # Temizle ama stringi bozma
                cleaned = normalize_term(line)
                if len(cleaned) >= 2:
                    new_terms.add(cleaned)
                continue

            # 2. Uzun satırları parçala
            parts = re.split(r'[,;()]', line)
            for part in parts:
                part = part.strip()
                if not part: continue
                cleaned = normalize_term(part)
                if len(cleaned) < 3: continue
                new_terms.add(cleaned)

    # Manually seeded legal terms to ensure coverage
    manual_terms = [
        "Yargıtay", "Danıştay", "Anayasa Mahkemesi", "Bölge Adliye Mahkemesi", "Cumhuriyet Başsavcılığı", 
        "Ağır Ceza Mahkemesi", "Asliye Ceza Mahkemesi", "Sulh Ceza Hakimliği", "İcra Hukuk Mahkemesi", 
        "Asliye Hukuk Mahkemesi", "İddianame", "Kovuşturma", "Soruşturma", "Beraat", "Mahkumiyet", 
        "Hükmün Açıklanmasının Geri Bırakılması", "Takipsizlik", "İstinaf", "Temyiz", "Karar Düzeltme", 
        "Kanun Yararına Bozma", "Müdafii", "Sanık", "Mağdur", "Müşteki", "Katılan", "Tanık", "Bilirkişi", 
        "Keşif", "Duruşma", "Gerekçeli Karar", "Kıdem Tazminatı", "İhbar Tazminatı", "Kötüniyet Tazminatı", 
        "Maddi Tazminat", "Manevi Tazminat", "Boşanma", "Velayet", "Nafaka", "Mal Rejimi", "Miras", 
        "Vasiyetname", "Tapu İptali ve Tescil", "Men'i Müdahale", "Ecrimisil", "Kamulaştırma", 
        "Ortaklığın Giderilmesi", "İhtiyati Tedbir", "İhtiyati Haciz", "İcra Takibi", "Ödeme Emri", 
        "Haciz", "Yakalama", "Gözaltı", "Tutuklama", "Tahliye", "Adli Kontrol", "Uzlaşma", "Önödeme", 
        "Tekerrür", "Haksız Tahrik", "Meşru Müdafaa", "İştirak", "Teşebbüs", "Gönüllü Vazgeçme", 
        "Etkin Pişmanlık", "Zincirleme Suç", "Fikri İçtima"
    ]
    
    for Term in manual_terms:
        # Normalize and add manual terms
        norm_term = unicodedata.normalize('NFKC', Term)
        new_terms.add(norm_term)

    final_terms = sorted(list(new_terms), key=len, reverse=True)
    
    print(f"New term count: {len(final_terms)}")
    
    # Hüküm kontrolü
    hukum_exists = any("hüküm" in t.lower() for t in final_terms)
    print(f"'Hüküm' kelimesi listede var mı (case-insensitive)? {hukum_exists}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for term in final_terms:
            f.write(term + "\n")

if __name__ == "__main__":
    input_file = r"c:\Workspace\hammurabi\RoBERTa-Further-PreTrain\data\raw\terms\raw_terms.txt"
    output_file = r"c:\Workspace\hammurabi\RoBERTa-Further-PreTrain\data\resources\terms\legal_terms_cleaned.txt"
    
    check_raw_file(input_file)
    clean_legal_terms(input_file, output_file)
