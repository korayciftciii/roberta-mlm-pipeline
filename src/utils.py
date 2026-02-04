import unicodedata
import re

def normalize_text_for_search(text):
    """
    Aho-Corasick araması için metni standartlaştırır.
    KRİTİK KURAL: Girdi ve çıktı uzunluğu ASLA değişmemelidir.
    Karakter silmek yok, sadece dönüşüm var.
    """
    if not text:
        return ""
    
    # 1. Türkçe Karakter Dönüşümü (Uzunluk Korunur)
    # Python'un standart lower()'ı "İ"yi "i" yaparken bazen byte boyutu değişebilir 
    # ama string length genelde korunur. Yine de elle yapmak en güvenlisidir.
    text = text.replace("İ", "i").replace("I", "ı").lower()
    
    # 2. Şapkalı Karakterler (Accent Removal) - OPSİYONEL
    # Yargıtay kararlarında "kâğıt" ve "kağıt" karışık kullanılır.
    # Şapkayı kaldırmak eşleşme şansını artırır.
    # ANCAK: Şapka kalkınca uzunluk değişmemeli.
    # unicodedata.normalize('NFD', text) bazen karakteri böler (a + ^).
    # Biz basitçe replace yapalım, garanti olsun.
    
    replacements = {
        "â": "a", "î": "i", "û": "u",
        "Â": "a", "Î": "i", "Û": "u"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 3. Noktalama İşaretleri: SİLMEK YOK!
    # "bi-hükm'ül-kanun" terimini bulmak istiyoruz.
    # Eğer metinde tire varsa, ararken de tire olmalı. 
    # O yüzden re.sub ile silme işlemini TAMAMEN İPTAL EDİYORUZ.
    # Metin olduğu gibi kalsın.
    
    return text