import unicodedata
import re


def normalize_text_for_search(text):
    """
    Aho-Corasick araması için metni 'Gölge Kopya'ya çevirir.
    Orijinal metni bozmaz, sadece arama için standartlaştırır.
    """
    if not text:
        return ""
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r"[\-\'\'\.]", "", text)
    return text


def aggressive_legal_cleaner(text):
    """
    Hukuk metinlerindeki OCR/Format hatalarını (yapışık kelimeler, noktalama) temizler.
    Modelin 'KARARDava' gibi yapışık tokenları ezberlemesini önler.

    V2: Genişletilmiş temizlik kuralları
    """
    if not text:
        return ""

    # ============================================================
    # BÖLÜM 1: OCR KARAR FORMATI DÜZELTMESİ
    # ============================================================
    # "K A R A R", "KA R A R", "_ K A R A R _" gibi tüm varyasyonları düzelt
    # Önce alt çizgileri temizle
    text = re.sub(r'_\s*K\s*A\s*R\s*A\s*R\s*_', 'KARAR', text)
    # Herhangi bir boşluk kombinasyonu ile ayrılmış "K A R A R" formatını yakala
    text = re.sub(r'\bK\s*A\s*R\s*A\s*R\b', 'KARAR', text)
    text = re.sub(r'\bK\s+A\s+R\s+A\s+R\b', 'KARAR', text)
    # "KA R A R" gibi kısmi boşluklu versiyonlar
    text = re.sub(r'\bK\s*A\s*R\s*A\s*R\b', 'KARAR', text)
    text = re.sub(r'\bKA\s+R\s+A\s+R\b', 'KARAR', text)
    text = re.sub(r'\bK\s+AR\s+AR\b', 'KARAR', text)

    # Benzer şekilde "S O N U Ç" formatını düzelt (tüm varyasyonlar)
    text = re.sub(r'\bS\s*O\s*N\s*U\s*[ÇC]\b', 'SONUÇ', text)
    text = re.sub(r'\bS\s+O\s+N\s+U\s+[ÇC]\b', 'SONUÇ', text)

    # ============================================================
    # BÖLÜM 2: TIRNAK SONRASI YAPIŞIKLIK
    # ============================================================
    # "İçtihat Metni"MAHKEMESİ -> "İçtihat Metni" MAHKEMESİ
    text = re.sub(r'"([^"]+)"([A-ZÇĞİÖŞÜ])', r'"\1" \2', text)

    # ============================================================
    # BÖLÜM 3: TARİH/NUMARA SONRASI YAPIŞIKLIK
    # ============================================================
    # 02/04/2013NUMARASI -> 02/04/2013 NUMARASI
    # Tarih formatı: DD/MM/YYYY veya YYYY sonrası büyük harf
    text = re.sub(r'(\d{2}/\d{2}/\d{4})([A-ZÇĞİÖŞÜ])', r'\1 \2', text)
    text = re.sub(r'(\d{4})([A-ZÇĞİÖŞÜ][a-zçğıöşü])', r'\1 \2', text)

    # 2012/535-2013/200Davacı -> 2012/535-2013/200 Davacı
    text = re.sub(r'(\d+/\d+)([A-ZÇĞİÖŞÜ][a-zçğıöşü])', r'\1 \2', text)

    # ============================================================
    # BÖLÜM 4: BÜYÜK HARF YAPIŞTIKLARI
    # ============================================================
    # MahkemesiTARİHİ -> Mahkemesi TARİHİ (küçük harf + BÜYÜK HARF dizisi)
    text = re.sub(r'([a-zçğıöşü])([A-ZÇĞİÖŞÜ]{2,})', r'\1 \2', text)

    # KARARDava -> KARAR Dava (2+ BÜYÜK HARF + Büyük+küçük)
    text = re.sub(r'([A-ZÇĞİÖŞÜ]{2,})([A-ZÇĞİÖŞÜ][a-zçğıöşü])', r'\1 \2', text)

    # ============================================================
    # BÖLÜM 5: NOKTALAMA SONRASI YAPIŞIKLIK
    # ============================================================
    # gerekmiştir.1-Yapılan -> gerekmiştir. 1-Yapılan (nokta + rakam)
    # Basit yaklaşım: Küçük harf veya boşluk sonrası nokta + rakam
    # Tarih formatlarını (DD.MM.YYYY) korumak için dikkatli ol
    text = re.sub(r'([a-zçğıöşü\s])\.(\d)', r'\1. \2', text)

    # ilişkindir.Davalı -> ilişkindir. Davalı (küçük.Büyük)
    text = re.sub(r'(?<=[a-zçğıöşü])\.(?=[A-ZÇĞİÖŞÜ])', '. ', text)

    # edilmiştir,ancak -> edilmiştir, ancak (virgül sonrası)
    text = re.sub(r',(?=[a-zçğıöşüA-ZÇĞİÖŞÜ])', ', ', text)

    # ============================================================
    # BÖLÜM 6: İKİ NOKTA ÜST ÜSTE SONRASI
    # ============================================================
    # düşünüldü:KARAR -> düşünüldü: KARAR
    text = re.sub(r':(?=[A-ZÇĞİÖŞÜa-zçğıöşü])', ': ', text)

    # ============================================================
    # BÖLÜM 7: PARANTEZ SONRASI YAPIŞIKLIK
    # ============================================================
    # (E)ve -> (E) ve
    text = re.sub(r'\)(?=[a-zçğıöşü])', ') ', text)
    # (K)SUÇ -> (K) SUÇ
    text = re.sub(r'\)(?=[A-ZÇĞİÖŞÜ])', ') ', text)

    # ============================================================
    # BÖLÜM 8: SAYI FORMATLARI
    # ============================================================
    # 6.000, 00 -> 6.000,00 (para formatı düzeltme)
    text = re.sub(r'(\d+),\s+(\d{2})\b', r'\1,\2', text)

    # ============================================================
    # BÖLÜM 9: NOKTA-TARİH DÜZELTMELERİ (Tarihleri koruma)
    # ============================================================
    # ". 03. 2014" gibi bozulmuş formatları düzelt -> ".03.2014"
    # Ancak "03. 2014" formatı korunmalı - dikkatli ol
    # Bu kural riski yüksek, şimdilik kapalı

    # ============================================================
    # BÖLÜM 10: GEREKSIZ BOŞLUKLARI TEMİZLE
    # ============================================================
    # Çoklu boşlukları tek boşluğa indir
    text = re.sub(r'\s+', ' ', text).strip()

    return text