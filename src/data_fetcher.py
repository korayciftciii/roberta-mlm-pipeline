import os
import json
import time
import re
import requests
import random
from datetime import datetime
from config import Config

BASE_URL = os.environ.get("BASE_URL")  
INDEX_NAME = os.environ.get("INDEX_NAME")       
API_TOKEN = os.environ.get("API_TOKEN") 


TARGETS = {
    "yargitay": 400,
    "danistay": 200,
    "bam": 200,
    "first_degree": 200
}


DELAY_MIN = 1.5  # Min saniye bekleme
DELAY_MAX = 3.0  # Max saniye bekleme


PROJECT_ROOT = Config.ROOT_DIR
QUERY_FILE = os.path.join(PROJECT_ROOT, "data", "test", "final_queries.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "json")
STATE_FILE = "fetch_state.json"

# -----------------------------------------------

def clean_filename(text):
    text = str(text).replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[-\s]+', '_', text)[:50] # Çok uzun isimleri kes

def load_json(filepath, default=None):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_from_meili(query, court_filter, limit=5):
    """
    Meilisearch API'sine güvenli istek atar.
    """
    url = f"{BASE_URL}/indexes/{INDEX_NAME}/search"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    payload = {
        "q": query,
        "filter": [f'high_court = "{court_filter}"'],
        "limit": limit,
        # Veri trafiğini azaltmak için sadece gerekenleri çekiyoruz
        "attributesToRetrieve": ["id", "title", "text", "decisionContent", "high_court"]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            return response.json().get("hits", [])
        elif response.status_code == 401:
            print("\nHATA: Yetkisiz Erişim (401). Token'ı kontrol et!")
            return []
        elif response.status_code == 429:
            print("\nÇok fazla istek (429). 10 saniye soğuma...")
            time.sleep(10)
            return []
        else:
            print(f"\nAPI Hatası ({response.status_code}): {response.text}")
            return []
            
    except Exception as e:
        print(f"\nBağlantı Hatası: {e}")
        return []

def main():
    print(" Çekme Motoru Başlatılıyor...")
    print(f"Hedef Index: {INDEX_NAME}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Kaynakları Yükle
    if not os.path.exists(QUERY_FILE):
        print(f" HATA: {QUERY_FILE} bulunamadı. Önce query oluşturma adımını yapmalısın.")
        return

    queries = load_json(QUERY_FILE, [])
    
    # 2. State Yükle (Kaldığımız Yer)
    state = load_json(STATE_FILE, {
        "counts": {k: 0 for k in TARGETS}, 
        "processed_queries": [],           
        "seen_ids": []                     
    })
    
    # Performans için set'e çevir
    seen_ids_set = set(state["seen_ids"])
    processed_queries_set = set(state["processed_queries"])
    
    print(f"Mevcut İlerleme: {state['counts']}")
    
    # 3. Ana Döngü
    for i, query in enumerate(queries):
        # A. Genel Hedef Kontrolü
        if all(state["counts"][k] >= TARGETS[k] for k in TARGETS):
            print("\nTEBRİKLER! Tüm hedeflere (1000 Karar) ulaşıldı.")
            break
            
        # B. Bu kelime daha önce işlendi mi?
        if query in processed_queries_set:
            continue
            
        print(f"🔍 [{i+1}/{len(queries)}] Sorgu: '{query}' işleniyor...", end="\r")
        
        any_new_save = False
        
        # C. Her Mahkeme İçin İstek At
        for court_type, target_limit in TARGETS.items():
            # Mahkeme kotası dolduysa atla
            if state["counts"][court_type] >= target_limit:
                continue
                
            # --- DELAY (Sistemi Korumak İçin) ---
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            
            # İstek
            hits = fetch_from_meili(query, court_type, limit=5)
            
            if not hits:
                continue

            # D. Kayıt İşlemleri
            for hit in hits:
                doc_id = str(hit.get("id"))
                
                # Daha önce indirdik mi?
                if doc_id in seen_ids_set:
                    continue
                
                # İçerik Kontrolü (text veya decisionContent)
                content = hit.get("text") or hit.get("decisionContent")
                if not content or len(content) < 50:
                    continue
                
                # JSON Formatı
                record = {
                    "id": doc_id,
                    "title": hit.get("title", "Başlıksız"),
                    "text": content,
                    "meta": {
                        "court": court_type,
                        "query": query,
                        "fetched_at": datetime.now().isoformat()
                    }
                }
                
                # Dosyaya Yaz
                safe_q = clean_filename(query)
                filename = f"{court_type}_{safe_q}_{doc_id}.json"
                save_json(os.path.join(OUTPUT_DIR, filename), record)
                
                # State Güncelle
                seen_ids_set.add(doc_id)
                state["counts"][court_type] += 1
                any_new_save = True
        
        # E. Query Tamamlandı
        processed_queries_set.add(query)
        
        # State Kaydet (Her 5 sorguda bir diske yazalım ki yavaşlamasın)
        if i % 5 == 0 or any_new_save:
            state["seen_ids"] = list(seen_ids_set)
            state["processed_queries"] = list(processed_queries_set)
            save_json(STATE_FILE, state)
            
            # Ekrana durum bas
            if any_new_save:
                print(f"\n   Kaydedildi. Durum: {state['counts']}")

    print("\nProgram Sonlandı.")

if __name__ == "__main__":
    main()