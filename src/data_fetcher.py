import os
import json
import time
import re
import requests
import random
from dotenv import load_dotenv
from datetime import datetime
from config import Config
load_dotenv()
# --- CONFIGURATION ---
BASE_URL = os.environ.get("BASE_URL")
INDEX_NAME = os.environ.get("INDEX_NAME")
API_TOKEN = os.environ.get("API_TOKEN")

HIGH_COURT_FILTER = "Yargıtay"

DELAY_MIN = 0.5
DELAY_MAX = 1.0

# --- PATHS ---
PROJECT_ROOT = Config.ROOT_DIR
QUERY_FILE = os.path.join(PROJECT_ROOT, "data", "resources", "queries", "query_by_chamber.json")

# Output directory changed to 'jsonl'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "jsonl")
CORRUPTED_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "corrupted_files.json")
STATE_FILE = "fetch_state.json"

# --- HELPERS ---

def sanitize_filename(text):
    """
    Creates a safe filename from chamber names.
    Ex: "1. Hukuk Dairesi" -> "1_hukuk_dairesi"
    """
    text = str(text).replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text) # Remove punctuation
    return re.sub(r'[-\s]+', '_', text).strip()

def load_json(filepath, default=None):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from {filepath}. Using default.")
            return default
    return default

def save_json(filepath, data):
    # Atomic save could be better, but standard write is fine for state
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_jsonl(filepath, record):
    """
    Appends a single record as a JSON line to the specified file.
    Efficient for large datasets as it doesn't rewriting the whole file.
    """
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def fetch_from_meili(query, filters, limit):
    url = f"{BASE_URL}/indexes/{INDEX_NAME}/search"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    payload = {
        "q": query,
        "filter": filters,
        "limit": limit
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            return response.json().get("hits", [])
        elif response.status_code == 401:
            print("\nError: API Unauthorized Access (401).")
            return []
        elif response.status_code == 429:
            print("\nError: API Too Many Requests (429).")
            time.sleep(10)
            return []
        else:
            print(f"\nAPI Error ({response.status_code}): {response.text}")
            return []
            
    except Exception as e:
        print(f"\n Connection Error: {e}")
        return []

# --- MAIN ---

def main():
    print(" Booting Fetcher (JSONL Mode)...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(QUERY_FILE):
        print(f" Error: {QUERY_FILE} not found.")
        return

    chamber_data = load_json(QUERY_FILE, [])
    
    # State Structure: 
    # { 
    #   "current_index": 0, 
    #   "processed_queries": [], 
    #   "downloaded_files": { "id": {meta} } 
    # }
    state = load_json(STATE_FILE, {"current_index": 0, "processed_queries": [], "downloaded_files": {}})
    
    current_index = state.get("current_index", 0)
    processed_queries_set = set(state.get("processed_queries", []))
    downloaded_files = state.get("downloaded_files", {})
    
    # Load Corrupted Files list
    corrupted_files = load_json(CORRUPTED_FILE, [])
    
    # Build a fast lookup set for all seen IDs (Valid + Corrupted)
    # This prevents re-fetching the same doc even if queries overlap.
    seen_ids_set = set(downloaded_files.keys())
    for cf in corrupted_files:
        if "id" in cf:
            seen_ids_set.add(cf["id"])
    
    total_new_saved = 0
    total_corrupted = 0
    
    print(f" Loaded State: {len(seen_ids_set)} items seen previously.")

    for i, group in enumerate(chamber_data):
        # 1. Resume Capability
        if i < current_index:
            continue

        chamber_name = group.get("chamber")
        weight = group.get("weight")
        queries = group.get("queries")
        
        if not queries:
            print(f"Skipping empty group: {chamber_name}")
            continue
            
        # 2. Setup JSONL Path for this Chamber
        safe_name = sanitize_filename(chamber_name)
        jsonl_path = os.path.join(OUTPUT_DIR, f"{safe_name}.jsonl")
        
        # 3. Dynamic Limit Calculation
        limit_per_query = max(1, int(weight / len(queries)))
        
        print(f"\n🔹 [{i+1}/{len(chamber_data)}] Group: {chamber_name}")
        print(f"   Target: {jsonl_path}")
        print(f"   Limit/Query: {limit_per_query}")
        
        group_saved_count = 0
        
        for q_idx, query in enumerate(queries):
            # Unique Key for this specific query run
            query_key = f"{chamber_name}:{query}"
            
            if query_key in processed_queries_set:
                 continue

            print(f"   [{q_idx+1}/{len(queries)}] Query: '{query}'...", end="\r")
            
            # 4. Filter Logic
            filters = [
                f"court = '{chamber_name}'",
                f"high_court = '{HIGH_COURT_FILTER}'"
            ]
            
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            
            hits = fetch_from_meili(query, filters, limit_per_query)
            
            # Even if 0 hits, mark as processed to avoid retry loops
            if not hits:
                processed_queries_set.add(query_key)
                continue
                
            for hit in hits:
                doc_id = str(hit.get("id"))
                
                # Check Global Deduplication
                if doc_id in seen_ids_set:
                    continue
                
                # 5. Corruption Check
                text_content = hit.get("text") or ""
                # Check for known error signatures in the document text
                is_corrupted = "ADALET_RUNTIME_EXCEPTION" in text_content or \
                               "FMTY\":\"ERROR" in text_content or \
                               "FMC\":\"ADALET_RUNTIME_EXCEPTION" in text_content

                if is_corrupted:
                    corrupted_files.append({
                        "id": doc_id,
                        "filename": hit.get("filename"),
                        "chamber": chamber_name,
                        "query": query,
                        "fetched_at": datetime.now().isoformat(),
                        "error_snippet": text_content[:200]
                    })
                    total_corrupted += 1
                    seen_ids_set.add(doc_id) 
                    continue

                # 6. Save Record (JSONL)
                record = hit.copy()
                
                # Keep metadata inside the record for traceability
                record["meta"] = {
                    "court": chamber_name,
                    "query": query,
                    "fetched_at": datetime.now().isoformat(),
                    "original_filename": hit.get("filename", "")
                }
                
                append_jsonl(jsonl_path, record)
                
                # 7. Update RAM State
                downloaded_files[doc_id] = {
                    "f": hit.get("filename"), # Short keys to save RAM if needed
                    "c": chamber_name,
                    "q": query,
                    "t": datetime.now().isoformat()
                }
                seen_ids_set.add(doc_id)
                total_new_saved += 1
                group_saved_count += 1
            
            # Mark query as done
            processed_queries_set.add(query_key)
            
            # Regular Checkpoint (Every 10 items)
            if total_new_saved % 10 == 0:
                 state["downloaded_files"] = downloaded_files
                 state["processed_queries"] = list(processed_queries_set)
                 save_json(STATE_FILE, state)
                 save_json(CORRUPTED_FILE, corrupted_files)

        # Force save at end of group
        state["downloaded_files"] = downloaded_files
        state["processed_queries"] = list(processed_queries_set)
        save_json(STATE_FILE, state)
        save_json(CORRUPTED_FILE, corrupted_files)
        
        # --- INTERACTIVE PAUSE ---
        print(f"\n\n✅ Group Completed: {chamber_name}")
        print(f"   - Sent Queries: {len(queries)}")
        print(f"   - New Saved in this group: {group_saved_count}")
        print(f"   - Total New Saved: {total_new_saved}")
        
        choice = input(f"\n>>> Move to next group ({i+2})? (y/n): ").strip().lower()
        if choice == 'y':
            state["current_index"] = i + 1
            save_json(STATE_FILE, state)
        else:
            print("\nFetching stopped by user.")
            break
    
    print(f"\nFetching Process Completed.")
    print(f"Total New Saved: {total_new_saved}")
    print(f"Total Corrupted Files: {total_corrupted}")

if __name__ == "__main__":
    main()