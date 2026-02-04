import os
import json
import re
import sys
from config import Config

# --- CONFIGURATION ---
INPUT_DIR = os.path.join(Config.ROOT_DIR, "data", "raw", "jsonl")
OUTPUT_DIR = os.path.join(Config.ROOT_DIR, "data", "resources", "decisions")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """
    Cleans the decision text for WWM.
    1. Splits text by "İçtihat Metni" (case-insensitive) and takes the second part.
       If NOT found, keeps the entire text.
    2. Replaces newlines (\n) and tabs (\t) with a single space.
    3. Collapses multiple spaces into one.
    4. Trims leading/trailing whitespace.
    """
    if not text:
        return None
    
    # 1. Extraction Strategy (Case Insensitive)
    # Look for "İçtihat Metni" to strip the header info
    # using regex split for case-insensitivity
    parts = re.split(r"İçtihat Metni", text, maxsplit=1, flags=re.IGNORECASE)
    
    if len(parts) > 1:
        # Found it, take everything AFTER the phrase
        content = parts[1]
    else:
        # Fallback: Phrase not found, use original text
        content = text

    # 2. whitespace Normalization
    # Replace \n and \t with space
    content = content.replace("\n", " ").replace("\t", " ")
    
    # 3. Collapse multiple spaces
    content = re.sub(r'\s+', ' ', content)
    
    # 4. Remove leading/trailing quotes if they exist (common in some JSON exports)
    content = content.strip().strip('"').strip("'").strip()

    return content

def process_file(filename):
    """
    Processes a single JSONL file.
    """
    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = f"p_{filename}" # Prefix 'p_' for processed
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
                    
                    if cleaned_text and len(cleaned_text) > 10: # Basic length check
                        new_record = {
                            "id": doc_id,
                            "text": cleaned_text
                        }
                        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                        total_saved += 1
                    else:
                        total_skipped += 1
                        
                except json.JSONDecodeError:
                    print(f"   ⚠️ Warning: Failed to decode JSON at line {total_read}")
                    continue
                    
    except Exception as e:
        print(f"  Error processing file: {e}")
        return

    print(f" Complete.")
    print(f"      - Read: {total_read}")
    print(f"      - Saved: {total_saved}")
    print(f"      - Skipped: {total_skipped} (Empty text)")

def main():
    print("Data Preprocessor for WWM Started")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Get all .jsonl files
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    
    if not files:
        print("No JSONL files found in input directory.")
        return

    print(f"Found {len(files)} files queue.\n")
    
    for i, filename in enumerate(files):
        process_file(filename)
        
        # Interactive Check
        if i < len(files) - 1:
            while True:
                choice = input(f"\n>>> Continue to next file ({i+2}/{len(files)})? (y/n): ").strip().lower()
                if choice == 'y':
                    break
                elif choice == 'n':
                    print("\nProcessing stopped by user.")
                    sys.exit(0)
    
    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()
