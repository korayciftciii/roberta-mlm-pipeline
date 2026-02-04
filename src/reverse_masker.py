import os
import ujson
import json
from collections import Counter, defaultdict
from datetime import datetime
from transformers import AutoTokenizer
import colorama
from colorama import Fore, Style, Back

colorama.init(autoreset=True)

from config import Config

# ============ KONFIGURASYON ============
PROJECT_ROOT = Config.ROOT_DIR
MODEL_NAME = Config.MODEL_NAME
INPUT_FILE = Config.OUTPUT_FILE
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Renk sabitleri
C_HEADER = Fore.CYAN + Style.BRIGHT
C_SUCCESS = Fore.GREEN + Style.BRIGHT
C_WARNING = Fore.YELLOW + Style.BRIGHT
C_ERROR = Fore.RED + Style.BRIGHT
C_INFO = Fore.BLUE + Style.BRIGHT
C_RESET = Style.RESET_ALL

# ============ YARDIMCI FONKSIYONLAR ============

def format_token(token):
    """SentencePiece tokenını temizle"""
    return token.replace("▁", " ").replace("##", "").strip()

def reconstruct_words(labels, input_ids, tokenizer):
    """
    Maskelenen kelimeleri ve pozisyonlarını çıkar
    Returns: [(word, start_idx, end_idx, is_legal_guess), ...]
    """
    words = []
    current_parts = []
    current_start = None
    
    safe_labels = [l if l != -100 else 0 for l in labels]
    raw_tokens = tokenizer.convert_ids_to_tokens(safe_labels)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    for idx, label_id in enumerate(labels):
        if label_id == -100:
            if current_parts:
                word = "".join(current_parts).replace("▁", " ").strip()
                words.append({
                    'word': word,
                    'start': current_start,
                    'end': idx - 1,
                    'tokens': current_parts.copy(),
                    'is_legal': guess_if_legal(word)
                })
                current_parts = []
                current_start = None
            continue
        
        token_str = raw_tokens[idx]
        
        if token_str.startswith("▁") or token_str.startswith("##") == False:
            # Yeni kelime başlangıcı
            if current_parts:
                word = "".join(current_parts).replace("▁", " ").strip()
                words.append({
                    'word': word,
                    'start': current_start,
                    'end': idx - 1,
                    'tokens': current_parts.copy(),
                    'is_legal': guess_if_legal(word)
                })
            current_start = idx
            current_parts = [token_str]
        else:
            current_parts.append(token_str)
    
    if current_parts:
        word = "".join(current_parts).replace("▁", " ").strip()
        words.append({
            'word': word,
            'start': current_start,
            'end': len(labels) - 1,
            'tokens': current_parts.copy(),
            'is_legal': guess_if_legal(word)
        })
    
    return words

def guess_if_legal(word):
    """Basit heuristic: uzunluk ve içerik bazlı legal tahmini"""
    word_lower = word.lower()
    
    # Legal indicator kelimeler
    legal_indicators = [
        'mahkeme', 'temyiz', 'istinaf', 'yargıtay', 'danıştay',
        'ceza', 'suç', 'hüküm', 'karar', 'kanun', 'madde',
        'tazminat', 'hapis', 'tutuklama', 'soruşturma',
        'davacı', 'davalı', 'sanık', 'maktul', 'şikayet',
        'delil', 'tanık', 'bilirkişi', 'vekalet', 'tensip'
    ]
    
    score = 0
    for indicator in legal_indicators:
        if indicator in word_lower:
            score += 1
    
    # Uzun kelimeler daha muhtemel legal
    if len(word) > 10:
        score += 0.5
    
    return score >= 1

def analyze_masking_strategy(input_ids, labels, tokenizer):
    """
    Maskeleme stratejisini analiz et (80/10/10)
    """
    stats = {
        'mask_token': 0,      # [MASK]
        'random_token': 0,    # Rastgele
        'original_keep': 0,   # Değişmemiş
        'unmasked': 0         # -100
    }
    
    mask_id = tokenizer.mask_token_id
    
    for inp, lbl in zip(input_ids, labels):
        if lbl == -100:
            stats['unmasked'] += 1
        elif inp == mask_id:
            stats['mask_token'] += 1
        elif inp == lbl:
            stats['original_keep'] += 1
        else:
            stats['random_token'] += 1
    
    total_masked = stats['mask_token'] + stats['random_token'] + stats['original_keep']
    
    if total_masked > 0:
        stats['mask_pct'] = stats['mask_token'] / total_masked * 100
        stats['random_pct'] = stats['random_token'] / total_masked * 100
        stats['keep_pct'] = stats['original_keep'] / total_masked * 100
    
    return stats

def visualize_sample(input_ids, labels, tokenizer, max_width=80):
    """
    Renkli görselleştirme
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    mask_id = tokenizer.mask_token_id
    
    lines = []
    current_line = ""
    current_colors = []
    
    for inp, lbl, tok in zip(input_ids, labels, tokens):
        # Tokenı temizle
        display_tok = format_token(tok)
        if not display_tok:
            display_tok = "[UNK]"
        
        # Renk belirle
        if lbl == -100:
            color = Fore.WHITE      # Maskeleme yok
        elif inp == mask_id:
            color = Fore.RED + Style.BRIGHT    # [MASK]
        elif inp == lbl:
            color = Fore.GREEN      # Orijinal korundu (10%)
        else:
            color = Fore.YELLOW     # Rastgele token (10%)
        
        # Satır yönetimi
        if len(current_line) + len(display_tok) + 1 > max_width:
            lines.append((current_line, current_colors))
            current_line = display_tok
            current_colors = [(len(display_tok), color)]
        else:
            if current_line:
                current_line += " "
            current_line += display_tok
            current_colors.append((len(display_tok), color))
    
    if current_line:
        lines.append((current_line, current_colors))
    
    return lines

# ============ ANA ANALIZ FONKSIYONLARI ============

class MaskingAnalyzer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.global_stats = {
            'total_samples': 0,
            'total_tokens': 0,
            'total_masked': 0,
            'legal_guessed': 0,
            'random_guessed': 0,
            'word_lengths': [],
            'masking_types': Counter(),
            'top_masked_words': Counter(),
            'legal_ratio_distribution': [],
            'sample_lengths': []
        }
        self.samples = []
    
    def process_sample(self, input_ids, labels, sample_idx):
        """Tek örneği analiz et"""
        
        # Temel istatistikler
        stats = analyze_masking_strategy(input_ids, labels, self.tokenizer)
        masked_words = reconstruct_words(labels, input_ids, self.tokenizer)
        
        # Legal tahmini
        legal_words = [w for w in masked_words if w['is_legal']]
        random_words = [w for w in masked_words if not w['is_legal']]
        
        # Global stats güncelle
        self.global_stats['total_samples'] += 1
        self.global_stats['total_tokens'] += len([l for l in labels if l != -100])
        self.global_stats['total_masked'] += len(masked_words)
        self.global_stats['legal_guessed'] += len(legal_words)
        self.global_stats['random_guessed'] += len(random_words)
        self.global_stats['word_lengths'].extend([len(w['word']) for w in masked_words])
        self.global_stats['legal_ratio_distribution'].append(
            len(legal_words) / len(masked_words) if masked_words else 0
        )
        self.global_stats['sample_lengths'].append(len(input_ids))
        
        # En çok maskelenen kelimeler
        for w in masked_words:
            self.global_stats['top_masked_words'][w['word']] += 1
        
        # Maskeleme tipleri
        self.global_stats['masking_types'].update({
            'mask_token': stats.get('mask_token', 0),
            'random_token': stats.get('random_token', 0),
            'original_keep': stats.get('original_keep', 0)
        })
        
        # Sample kaydet
        sample_data = {
            'idx': sample_idx,
            'stats': stats,
            'masked_words': masked_words,
            'legal_words': legal_words,
            'random_words': random_words,
            'visualization': visualize_sample(input_ids, labels, self.tokenizer)
        }
        self.samples.append(sample_data)
        
        return sample_data
    
    def print_sample(self, sample, detailed=False):
        """Örneği görsel olarak yazdır"""
        print(f"\n{C_HEADER}{'='*60}{C_RESET}")
        print(f"{C_HEADER}ÖRNEK #{sample['idx'] + 1}{C_RESET}")
        print(f"{C_HEADER}{'='*60}{C_RESET}")
        
        # Görselleştirme
        print(f"\n{C_INFO}Görselleştirme (Renkler):{C_RESET}")
        print(f"  {Fore.RED}Kırmızı{Fore.WHITE} = [MASK] (80%)")
        print(f"  {Fore.YELLOW}Sarı{Fore.WHITE} = Rastgele Token (10%)")
        print(f"  {Fore.GREEN}Yeşil{Fore.WHITE} = Orijinal Korundu (10%)")
        print(f"  {Fore.WHITE}Beyaz{Fore.WHITE} = Maskeleme Yok")
        print()
        
        for line, colors in sample['visualization']:
            colored_line = ""
            pos = 0
            for length, color in colors:
                colored_line += color + line[pos:pos+length]
                pos += length
            print(colored_line + C_RESET)
        
        # İstatistikler
        print(f"\n{C_INFO}Maskeleme İstatistiği:{C_RESET}")
        s = sample['stats']
        print(f"  [MASK] kullanımı: {s.get('mask_pct', 0):.1f}%")
        print(f"  Rastgele token:   {s.get('random_pct', 0):.1f}%")
        print(f"  Orijinal korundu: {s.get('keep_pct', 0):.1f}%")
        
        # Maskelenen kelimeler
        print(f"\n{C_SUCCESS}Maskelenen Kelimeler ({len(sample['masked_words'])} adet):{C_RESET}")
        
        if sample['legal_words']:
            print(f"  {C_INFO}Legal Tahmini ({len(sample['legal_words'])}):{C_RESET}")
            for w in sample['legal_words'][:10]:
                print(f"    • {w['word']:<20} (token: {w['start']}-{w['end']})")
            if len(sample['legal_words']) > 10:
                print(f"    ... ve {len(sample['legal_words'])-10} daha")
        
        if sample['random_words']:
            print(f"  {C_WARNING}Genel Tahmini ({len(sample['random_words'])}):{C_RESET}")
            for w in sample['random_words'][:5]:
                print(f"    • {w['word']}")
    
    def print_global_report(self):
        """Global rapor"""
        print(f"\n{C_HEADER}{'='*70}{C_RESET}")
        print(f"{C_HEADER}GLOBAL ANALIZ RAPORU{C_RESET}")
        print(f"{C_HEADER}{'='*70}{C_RESET}")
        
        g = self.global_stats
        
        print(f"\n{C_INFO}Temel İstatistikler:{C_RESET}")
        print(f"  Toplam örnek:      {g['total_samples']:,}")
        print(f"  Toplam token:      {g['total_tokens']:,}")
        print(f"  Toplam maskelenen: {g['total_masked']:,}")
        print(f"  Ortalama örnek uzunluğu: {sum(g['sample_lengths'])/len(g['sample_lengths']):.1f} token")
        
        print(f"\n{C_INFO}Legal vs Random Dağılımı:{C_RESET}")
        total = g['legal_guessed'] + g['random_guessed']
        if total > 0:
            legal_pct = g['legal_guessed'] / total * 100
            print(f"  Legal tahmini:  {g['legal_guessed']:,} ({legal_pct:.1f}%)")
            print(f"  Random tahmini: {g['random_guessed']:,} ({100-legal_pct:.1f}%)")
        
        print(f"\n{C_INFO}Maskeleme Stratejisi:{C_RESET}")
        mt = g['masking_types']
        total_types = sum(mt.values())
        if total_types > 0:
            print(f"  [MASK] token:     {mt['mask_token']:,} ({mt['mask_token']/total_types*100:.1f}%)")
            print(f"  Rastgele token:   {mt['random_token']:,} ({mt['random_token']/total_types*100:.1f}%)")
            print(f"  Orijinal korundu: {mt['original_keep']:,} ({mt['original_keep']/total_types*100:.1f}%)")
        
        print(f"\n{C_INFO}Kelime Uzunluk İstatistiği:{C_RESET}")
        if g['word_lengths']:
            import statistics
            print(f"  Ortalama: {statistics.mean(g['word_lengths']):.1f} karakter")
            print(f"  Median:   {statistics.median(g['word_lengths']):.1f}")
            print(f"  En uzun:  {max(g['word_lengths'])}")
        
        print(f"\n{C_INFO}En Çok Maskelenen Kelimeler (Top 20):{C_RESET}")
        for word, count in g['top_masked_words'].most_common(20):
            print(f"  {word:<25} {count:>5} kez")
        
        # Legal ratio dağılımı
        if g['legal_ratio_distribution']:
            print(f"\n{C_INFO}Legal Ratio Dağılımı (Örnek başına):{C_RESET}")
            avg_ratio = sum(g['legal_ratio_distribution']) / len(g['legal_ratio_distribution'])
            print(f"  Ortalama: {avg_ratio*100:.1f}%")
            high_legal = sum(1 for r in g['legal_ratio_distribution'] if r > 0.6)
            print(f"  %60+ legal: {high_legal} örnek ({high_legal/len(g['legal_ratio_distribution'])*100:.1f}%)")
    
    def save_json_report(self, filename="masking_analysis.json"):
        """JSON raporu kaydet"""
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': MODEL_NAME,
            'global_stats': {
                k: (dict(v) if isinstance(v, Counter) else v)
                for k, v in self.global_stats.items()
            },
            'sample_summaries': [
                {
                    'idx': s['idx'],
                    'masked_count': len(s['masked_words']),
                    'legal_count': len(s['legal_words']),
                    'random_count': len(s['random_words'])
                }
                for s in self.samples
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{C_SUCCESS}JSON raporu kaydedildi: {filepath}{C_RESET}")

# ============ MAIN ============

def main():
    print(f"{C_HEADER}Enhanced Legal Masking Analyzer{C_RESET}")
    print(f"Model: {MODEL_NAME}")
    print(f"Input: {INPUT_FILE}")
    print("-" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    analyzer = MaskingAnalyzer(tokenizer)
    
    # Dosyayı oku ve analiz et
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i in range(100):  # İlk 100 örnek
            line = f.readline()
            if not line:
                break
            
            data = ujson.loads(line)
            sample = analyzer.process_sample(
                data["input_ids"], 
                data["labels"], 
                i
            )
            
            # Detaylı göster (ilk 10)
            if i < 10:
                analyzer.print_sample(sample, detailed=True)
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"{C_INFO}İşlenen: {i+1} örnek...{C_RESET}")
    
    # Global rapor
    analyzer.print_global_report()
    
    # JSON kaydet
    analyzer.save_json_report()
    
    print(f"\n{C_SUCCESS}Analiz tamamlandı!{C_RESET}")

if __name__ == "__main__":
    main()