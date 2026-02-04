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
    Returns: list of dict with word info
    """
    words = []
    current_parts = []
    current_start = None
    
    safe_labels = [l if l != -100 else 0 for l in labels]
    raw_tokens = tokenizer.convert_ids_to_tokens(safe_labels)
    
    for idx, label_id in enumerate(labels):
        if label_id == -100:
            if current_parts:
                word = "".join(current_parts).replace("▁", " ").strip()
                if word:
                    words.append({
                        'word': word,
                        'start': current_start,
                        'end': idx - 1,
                        'tokens': current_parts.copy()
                    })
                current_parts = []
                current_start = None
            continue
        
        token_str = raw_tokens[idx]
        
        # Yeni kelime başlangıcı (SentencePiece: ▁ ile başlar)
        if token_str.startswith("▁"):
            if current_parts:
                word = "".join(current_parts).replace("▁", " ").strip()
                if word:
                    words.append({
                        'word': word,
                        'start': current_start,
                        'end': idx - 1,
                        'tokens': current_parts.copy()
                    })
            current_start = idx
            current_parts = [token_str]
        else:
            current_parts.append(token_str)
    
    # Son kelimeyi ekle
    if current_parts:
        word = "".join(current_parts).replace("▁", " ").strip()
        if word:
            words.append({
                'word': word,
                'start': current_start,
                'end': len(labels) - 1,
                'tokens': current_parts.copy()
            })
    
    return words

class LegalTermMatcher:
    """Terms dosyasından yüklenen terimlerle eşleştirme"""
    
    def __init__(self, terms_file):
        self.terms = set()
        self.phrases = []
        self.single_words = set()
        self._load_terms(terms_file)
    
    def _load_terms(self, filepath):
        """Terms dosyasını yükle"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    term = line.strip().lower()
                    if len(term) < 2:
                        continue
                    
                    self.terms.add(term)
                    
                    words = term.split()
                    if len(words) > 1:
                        self.phrases.append({
                            'text': term,
                            'words': words,
                            'length': len(words)
                        })
                    else:
                        self.single_words.add(term)
            
            # Phrase'leri uzunluğa göre sırala (uzun önce)
            self.phrases.sort(key=lambda x: -x['length'])
            print(f"{C_SUCCESS}✓ {len(self.terms)} terim yüklendi ({len(self.phrases)} phrase){C_RESET}")
            
        except Exception as e:
            print(f"{C_ERROR}✗ Terms dosyası okunamadı: {e}{C_RESET}")
    
    def classify_word(self, word):
        """
        Kelimeyi sınıflandır
        Returns: ('phrase', 'single', 'none', 'partial')
        """
        word_lower = word.lower()
        
        # Direkt eşleşme
        if word_lower in self.terms:
            if word_lower in self.single_words:
                return 'single'
            return 'phrase'
        
        # Phrase içinde geçiyor mu?
        for phrase in self.phrases:
            if word_lower in phrase['words']:
                return 'partial'
        
        # İçerme kontrolü (substring)
        for term in self.terms:
            if len(term) > 4 and term in word_lower:
                return 'partial'
        
        return 'none'
    
    def find_phrases_in_sequence(self, words_list):
        """
        Ardışık kelimelerde phrase tespiti
        Returns: list of (start_idx, end_idx, phrase_text)
        """
        found = []
        text_lower = [w['word'].lower() for w in words_list]
        
        for phrase in self.phrases:
            phrase_words = phrase['words']
            
            for i in range(len(text_lower) - len(phrase_words) + 1):
                window = text_lower[i:i + len(phrase_words)]
                
                # Kelime bazlı eşleşme
                if window == phrase_words:
                    found.append({
                        'start': i,
                        'end': i + len(phrase_words) - 1,
                        'phrase': phrase['text'],
                        'words': words_list[i:i + len(phrase_words)]
                    })
        
        return found

def analyze_masking_strategy(input_ids, labels, tokenizer):
    """Maskeleme stratejisini analiz et"""
    stats = {
        'mask_token': 0,
        'random_token': 0,
        'original_keep': 0,
        'unmasked': 0
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

def visualize_sample(input_ids, labels, tokenizer, max_width=100):
    """Renkli görselleştirme"""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    mask_id = tokenizer.mask_token_id
    
    lines = []
    current_line = ""
    current_colors = []
    
    for inp, lbl, tok in zip(input_ids, labels, tokens):
        display_tok = format_token(tok)
        if not display_tok:
            display_tok = "[UNK]"
        
        # Renk belirle
        if lbl == -100:
            color = Fore.WHITE
        elif inp == mask_id:
            color = Fore.RED + Style.BRIGHT
        elif inp == lbl:
            color = Fore.GREEN
        else:
            color = Fore.YELLOW
        
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

# ============ ANA ANALIZ SINIFI ============

class MaskingAnalyzer:
    def __init__(self, tokenizer, terms_file=None):
        self.tokenizer = tokenizer
        self.term_matcher = LegalTermMatcher(terms_file) if terms_file else None
        
        self.global_stats = {
            'total_samples': 0,
            'total_tokens': 0,
            'total_masked_words': 0,
            'classified_as': Counter(),
            'masking_types': Counter(),
            'top_masked_words': Counter(),
            'phrase_detections': [],
            'legal_ratio_distribution': [],
            'sample_lengths': [],
            'word_length_stats': []
        }
        self.samples = []
    
    def process_sample(self, input_ids, labels, sample_idx):
        """Tek örneği analiz et"""
        
        # Temel istatistikler
        stats = analyze_masking_strategy(input_ids, labels, self.tokenizer)
        masked_words = reconstruct_words(labels, input_ids, self.tokenizer)
        
        # Terms ile sınıflandır
        classified_words = []
        if self.term_matcher:
            for w in masked_words:
                classification = self.term_matcher.classify_word(w['word'])
                w['classification'] = classification
                classified_words.append(w)
                self.global_stats['classified_as'][classification] += 1
        else:
            classified_words = masked_words
            for w in classified_words:
                w['classification'] = 'unknown'
        
        # Phrase tespiti
        detected_phrases = []
        if self.term_matcher:
            detected_phrases = self.term_matcher.find_phrases_in_sequence(classified_words)
        
        # Global stats güncelle
        self.global_stats['total_samples'] += 1
        self.global_stats['total_tokens'] += len([l for l in labels if l != -100])
        self.global_stats['total_masked_words'] += len(masked_words)
        self.global_stats['sample_lengths'].append(len(input_ids))
        self.global_stats['word_length_stats'].extend([len(w['word']) for w in masked_words])
        
        # Legal ratio (terms'e göre)
        legal_like = sum(1 for w in classified_words if w['classification'] in ('single', 'phrase', 'partial'))
        ratio = legal_like / len(masked_words) if masked_words else 0
        self.global_stats['legal_ratio_distribution'].append(ratio)
        
        # En çok maskelenen kelimeler
        for w in masked_words:
            self.global_stats['top_masked_words'][w['word']] += 1
        
        # Maskeleme tipleri
        self.global_stats['masking_types'].update({
            'mask_token': stats.get('mask_token', 0),
            'random_token': stats.get('random_token', 0),
            'original_keep': stats.get('original_keep', 0)
        })
        
        # Phrase kaydet
        if detected_phrases:
            self.global_stats['phrase_detections'].append({
                'sample_idx': sample_idx,
                'phrases': detected_phrases
            })
        
        # Sample kaydet
        sample_data = {
            'idx': sample_idx,
            'stats': stats,
            'masked_words': classified_words,
            'detected_phrases': detected_phrases,
            'legal_ratio': ratio,
            'visualization': visualize_sample(input_ids, labels, self.tokenizer)
        }
        self.samples.append(sample_data)
        
        return sample_data
    
    def print_sample(self, sample, detailed=False):
        """Örneği görsel olarak yazdır"""
        print(f"\n{C_HEADER}{'='*70}{C_RESET}")
        print(f"{C_HEADER}ÖRNEK #{sample['idx'] + 1}{C_RESET}")
        print(f"{C_HEADER}{'='*70}{C_RESET}")
        
        # Görselleştirme
        print(f"\n{C_INFO}Görselleştirme (Renkler):{C_RESET}")
        print(f"  {Fore.RED}█{C_RESET} [MASK] ({sample['stats'].get('mask_pct', 0):.0f}%)")
        print(f"  {Fore.YELLOW}█{C_RESET} Rastgele Token ({sample['stats'].get('random_pct', 0):.0f}%)")
        print(f"  {Fore.GREEN}█{C_RESET} Orijinal Korundu ({sample['stats'].get('keep_pct', 0):.0f}%)")
        print(f"  {Fore.WHITE}█{C_RESET} Maskeleme Yok")
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
        print(f"  [MASK]:      {s.get('mask_token', 0):>4} ({s.get('mask_pct', 0):>5.1f}%)")
        print(f"  Rastgele:    {s.get('random_token', 0):>4} ({s.get('random_pct', 0):>5.1f}%)")
        print(f"  Korundu:     {s.get('original_keep', 0):>4} ({s.get('keep_pct', 0):>5.1f}%)")
        print(f"  Legal Ratio: {sample['legal_ratio']*100:>5.1f}%")
        
        # Maskelenen kelimeler
        words = sample['masked_words']
        print(f"\n{C_SUCCESS}Maskelenen Kelimeler ({len(words)} adet):{C_RESET}")
        
        # Sınıflandırmaya göre grupla
        by_class = defaultdict(list)
        for w in words:
            by_class[w['classification']].append(w)
        
        # Phrase'ler (bütün maskelenmiş)
        if by_class['phrase']:
            print(f"\n  {C_SUCCESS}✓ Phrase (Tam Eşleşme):{C_RESET}")
            for w in by_class['phrase'][:5]:
                marker = "🟢" if w['classification'] == 'phrase' else "🟡"
                print(f"    {marker} {w['word']:<25} (tokens: {w['start']}-{w['end']})")
        
        # Single terms
        if by_class['single']:
            print(f"\n  {C_INFO}● Single Terms:{C_RESET}")
            for w in by_class['single'][:5]:
                print(f"    • {w['word']}")
        
        # Partial (phrase parçası)
        if by_class['partial']:
            print(f"\n  {C_WARNING}◐ Partial (Phrase Parçası):{C_RESET}")
            for w in by_class['partial'][:5]:
                print(f"    • {w['word']}")
        
        # Detected phrases (ardışık kelimeler)
        if sample['detected_phrases']:
            print(f"\n  {C_SUCCESS}🔗 Tespit Edilen Phrase'ler:{C_RESET}")
            for p in sample['detected_phrases'][:3]:
                phrase_words = [w['word'] for w in p['words']]
                print(f"    \"{' '.join(phrase_words)}\"")
        
        # Genel/Unknown
        others = by_class.get('none', []) + by_class.get('unknown', [])
        if others:
            print(f"\n  {Fore.WHITE}○ Genel Kelimeler:{C_RESET}")
            for w in others[:3]:
                print(f"    • {w['word']}")
    
    def print_global_report(self):
        """Global rapor"""
        print(f"\n{C_HEADER}{'='*75}{C_RESET}")
        print(f"{C_HEADER}                    GLOBAL ANALIZ RAPORU{C_RESET}")
        print(f"{C_HEADER}{'='*75}{C_RESET}")
        
        g = self.global_stats
        
        # Temel istatistikler
        print(f"\n{C_INFO}📊 Temel İstatistikler:{C_RESET}")
        print(f"  Toplam Örnek:           {g['total_samples']:,}")
        print(f"  Toplam Token:           {g['total_tokens']:,}")
        print(f"  Maskelenen Kelime:      {g['total_masked_words']:,}")
        avg_len = sum(g['sample_lengths']) / len(g['sample_lengths']) if g['sample_lengths'] else 0
        print(f"  Ortalama Uzunluk:       {avg_len:.1f} token")
        
        # Sınıflandırma dağılımı
        print(f"\n{C_INFO}🏛️  Legal Terim Sınıflandırması:{C_RESET}")
        total_classified = sum(g['classified_as'].values())
        if total_classified > 0:
            for cls, count in g['classified_as'].most_common():
                pct = count / total_classified * 100
                icon = {'phrase': '✓', 'single': '●', 'partial': '◐', 'none': '○', 'unknown': '?'}.get(cls, '?')
                color = {'phrase': C_SUCCESS, 'single': C_INFO, 'partial': C_WARNING, 'none': Fore.WHITE, 'unknown': C_ERROR}.get(cls, C_RESET)
                print(f"  {color}{icon} {cls:<12}{C_RESET}: {count:>5} ({pct:>5.1f}%)")
        
        # Legal ratio dağılımı
        if g['legal_ratio_distribution']:
            print(f"\n{C_INFO}📈 Legal Ratio Dağılımı:{C_RESET}")
            avg_ratio = sum(g['legal_ratio_distribution']) / len(g['legal_ratio_distribution'])
            print(f"  Ortalama:               {avg_ratio*100:.1f}%")
            
            bins = [(0, 0.2, "Çok Düşük"), (0.2, 0.4, "Düşük"), 
                   (0.4, 0.6, "Orta"), (0.6, 0.8, "Yüksek"), (0.8, 1.0, "Çok Yüksek")]
            for low, high, label in bins:
                count = sum(1 for r in g['legal_ratio_distribution'] if low <= r < high)
                pct = count / len(g['legal_ratio_distribution']) * 100
                bar = "█" * int(pct / 5)
                print(f"  {label:<12}: {count:>4} ({pct:>5.1f}%) {bar}")
        
        # Maskeleme stratejisi
        print(f"\n{C_INFO}🎭 Maskeleme Stratejisi:{C_RESET}")
        mt = g['masking_types']
        total_types = sum(mt.values())
        if total_types > 0:
            print(f"  {Fore.RED}█ [MASK]{C_RESET}:       {mt['mask_token']:>6,} ({mt['mask_token']/total_types*100:>5.1f}%)")
            print(f"  {Fore.YELLOW}█ Random{C_RESET}:     {mt['random_token']:>6,} ({mt['random_token']/total_types*100:>5.1f}%)")
            print(f"  {Fore.GREEN}█ Keep{C_RESET}:       {mt['original_keep']:>6,} ({mt['original_keep']/total_types*100:>5.1f}%)")
        
        # Phrase tespiti
        total_phrases = len(g['phrase_detections'])
        print(f"\n{C_INFO}🔗 Phrase Tespiti:{C_RESET}")
        print(f"  Phrase içeren örnek:    {total_phrases}")
        if total_phrases > 0:
            print(f"  Oran:                   {total_phrases/g['total_samples']*100:.1f}%")
        
        # En çok maskelenen kelimeler
        print(f"\n{C_INFO}🏆 En Çok Maskelenen Kelimeler (Top 15):{C_RESET}")
        for word, count in g['top_masked_words'].most_common(15):
            # Sınıflandırma göre renk
            cls = 'unknown'
            if self.term_matcher:
                cls = self.term_matcher.classify_word(word)
            color = {'phrase': C_SUCCESS, 'single': C_INFO, 'partial': C_WARNING, 'none': Fore.WHITE}.get(cls, C_RESET)
            print(f"  {color}{word:<30}{C_RESET} {count:>5} kez")
        
        # Kelime uzunluğu
        if g['word_length_stats']:
            import statistics
            print(f"\n{C_INFO}📏 Kelime Uzunluk İstatistiği:{C_RESET}")
            print(f"  Ortalama:               {statistics.mean(g['word_length_stats']):.1f}")
            print(f"  Median:                 {statistics.median(g['word_length_stats']):.1f}")
            print(f"  En Uzun:                {max(g['word_length_stats'])}")
    
    def save_json_report(self, filename="masking_analysis.json"):
        """JSON raporu kaydet"""
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': MODEL_NAME,
            'config': {
                'phrase_priority': getattr(Config, 'PHRASE_PRIORITY', 5.0),
                'legal_mask_aggressive': getattr(Config, 'LEGAL_MASK_AGGRESSIVE', 0.90),
                'base_legal_ratio': getattr(Config, 'BASE_LEGAL_RATIO', 0.60)
            },
            'global_stats': {
                k: (dict(v) if isinstance(v, Counter) else v)
                for k, v in self.global_stats.items()
            },
            'sample_summaries': [
                {
                    'idx': s['idx'],
                    'legal_ratio': s['legal_ratio'],
                    'masked_count': len(s['masked_words']),
                    'phrase_count': len(s['detected_phrases'])
                }
                for s in self.samples
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{C_SUCCESS}💾 JSON raporu kaydedildi:{C_RESET}")
        print(f"   {filepath}")

# ============ MAIN ============

def main():
    print(f"{C_HEADER}{'='*75}{C_RESET}")
    print(f"{C_HEADER}          ENHANCED LEGAL MASKING ANALYZER v2.0{C_RESET}")
    print(f"{C_HEADER}{'='*75}{C_RESET}")
    print(f"Model:    {MODEL_NAME}")
    print(f"Input:    {INPUT_FILE}")
    print(f"Terms:    {Config.TERMS_FILE}")
    print("-" * 75)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    analyzer = MaskingAnalyzer(tokenizer, terms_file=Config.TERMS_FILE)
    
    # Dosyayı oku ve analiz et
    sample_count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = ujson.loads(line)
                sample = analyzer.process_sample(
                    data["input_ids"], 
                    data["labels"], 
                    sample_count
                )
                
                # Detaylı göster (ilk 5)
                if sample_count < 5:
                    analyzer.print_sample(sample, detailed=True)
                
                sample_count += 1
                
                # Progress
                if sample_count % 100 == 0:
                    print(f"{C_INFO}⏳ İşlenen: {sample_count} örnek...{C_RESET}")
                
                # Limit (opsiyonel)
                if sample_count >= 1000:  # İlk 1000 örnek
                    print(f"{C_WARNING}⚠️  Limit: İlk 1000 örnek analiz edildi{C_RESET}")
                    break
                    
            except Exception as e:
                print(f"{C_ERROR}✗ Hata (örnek {sample_count}): {e}{C_RESET}")
                continue
    
    # Global rapor
    analyzer.print_global_report()
    
    # JSON kaydet
    analyzer.save_json_report()
    
    print(f"\n{C_HEADER}{'='*75}{C_RESET}")
    print(f"{C_SUCCESS}                    ANALIZ TAMAMLANDI ✓{C_RESET}")
    print(f"{C_HEADER}{'='*75}{C_RESET}")

if __name__ == "__main__":
    main()