import ahocorasick
import torch
import random
import numpy as np
from utils import normalize_text_for_search
from collections import defaultdict
from config import Config

class SmartLegalMasker:
    def __init__(self, tokenizer, terms_file_path):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        
        # Config'den değerleri al
        self.MIN_MASK_PROB = getattr(Config, 'MIN_MASK_PROB', 0.12)
        self.MAX_MASK_PROB = getattr(Config, 'MAX_MASK_PROB', 0.28)
        self.MIN_LEGAL_RATIO = getattr(Config, 'MIN_LEGAL_RATIO', 0.35)
        self.MAX_LEGAL_RATIO = getattr(Config, 'MAX_LEGAL_RATIO', 0.80)
        
        # Yeni config değerleri
        self.PHRASE_PRIORITY = getattr(Config, 'PHRASE_PRIORITY', 5.0)
        self.LEGAL_MASK_AGGRESSIVE = getattr(Config, 'LEGAL_MASK_AGGRESSIVE', 0.90)
        self.RANDOM_MASK_STANDARD = getattr(Config, 'RANDOM_MASK_STANDARD', 0.80)
        
        # Hızlı stopword check için token ID seti
        self.protected_ids = self._build_protected_set()
        
        # Automaton ve phrase listesi
        self.automaton = self._build_automaton(terms_file_path)
        self.phrase_terms = self._load_phrases(terms_file_path)

    def _build_protected_set(self):
        """Token ID'lere çevir - hızlı lookup"""
        protected_words = {
            "ve", "veya", "ile", "bir", "bu", "şu", "o", "ki", 
            "da", "de", "ama", "fakat", "için", "gibi", "çünkü",
            ".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", 
            "-", "/", "\\", "'", "\"", "`", "’", "“", "”", "…", "«", "»"
        }
        protected_ids = set()
        for word in protected_words:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(ids) == 1:
                protected_ids.add(ids[0])
        return protected_ids

    def _load_phrases(self, path):
        """Çok kelimeli terimleri yükle"""
        phrases = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    term = line.strip()
                    if len(term.split()) > 1:
                        phrases.append(normalize_text_for_search(term))
        except Exception as e:
            print(f"UYARI: Phrase dosyası okunamadı: {e}")
        phrases.sort(key=len, reverse=True)
        print(f"-> {len(phrases)} çok kelimeli terim yüklendi")
        return phrases

    def _build_automaton(self, path):
        A = ahocorasick.Automaton()
        term_count = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    term = line.strip()
                    if len(term) < 2:
                        continue
                    
                    norm = normalize_text_for_search(term)
                    
                    # Priority belirle
                    priority = 1.0
                    word_count = len(term.split())
                    
                    if word_count > 1:
                        priority = 1.5
                    if any(k in norm for k in ["kanun", "madde", "fıkra", "bend"]):
                        priority = 2.0
                    elif any(k in norm for k in ["ceza", "suç", "tazminat", "hapis"]):
                        priority = 1.8
                    
                    A.add_word(norm, (idx, norm, priority))
                    term_count += 1
            A.make_automaton()
            print(f"-> {term_count} terim automaton'a yüklendi")
        except Exception as e:
            print(f"KRİTİK HATA: Automaton oluşturulamadı: {e}")
        return A

    def get_legal_word_map(self, text, offsets, word_ids):
        """
        Returns: (legal_word_map, phrase_spans)
        """
        legal_map = {}
        phrase_word_groups = []
        
        norm_text = normalize_text_for_search(text)
        if len(norm_text) != len(text):
            return {}, []

        # Char index -> word_id mapping
        char_to_word = {}
        for i, (wid, (start, end)) in enumerate(zip(word_ids, offsets)):
            if wid >= 0 and start != end:
                for c in range(start, end):
                    char_to_word[c] = wid

        # 1. Tek kelimeliler (Aho-Corasick)
        for end_idx, (_, norm_term, priority) in self.automaton.iter(norm_text):
            start_idx = end_idx - len(norm_term) + 1
            mid = (start_idx + end_idx + 1) // 2
            
            if mid in char_to_word:
                wid = char_to_word[mid]
                current = legal_map.get(wid, 0)
                legal_map[wid] = max(current, priority)

        # 2. Phrase'leri işaretle - TÜM KELİMELERİ aynı grup olarak işaretle
        for phrase in self.phrase_terms:
            start = 0
            while True:
                idx = norm_text.find(phrase, start)
                if idx == -1:
                    break
                end = idx + len(phrase)
                
                covered_wids = set()
                for c in range(idx, end):
                    if c in char_to_word:
                        covered_wids.add(char_to_word[c])
                
                if len(covered_wids) >= 2:
                    min_wid, max_wid = min(covered_wids), max(covered_wids)
                    
                    # Phrase içindeki TÜM kelimelere PHRASE_PRIORITY
                    for wid in range(min_wid, max_wid + 1):
                        legal_map[wid] = self.PHRASE_PRIORITY
                        
                    phrase_word_groups.append((min_wid, max_wid, phrase))
                
                start = idx + 1

        return legal_map, phrase_word_groups

    def get_dynamic_config(self, legal_density, num_legal_words):
        """Metin yoğunluğuna göre dinamik oranlar"""
        if legal_density > 0.30:
            mask_prob = 0.25
            legal_ratio = 0.75
        elif legal_density > 0.15:
            mask_prob = 0.20
            legal_ratio = 0.65
        elif legal_density > 0.05:
            mask_prob = 0.18
            legal_ratio = 0.55
        else:
            mask_prob = 0.12
            legal_ratio = 0.35
        
        # Sınırları zorla
        mask_prob = max(self.MIN_MASK_PROB, min(self.MAX_MASK_PROB, mask_prob))
        legal_ratio = max(self.MIN_LEGAL_RATIO, min(self.MAX_LEGAL_RATIO, legal_ratio))
        
        return mask_prob, legal_ratio

    def mask_segment_optimized(self, input_ids, word_ids, legal_map, phrase_spans):
        """
        Ana maskeleme fonksiyonu
        """
        seq_len = len(input_ids)
        labels = [-100] * seq_len
        input_ids_tensor = torch.tensor(input_ids).clone()

        # Word ID -> Token indices
        word_to_tokens = defaultdict(list)
        for idx, wid in enumerate(word_ids):
            if wid >= 0:
                word_to_tokens[wid].append(idx)

        if not word_to_tokens:
            return input_ids_tensor.tolist(), labels

        unique_wids = list(word_to_tokens.keys())
        legal_wids = [w for w in unique_wids if w in legal_map]
        legal_density = len(legal_wids) / len(unique_wids) if unique_wids else 0
        
        mask_prob, legal_ratio = self.get_dynamic_config(legal_density, len(legal_wids))

        # Phrase word'lerini işaretle
        phrase_words = set()
        for start_wid, end_wid, _ in phrase_spans:
            phrase_words.update(range(start_wid, end_wid + 1))

        # Grupları hazırla
        legal_groups = []      # (is_phrase, priority, tokens)
        random_groups = []     # (tokens)
        
        for wid in unique_wids:
            tokens = word_to_tokens[wid]
            
            # Stopword check
            if any(input_ids[t] in self.protected_ids for t in tokens):
                continue
            
            if wid in legal_map:
                priority = legal_map[wid]
                is_phrase = wid in phrase_words
                legal_groups.append((is_phrase, priority, tokens))
            else:
                random_groups.append(tokens)

        # Bütçe hesapla
        maskable = len(legal_groups) + len(random_groups)
        total_mask = max(1, int(maskable * mask_prob))
        legal_budget = int(total_mask * legal_ratio)

        # SEÇİM: Önce PHRASE'leri, sonra diğer legal'leri
        selected = []
        
        # 1. TÜM phrase'leri seç (priority >= PHRASE_PRIORITY)
        phrase_groups = [g for g in legal_groups if g[0]]  # is_phrase=True
        other_legal = [g for g in legal_groups if not g[0]]
        
        selected.extend([g[2] for g in phrase_groups])
        
        # 2. Kalan bütçeyle diğer legal'leri seç
        remaining_budget = legal_budget - len(phrase_groups)
        if remaining_budget > 0:
            other_legal.sort(key=lambda x: -x[1])  # Priority'ye göre
            selected.extend([g[2] for g in other_legal[:remaining_budget]])
        
        # 3. Eksik varsa random'dan tamamla
        shortfall = total_mask - len(selected)
        if shortfall > 0:
            random.shuffle(random_groups)
            selected.extend(random_groups[:shortfall])

        # Maskeleme
        for group in selected:
            # Bu grup legal/phrase mi?
            group_wids = {word_ids[i] for i in group if i < len(word_ids)}
            is_legal_group = any(wid in legal_map for wid in group_wids)
            is_phrase_group = any(wid in phrase_words for wid in group_wids)
            
            for idx in group:
                labels[idx] = input_ids[idx]
                r = random.random()
                
                if is_legal_group:
                    # Legal/Phrase: AGGRESSIVE
                    if r < self.LEGAL_MASK_AGGRESSIVE:
                        input_ids_tensor[idx] = self.mask_token_id
                    elif r < (self.LEGAL_MASK_AGGRESSIVE + 0.05):
                        input_ids_tensor[idx] = random.randint(0, self.vocab_size - 1)
                else:
                    # Random: STANDARD
                    if r < self.RANDOM_MASK_STANDARD:
                        input_ids_tensor[idx] = self.mask_token_id
                    elif r < (self.RANDOM_MASK_STANDARD + 0.10):
                        input_ids_tensor[idx] = random.randint(0, self.vocab_size - 1)
        
        return input_ids_tensor.tolist(), labels