import ahocorasick
import torch
import random
import numpy as np
from utils import normalize_text_for_search

class LegalMasker:
    def __init__(self, tokenizer, terms_file_path, mask_prob=0.20, legal_ratio=0.60):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.legal_ratio = legal_ratio
        
        # Token ID'leri
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        
        # Aho-Corasick Kurulumu
        print(f"-> Otomat hazırlanıyor: {terms_file_path}")
        self.automaton = self._build_automaton(terms_file_path)

    def _build_automaton(self, path):
        A = ahocorasick.Automaton()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    term = line.strip()
                    if len(term) < 2: continue # Çok kısa terimleri atla
                    norm_key = normalize_text_for_search(term)
                    A.add_word(norm_key, (idx, term))
            A.make_automaton()
        except Exception as e:
            print(f"KRİTİK HATA: Sözlük okunamadı! {e}")
            return ahocorasick.Automaton()
        return A

    def get_legal_bitmap(self, text, offsets):
        """
        Metindeki her bir tokenin legal olup olmadığını belirler.
        Tokens listesi yerine sadece offsets kullanır.
        """
        is_legal_token = [False] * len(offsets)
        norm_text = normalize_text_for_search(text)
        
        # 1. Aho-Corasick ile karakter aralıklarını bul
        found_spans = []
        for end_idx, (original_idx, original_term) in self.automaton.iter(norm_text):
            # Aho-corasick end index'i inclusive verir, length çıkar
            # normalize edilmiş metin üzerinden uzunluk alıyoruz
            # Not: Normalize metin uzunluğu orijinalle aynı olmalı (utils.py'deki mantığa göre)
            start_idx = end_idx - len(normalize_text_for_search(original_term)) + 1
            found_spans.append((start_idx, end_idx + 1))
            
        # 2. Tokenları Spans ile Eşleştir
        # Eğer tokenin karakter aralığı, legal terim aralığı ile kesişiyorsa işaretle
        for i, (start, end) in enumerate(offsets):
            if start == end: continue # Special tokens veya boşluklar
            
            # Basit optimizasyon: Tokenin orta noktası terimin içinde mi?
            token_mid = (start + end) / 2
            
            for sp_start, sp_end in found_spans:
                if sp_start <= token_mid < sp_end:
                    is_legal_token[i] = True
                    break
                    
        return is_legal_token

    def mask_segment(self, input_ids, word_ids, is_legal_map):
        """
        Word IDs kullanarak Whole Word Masking yapar.
        """
        seq_len = len(input_ids)
        labels = [-100] * seq_len
        input_ids_tensor = torch.tensor(input_ids).clone()

        # 1. Kelimeleri Grupla (Word ID bazlı)
        # word_ids örneği: [None, 0, 0, 1, 2, 2, None]
        word_groups = {}
        
        for idx, wid in enumerate(word_ids):
            if wid is None: continue # CLS, SEP, PAD atla
            
            if wid not in word_groups:
                word_groups[wid] = []
            word_groups[wid].append(idx)
            
        all_groups = list(word_groups.values())
        
        # Eğer hiç kelime grubu yoksa (boş chunk), direkt dön
        if not all_groups:
            return input_ids_tensor.tolist(), labels

        legal_word_groups = []
        random_word_groups = []

        # 2. Grupları Sınıflandır (Legal vs Random)
        for group in all_groups:
            # Grup içinde bir tane bile legal token varsa o kelimeyi Legal say
            # is_legal_map boyutu chunk içinde kesilmiş olabilir, index kontrolü yap
            is_group_legal = False
            for idx in group:
                if idx < len(is_legal_map) and is_legal_map[idx]:
                    is_group_legal = True
                    break
            
            if is_group_legal:
                legal_word_groups.append(group)
            else:
                random_word_groups.append(group)

        # 3. Maskeleme Bütçesi
        num_maskable_words = len(all_groups)
        total_tokens_to_mask = max(1, int(num_maskable_words * self.mask_prob))
        
        legal_budget = int(total_tokens_to_mask * self.legal_ratio)
        random_budget = total_tokens_to_mask - legal_budget
        
        selected_groups = []
        
        # A) Legal Maskeleme
        random.shuffle(legal_word_groups)
        count = 0
        for group in legal_word_groups:
            if count >= legal_budget: break
            selected_groups.append(group)
            count += 1
            
        # Legal yetmediyse random'a devret
        if count < legal_budget:
            random_budget += (legal_budget - count)

        # B) Random Maskeleme
        random.shuffle(random_word_groups)
        count = 0
        for group in random_word_groups:
            if count >= random_budget: break
            selected_groups.append(group)
            count += 1
            
        # Hiçbir şey seçilmediyse en az 1 tane random seç (Data boş gitmesin)
        if not selected_groups and random_word_groups:
            selected_groups.append(random_word_groups[0])

        # 4. Maskeleme İşlemi (BERT Style: 80% MASK, 10% Random, 10% Original)
        for group in selected_groups:
            for idx in group:
                original_token = input_ids[idx]
                labels[idx] = original_token # Label'a gerçeği yaz
                
                prob = random.random()
                if prob < 0.8:
                    input_ids_tensor[idx] = self.mask_token_id
                elif prob < 0.9:
                    input_ids_tensor[idx] = random.randint(0, self.vocab_size - 1)
                else:
                    pass # %10 ihtimalle token değişmez ama loss hesaplanır

        return input_ids_tensor.tolist(), labels