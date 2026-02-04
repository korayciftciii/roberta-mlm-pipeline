"""
Biased MLM Data Collator for Turkish Legal Domain
==================================================

This module implements a custom data collator that applies biased masking
toward legal terminology. The algorithm prioritizes masking domain-specific
terms to help the model better learn legal language patterns.

GREEDY MATCHING ALGORITHM EXPLANATION:
======================================
The legal_terms.txt file is sorted from longest to shortest terms.
We use this ordering to implement greedy (longest-first) matching:

1. For each token position, we check if the decoded text starting from
   that position matches any legal term (checking longest terms first).

2. If a match is found, ALL tokens that compose that legal term are
   marked as "legal tokens".

3. Longer matches take precedence. For example:
   - Terms: ["arsa payı karşılığı inşaat sözleşmesi", "inşaat sözleşmesi", "sözleşme"]
   - Text: "...arsa payı karşılığı inşaat sözleşmesi..."
   - Result: The entire phrase is matched, not just "sözleşme"

4. This prevents overlapping matches and ensures domain expressions
   are treated as single semantic units.

MASKING STRATEGY:
=================
- Total mask probability: 15% of tokens
- 70% of masked tokens: Legal terms (biased selection)
- 30% of masked tokens: Random selection

Within masked positions:
- 80%: Replace with [MASK]
- 10%: Replace with random vocabulary token
- 10%: Keep original (but still predict)
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import ahocorasick
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

from utils import normalize_text_for_search


@dataclass
class BiasedMLMDataCollator(DataCollatorMixin):
    """
    Data collator for biased Masked Language Modeling.

    Prioritizes masking legal terms over random tokens while maintaining
    the MLM objective. Uses Aho-Corasick automaton for efficient multi-pattern
    matching and greedy (longest-first) term selection.

    Attributes:
        tokenizer: The tokenizer used for encoding
        terms_file_path: Path to legal terms file (sorted longest to shortest)
        mlm_probability: Total probability of masking a token (default: 0.15)
        legal_bias_ratio: Fraction of masked tokens from legal terms (default: 0.70)
        mask_token_prob: Probability of replacing with [MASK] (default: 0.80)
        random_token_prob: Probability of random replacement (default: 0.10)
        whole_word_masking: Whether to mask entire words (default: True)
        seed: Random seed for reproducibility
    """

    tokenizer: PreTrainedTokenizerBase
    terms_file_path: str
    mlm_probability: float = 0.15
    legal_bias_ratio: float = 0.70
    mask_token_prob: float = 0.80
    random_token_prob: float = 0.10
    whole_word_masking: bool = True
    seed: int = 42
    return_tensors: str = "pt"

    def __post_init__(self):
        """Initialize the Aho-Corasick automaton and random state."""
        self.rng = random.Random(self.seed)
        self._build_automaton()

        # Cache frequently used values
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.special_tokens_ids = set(self.tokenizer.all_special_ids)

    def _build_automaton(self) -> None:
        """
        Build Aho-Corasick automaton for efficient multi-pattern matching.

        The automaton is built from legal terms sorted by length (descending).
        This enables O(n) text scanning where n is text length, regardless
        of the number of patterns.

        The terms file should be pre-sorted with longest terms first for
        greedy matching to work correctly.
        """
        self.automaton = ahocorasick.Automaton()
        self.term_lengths = {}  # Store original term lengths for overlap handling

        try:
            with open(self.terms_file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    term = line.strip()
                    if len(term) < 2:
                        continue

                    # Normalize for matching (lowercase, remove accents)
                    norm_key = normalize_text_for_search(term)
                    if norm_key:
                        self.automaton.add_word(norm_key, (idx, term, len(norm_key)))
                        self.term_lengths[norm_key] = len(norm_key)

            self.automaton.make_automaton()
            print(f"[Collator] Built automaton with {len(self.term_lengths)} legal terms")

        except FileNotFoundError:
            print(f"[WARNING] Terms file not found: {self.terms_file_path}")
            print("[WARNING] Falling back to random-only masking")

    def _find_legal_token_spans(
        self,
        text: str,
        offset_mapping: List[Tuple[int, int]]
    ) -> List[bool]:
        """
        Find which tokens belong to legal terms using greedy matching.

        The algorithm:
        1. Run Aho-Corasick to find all term occurrences (character positions)
        2. Apply greedy selection: longer matches take precedence
        3. Map character spans back to token indices

        Args:
            text: Original text before tokenization
            offset_mapping: List of (start, end) character offsets per token

        Returns:
            Boolean list where True indicates token is part of a legal term
        """
        is_legal = [False] * len(offset_mapping)

        if not hasattr(self, 'automaton') or not self.automaton:
            return is_legal

        # Normalize text for matching
        norm_text = normalize_text_for_search(text)

        # Step 1: Find all matches using Aho-Corasick
        # Returns (end_index, (original_idx, original_term, term_length))
        raw_matches = []
        for end_idx, (orig_idx, orig_term, term_len) in self.automaton.iter(norm_text):
            start_idx = end_idx - term_len + 1
            raw_matches.append((start_idx, end_idx + 1, term_len, orig_term))

        if not raw_matches:
            return is_legal

        # Step 2: Greedy selection - remove overlapping matches
        # Sort by: start position (ascending), then length (descending)
        raw_matches.sort(key=lambda x: (x[0], -x[2]))

        selected_spans = []
        last_end = -1

        for start, end, length, term in raw_matches:
            # Non-overlapping: this span starts after the last selected span ends
            if start >= last_end:
                selected_spans.append((start, end))
                last_end = end

        # Step 3: Map character spans to token indices
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:  # Special token or empty
                continue

            # Check if this token overlaps with any legal term span
            tok_mid = (tok_start + tok_end) / 2
            for span_start, span_end in selected_spans:
                if span_start <= tok_mid < span_end:
                    is_legal[tok_idx] = True
                    break

        return is_legal

    def _get_word_groups(
        self,
        word_ids: List[Optional[int]],
        is_legal: List[bool]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Group token indices by word ID and classify as legal or random.

        For Whole Word Masking, all subword tokens of a word must be
        masked together. A word is considered "legal" if ANY of its
        subword tokens is marked as legal.

        Args:
            word_ids: Word ID for each token (None for special tokens)
            is_legal: Boolean mask indicating legal tokens

        Returns:
            Tuple of (legal_word_groups, random_word_groups)
            Each group is a list of token indices belonging to that word
        """
        word_to_tokens: Dict[int, List[int]] = {}

        for idx, wid in enumerate(word_ids):
            if wid is None:  # Skip special tokens
                continue
            if wid not in word_to_tokens:
                word_to_tokens[wid] = []
            word_to_tokens[wid].append(idx)

        legal_groups = []
        random_groups = []

        for wid, token_indices in word_to_tokens.items():
            # Word is legal if any subword token is legal
            is_legal_word = any(is_legal[idx] for idx in token_indices if idx < len(is_legal))

            if is_legal_word:
                legal_groups.append(token_indices)
            else:
                random_groups.append(token_indices)

        return legal_groups, random_groups

    def _apply_masking(
        self,
        input_ids: List[int],
        word_ids: List[Optional[int]],
        is_legal: List[bool],
        attention_mask: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Apply biased MLM masking to input sequence.

        Masking budget allocation:
        1. Calculate total tokens to mask: num_maskable * mlm_probability
        2. Allocate legal_bias_ratio (70%) to legal terms
        3. Allocate remaining (30%) to random tokens
        4. If insufficient legal tokens, shift budget to random

        Args:
            input_ids: Token IDs
            word_ids: Word ID mapping for WWM
            is_legal: Boolean mask for legal tokens
            attention_mask: Attention mask (1 for real, 0 for padding)

        Returns:
            Tuple of (masked_input_ids, labels)
            Labels are -100 for non-masked positions (ignored in loss)
        """
        seq_len = len(input_ids)
        labels = [-100] * seq_len
        masked_ids = input_ids.copy()

        # Get word groups
        if self.whole_word_masking:
            legal_groups, random_groups = self._get_word_groups(word_ids, is_legal)
        else:
            # Treat each token as its own group
            legal_groups = [[i] for i in range(seq_len)
                           if i < len(is_legal) and is_legal[i] and word_ids[i] is not None]
            random_groups = [[i] for i in range(seq_len)
                            if i < len(is_legal) and not is_legal[i] and word_ids[i] is not None]

        all_groups = legal_groups + random_groups
        if not all_groups:
            return masked_ids, labels

        # Calculate masking budget
        num_maskable_words = len(all_groups)
        total_to_mask = max(1, int(num_maskable_words * self.mlm_probability))

        legal_budget = int(total_to_mask * self.legal_bias_ratio)
        random_budget = total_to_mask - legal_budget

        # Select groups to mask
        selected_groups = []

        # A) Select from legal groups
        self.rng.shuffle(legal_groups)
        legal_selected = 0
        for group in legal_groups:
            if legal_selected >= legal_budget:
                break
            selected_groups.append(group)
            legal_selected += 1

        # B) Shift unmet legal budget to random
        if legal_selected < legal_budget:
            random_budget += (legal_budget - legal_selected)

        # C) Select from random groups
        self.rng.shuffle(random_groups)
        random_selected = 0
        for group in random_groups:
            if random_selected >= random_budget:
                break
            selected_groups.append(group)
            random_selected += 1

        # Apply BERT-style masking to selected groups
        for group in selected_groups:
            for idx in group:
                if idx >= seq_len or attention_mask[idx] == 0:
                    continue
                if input_ids[idx] in self.special_tokens_ids:
                    continue

                original_token = input_ids[idx]
                labels[idx] = original_token  # Set label for loss computation

                prob = self.rng.random()
                if prob < self.mask_token_prob:
                    masked_ids[idx] = self.mask_token_id
                elif prob < self.mask_token_prob + self.random_token_prob:
                    # Random token (avoid special tokens)
                    masked_ids[idx] = self.rng.randint(0, self.vocab_size - 1)
                # else: keep original token (10%)

        return masked_ids, labels

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch with biased masking applied.

        This method is called by the DataLoader. It:
        1. Pads sequences to the same length
        2. Applies biased masking to each sequence
        3. Returns tensors ready for training

        Args:
            examples: List of tokenized examples with keys:
                     - input_ids, attention_mask
                     - offset_mapping (optional, for legal term detection)
                     - original_text (optional, for legal term detection)

        Returns:
            Dictionary with input_ids, attention_mask, labels tensors
        """
        # Handle both dict and feature-based inputs
        if isinstance(examples[0], dict):
            batch = {key: [ex[key] for ex in examples] for key in examples[0].keys()}
        else:
            batch = {key: [getattr(ex, key) for ex in examples] for key in examples[0].keys()}

        # Pad sequences
        max_length = max(len(ids) for ids in batch["input_ids"])

        padded_input_ids = []
        padded_attention_masks = []
        all_labels = []

        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i]
            attention_mask = batch.get("attention_mask", [[1] * len(ids) for ids in batch["input_ids"]])[i]

            # Get offset mapping and text for legal term detection
            offset_mapping = batch.get("offset_mapping", [None] * len(batch["input_ids"]))[i]
            original_text = batch.get("original_text", [None] * len(batch["input_ids"]))[i]
            word_ids = batch.get("word_ids", [None] * len(batch["input_ids"]))[i]

            # Determine legal tokens
            if offset_mapping is not None and original_text is not None:
                is_legal = self._find_legal_token_spans(original_text, offset_mapping)
            else:
                is_legal = [False] * len(input_ids)

            # Generate word_ids if not provided
            if word_ids is None:
                # Simple heuristic: each position is its own word
                word_ids = list(range(len(input_ids)))
                # Mark special tokens (first and last) as None
                if len(word_ids) > 0:
                    word_ids[0] = None
                if len(word_ids) > 1:
                    word_ids[-1] = None

            # Apply masking
            masked_ids, labels = self._apply_masking(
                input_ids, word_ids, is_legal, attention_mask
            )

            # Pad to max_length
            pad_length = max_length - len(input_ids)
            padded_input_ids.append(masked_ids + [self.pad_token_id] * pad_length)
            padded_attention_masks.append(attention_mask + [0] * pad_length)
            all_labels.append(labels + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Make the collator callable."""
        return self.torch_call(examples)


class SimpleMLMCollator(DataCollatorMixin):
    """
    Simplified MLM collator for pre-masked data.

    Use this when input data already contains masked input_ids and labels
    (e.g., from preprocessed JSONL files).
    """

    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate pre-masked examples."""
        batch = {key: [ex[key] for ex in examples] for key in examples[0].keys()}

        max_length = max(len(ids) for ids in batch["input_ids"])

        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i]
            labels = batch.get("labels", [[-100] * len(ids) for ids in batch["input_ids"]])[i]

            pad_length = max_length - len(input_ids)

            padded_input_ids.append(input_ids + [self.pad_token_id] * pad_length)
            padded_labels.append(labels + [-100] * pad_length)
            padded_attention_masks.append([1] * len(input_ids) + [0] * pad_length)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return self.torch_call(examples)
