"""
Dataset Loader for Turkish Legal MLM Training
==============================================

This module handles loading and preprocessing of Turkish court decisions
for continued pre-training. It implements a sliding window approach with
configurable overlap to maintain contextual continuity across chunks.

SLIDING WINDOW APPROACH:
========================
Legal documents are typically longer than the model's max sequence length (512).
Instead of truncating, we use overlapping windows:

Example with max_length=512, stride=128 (384 token overlap):
- Document tokens: [t0, t1, t2, ..., t1000]
- Chunk 1: [t0 ... t509] (512 tokens)
- Chunk 2: [t128 ... t637] (overlaps t128-t509 from chunk 1)
- Chunk 3: [t256 ... t765]
- ...

This ensures:
1. No context is lost at chunk boundaries
2. The model sees each token in multiple contexts
3. Better learning of document-level coherence
"""

import os
import random
from typing import Dict, Iterator, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

from utils import aggressive_legal_cleaner


@dataclass
class LegalDocumentDataset(Dataset):
    """
    Map-style dataset for legal documents with sliding window chunking.

    Loads all documents into memory and creates fixed-size chunks.
    Suitable for small-to-medium datasets that fit in RAM.

    Attributes:
        data_dir: Directory containing .txt files
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (default: 512)
        stride: Window stride (overlap = max_length - stride)
        min_chunk_length: Minimum tokens to keep a chunk
        seed: Random seed for shuffling
    """

    data_dir: str
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    stride: int = 128
    min_chunk_length: int = 50
    seed: int = 42

    def __post_init__(self):
        """Load and tokenize all documents."""
        self.chunks = []
        self._load_documents()

        # Shuffle chunks deterministically
        rng = random.Random(self.seed)
        rng.shuffle(self.chunks)

        print(f"[Dataset] Loaded {len(self.chunks)} chunks from {self.data_dir}")

    def _load_documents(self) -> None:
        """Load all .txt files and create chunks."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")

        files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        print(f"[Dataset] Processing {len(files)} documents...")

        for filename in tqdm(files, desc="Loading documents"):
            filepath = os.path.join(self.data_dir, filename)
            self._process_file(filepath)

    def _process_file(self, filepath: str) -> None:
        """Process a single file and create chunks."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except Exception as e:
            print(f"[WARNING] Could not read {filepath}: {e}")
            return

        # Clean text
        text = aggressive_legal_cleaner(raw_text)
        if len(text) < self.min_chunk_length:
            return

        # Tokenize entire document
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )

        input_ids = encoded['input_ids']
        offset_mapping = encoded['offset_mapping']
        word_ids = encoded.word_ids() if hasattr(encoded, 'word_ids') else None

        # Handle None in word_ids (from fast tokenizers)
        if word_ids is not None:
            word_ids = self._fix_word_ids(word_ids)

        # Create sliding window chunks
        # Reserve 2 tokens for [CLS] and [SEP]
        content_length = self.max_length - 2

        for start in range(0, len(input_ids), self.stride):
            end = min(start + content_length, len(input_ids))

            chunk_ids = input_ids[start:end]
            if len(chunk_ids) < self.min_chunk_length:
                continue

            chunk_offsets = offset_mapping[start:end]
            chunk_word_ids = word_ids[start:end] if word_ids else None

            # Add special tokens
            final_ids = [self.tokenizer.cls_token_id] + chunk_ids + [self.tokenizer.sep_token_id]
            final_offsets = [(0, 0)] + chunk_offsets + [(0, 0)]

            if chunk_word_ids is not None:
                final_word_ids = [None] + chunk_word_ids + [None]
            else:
                final_word_ids = None

            self.chunks.append({
                'input_ids': final_ids,
                'attention_mask': [1] * len(final_ids),
                'offset_mapping': final_offsets,
                'word_ids': final_word_ids,
                'original_text': text,  # Needed for legal term detection
            })

            # If we've processed all tokens, stop
            if end >= len(input_ids):
                break

    def _fix_word_ids(self, word_ids: List[Optional[int]]) -> List[int]:
        """Fix None values in word_ids from fast tokenizers."""
        fixed = []
        current_word_id = -1

        for wid in word_ids:
            if wid is not None:
                current_word_id = wid
                fixed.append(wid)
            else:
                # Assign to previous word or 0 if at start
                fixed.append(max(0, current_word_id))

        return fixed

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.chunks[idx]


class StreamingLegalDataset(IterableDataset):
    """
    Streaming dataset for large document collections.

    Processes files on-the-fly without loading everything into memory.
    Suitable for very large datasets that don't fit in RAM.

    Note: Shuffling is limited to a buffer with this approach.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        stride: int = 128,
        min_chunk_length: int = 50,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.min_chunk_length = min_chunk_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.txt')
        ]
        print(f"[StreamingDataset] Found {len(self.files)} documents")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield chunks from documents with buffered shuffling."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Split files among workers
            per_worker = len(self.files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.files)
            files = self.files[start:end]
        else:
            files = self.files

        # Shuffle files
        rng = random.Random(self.seed)
        rng.shuffle(files)

        # Buffer for shuffling chunks
        buffer = []

        for filepath in files:
            for chunk in self._process_file(filepath):
                buffer.append(chunk)

                # Shuffle and yield when buffer is full
                if len(buffer) >= self.shuffle_buffer_size:
                    rng.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

        # Yield remaining items
        if buffer:
            rng.shuffle(buffer)
            for item in buffer:
                yield item

    def _process_file(self, filepath: str) -> Iterator[Dict[str, Any]]:
        """Process a single file and yield chunks."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except Exception:
            return

        text = aggressive_legal_cleaner(raw_text)
        if len(text) < self.min_chunk_length:
            return

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )

        input_ids = encoded['input_ids']
        offset_mapping = encoded['offset_mapping']
        word_ids = encoded.word_ids() if hasattr(encoded, 'word_ids') else None

        if word_ids is not None:
            word_ids = self._fix_word_ids(word_ids)

        content_length = self.max_length - 2

        for start in range(0, len(input_ids), self.stride):
            end = min(start + content_length, len(input_ids))

            chunk_ids = input_ids[start:end]
            if len(chunk_ids) < self.min_chunk_length:
                continue

            chunk_offsets = offset_mapping[start:end]
            chunk_word_ids = word_ids[start:end] if word_ids else None

            final_ids = [self.tokenizer.cls_token_id] + chunk_ids + [self.tokenizer.sep_token_id]
            final_offsets = [(0, 0)] + chunk_offsets + [(0, 0)]

            if chunk_word_ids is not None:
                final_word_ids = [None] + chunk_word_ids + [None]
            else:
                final_word_ids = None

            yield {
                'input_ids': final_ids,
                'attention_mask': [1] * len(final_ids),
                'offset_mapping': final_offsets,
                'word_ids': final_word_ids,
                'original_text': text,
            }

            if end >= len(input_ids):
                break

    def _fix_word_ids(self, word_ids: List[Optional[int]]) -> List[int]:
        """Fix None values in word_ids."""
        fixed = []
        current_word_id = -1
        for wid in word_ids:
            if wid is not None:
                current_word_id = wid
                fixed.append(wid)
            else:
                fixed.append(max(0, current_word_id))
        return fixed


class PreTokenizedDataset(Dataset):
    """
    Dataset for pre-tokenized JSONL data.

    Use this with data already processed by the existing main.py script.
    Each line should be a JSON object with 'input_ids' and 'labels'.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        import ujson

        self.samples = []

        print(f"[PreTokenizedDataset] Loading from {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading samples")):
                if max_samples and i >= max_samples:
                    break
                sample = ujson.loads(line.strip())
                self.samples.append(sample)

        # Shuffle
        rng = random.Random(seed)
        rng.shuffle(self.samples)

        print(f"[PreTokenizedDataset] Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.samples[idx]


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into training and validation sets.

    Args:
        dataset: Full dataset
        val_ratio: Fraction for validation (default: 0.1)
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from torch.utils.data import Subset

    n = len(dataset)
    indices = list(range(n))

    rng = random.Random(seed)
    rng.shuffle(indices)

    split = int(n * (1 - val_ratio))
    train_indices = indices[:split]
    val_indices = indices[split:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def get_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    stride: int = 128,
    min_chunk_length: int = 50,
    seed: int = 42,
    streaming: bool = False,
    shuffle_buffer_size: int = 10000,
) -> Dataset:
    """
    Factory function to create appropriate dataset.

    Args:
        data_dir: Directory with .txt files
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        stride: Sliding window stride
        min_chunk_length: Minimum chunk size
        seed: Random seed
        streaming: Use streaming dataset (for large data)
        shuffle_buffer_size: Buffer size for streaming shuffle

    Returns:
        Dataset instance
    """
    if streaming:
        return StreamingLegalDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            min_chunk_length=min_chunk_length,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
    else:
        return LegalDocumentDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            min_chunk_length=min_chunk_length,
            seed=seed,
        )


if __name__ == "__main__":
    # Test dataset loading
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "resources", "decisions"
    )

    dataset = get_dataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=512,
        stride=128,
        seed=42,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample keys: {dataset[0].keys()}")
    print(f"Sample input_ids length: {len(dataset[0]['input_ids'])}")
