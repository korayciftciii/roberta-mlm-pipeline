"""
Training Configuration for Turkish Legal Domain MLM
====================================================
Centralized configuration for continued pre-training of XLM-RoBERTa on Turkish legal texts.
All hyperparameters and paths are defined here for reproducibility.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model and tokenizer configuration."""
    model_name: str = "FacebookAI/xlm-roberta-base"
    use_fast_tokenizer: bool = True
    # Alternatives: "FacebookAI/xlm-roberta-large" (requires more VRAM)


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    # Directories
    data_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "resources", "decisions"
    ))
    terms_file: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "resources", "terms", "legal_terms.txt"
    ))
    output_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs"
    ))
    cache_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".cache"
    ))

    # Tokenization
    max_seq_length: int = 512
    stride: int = 128  # Overlap: max_seq_length - stride = 384 tokens overlap (75%)
    min_chunk_length: int = 50  # Minimum tokens per chunk to keep

    # Dataloader
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class MaskingConfig:
    """
    Biased Masking Strategy Configuration.

    The masking follows a two-tier approach:
    1. LEGAL_BIAS_RATIO (70%): Tokens belonging to legal terms are prioritized
    2. Remaining budget (30%): Standard random masking

    Within masked tokens, we apply BERT-style replacement:
    - 80% → [MASK] token
    - 10% → Random token from vocabulary
    - 10% → Keep original token (but still compute loss)
    """
    mlm_probability: float = 0.15  # Total masking probability
    legal_bias_ratio: float = 0.70  # 70% of masked tokens should be legal terms

    # BERT-style masking distribution
    mask_token_prob: float = 0.80  # Replace with [MASK]
    random_token_prob: float = 0.10  # Replace with random token
    keep_token_prob: float = 0.10  # Keep original (loss still computed)

    # Whole Word Masking: If any subword of a word is selected, mask entire word
    whole_word_masking: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters and optimization settings."""
    # Basic training params
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Batch sizing (Anti-OOM)
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    # Effective batch size = per_device * num_gpus * gradient_accumulation

    # Auto batch size detection
    auto_find_batch_size: bool = True
    # If True, will halve batch size on OOM until it fits

    # Mixed Precision (Anti-OOM)
    fp16: bool = False  # Set True for NVIDIA GPUs with Tensor Cores
    bf16: bool = False  # Set True for Ampere+ GPUs or TPUs

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500
    logging_steps: int = 50

    # Resume training
    resume_from_checkpoint: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Experiment tracking configuration."""
    use_wandb: bool = True
    use_tensorboard: bool = True

    # WandB settings
    wandb_project: str = "turkish-legal-mlm"
    wandb_run_name: Optional[str] = None  # Auto-generated if None
    wandb_entity: Optional[str] = None

    # Logging
    log_system_metrics: bool = True  # VRAM, throughput tracking
    log_interval_seconds: int = 30


@dataclass
class ReproducibilityConfig:
    """Reproducibility settings."""
    seed: int = 42
    deterministic: bool = True
    # Note: deterministic=True may slightly reduce performance


@dataclass
class FullConfig:
    """Complete training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)

    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FullConfig":
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            masking=MaskingConfig(**config_dict.get("masking", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            monitoring=MonitoringConfig(**config_dict.get("monitoring", {})),
            reproducibility=ReproducibilityConfig(**config_dict.get("reproducibility", {}))
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)


def get_config() -> FullConfig:
    """Factory function to get default configuration."""
    return FullConfig()


if __name__ == "__main__":
    # Print default configuration
    import json
    config = get_config()
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
