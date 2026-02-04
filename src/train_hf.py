#!/usr/bin/env python3
"""
Alternative Training Script using HuggingFace Trainer API
==========================================================

This version uses the HuggingFace Trainer class which provides:
- Auto batch size detection (auto_find_batch_size)
- Built-in gradient checkpointing
- Easy checkpoint management
- Native WandB integration
- Simpler multi-GPU handling

Use this if you prefer the Trainer API over custom training loops.

Usage:
------
    python train_hf.py

With accelerate (for multi-GPU):
    accelerate launch train_hf.py
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback

# Local imports
from training_config import FullConfig, get_config
from dataset_loader import get_dataset, create_train_val_split
from collators import BiasedMLMDataCollator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SystemMetricsCallback(TrainerCallback):
    """
    Callback to log GPU memory and throughput metrics.

    Integrates with WandB and TensorBoard via Trainer's logging.
    """

    def __init__(self, log_interval_steps: int = 50):
        self.log_interval_steps = log_interval_steps
        self.step_count = 0
        self.total_tokens = 0
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1

        if self.step_count % self.log_interval_steps == 0:
            metrics = self._get_system_metrics()
            if metrics:
                # Log via trainer's logging mechanism
                for key, value in metrics.items():
                    state.log_history.append({key: value, "step": state.global_step})

    def _get_system_metrics(self) -> Dict[str, float]:
        """Collect GPU memory metrics."""
        metrics = {}

        if torch.cuda.is_available():
            metrics["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            metrics["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            metrics["system/gpu_max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

            # Memory utilization percentage
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated()
            metrics["system/gpu_memory_utilization"] = allocated / props.total_memory

        return metrics


class MLMTrainer(Trainer):
    """
    Custom Trainer with MLM-specific features.

    Extends the base Trainer with:
    - Custom loss logging
    - Perplexity calculation
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss and optionally return outputs."""
        outputs = model(**inputs)
        loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss

    def _compute_metrics(self, eval_preds):
        """Compute perplexity from loss."""
        # Note: eval_preds contains (predictions, labels)
        # For MLM, we typically just use the loss
        return {}


def train(config: FullConfig) -> None:
    """Main training function using HuggingFace Trainer."""

    # Set seed
    set_seed(config.reproducibility.seed)

    logger.info("=" * 60)
    logger.info("Turkish Legal Domain MLM Training (HF Trainer)")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Output: {config.data.output_dir}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        use_fast=config.model.use_fast_tokenizer,
        cache_dir=config.data.cache_dir,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        config.model.model_name,
        cache_dir=config.data.cache_dir,
    )

    logger.info(f"Model parameters: {model.num_parameters():,}")

    # Load dataset
    full_dataset = get_dataset(
        data_dir=config.data.data_dir,
        tokenizer=tokenizer,
        max_length=config.data.max_seq_length,
        stride=config.data.stride,
        min_chunk_length=config.data.min_chunk_length,
        seed=config.reproducibility.seed,
    )

    train_dataset, eval_dataset = create_train_val_split(
        full_dataset,
        val_ratio=0.1,
        seed=config.reproducibility.seed,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # Create collator
    data_collator = BiasedMLMDataCollator(
        tokenizer=tokenizer,
        terms_file_path=config.data.terms_file,
        mlm_probability=config.masking.mlm_probability,
        legal_bias_ratio=config.masking.legal_bias_ratio,
        mask_token_prob=config.masking.mask_token_prob,
        random_token_prob=config.masking.random_token_prob,
        whole_word_masking=config.masking.whole_word_masking,
        seed=config.reproducibility.seed,
    )

    # Calculate training steps
    num_train_samples = len(train_dataset)
    steps_per_epoch = math.ceil(
        num_train_samples /
        (config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps)
    )
    max_steps = steps_per_epoch * config.training.num_epochs
    warmup_steps = int(max_steps * config.training.warmup_ratio)

    # Determine run name
    run_name = config.monitoring.wandb_run_name or f"legal-mlm-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.data.output_dir,
        overwrite_output_dir=True,

        # Training hyperparameters
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,

        # Optimization
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type="linear",

        # Mixed precision
        fp16=config.training.fp16,
        bf16=config.training.bf16,

        # Auto batch size (Anti-OOM)
        auto_find_batch_size=config.training.auto_find_batch_size,

        # Logging
        logging_dir=os.path.join(config.data.output_dir, "logs"),
        logging_steps=config.training.logging_steps,
        logging_strategy="steps",
        report_to=["wandb", "tensorboard"] if config.monitoring.use_wandb else ["tensorboard"],

        # Evaluation
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,

        # Checkpointing
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Reproducibility
        seed=config.reproducibility.seed,
        data_seed=config.reproducibility.seed,

        # Performance
        dataloader_num_workers=config.data.num_workers,
        dataloader_pin_memory=config.data.pin_memory,

        # W&B
        run_name=run_name,

        # Resume
        resume_from_checkpoint=config.training.resume_from_checkpoint,
    )

    # Initialize W&B if enabled
    if config.monitoring.use_wandb:
        import wandb
        wandb.init(
            project=config.monitoring.wandb_project,
            name=run_name,
            entity=config.monitoring.wandb_entity,
            config=config.to_dict(),
            tags=["mlm", "legal", "turkish", "xlm-roberta"],
        )

    # Create trainer
    trainer = MLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[SystemMetricsCallback(log_interval_steps=config.training.logging_steps)],
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train(
        resume_from_checkpoint=config.training.resume_from_checkpoint
    )

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(config.data.output_dir, "final-model"))
    tokenizer.save_pretrained(os.path.join(config.data.output_dir, "final-model"))

    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Final eval loss: {eval_metrics['eval_loss']:.4f}")
    logger.info(f"Final perplexity: {eval_metrics['eval_perplexity']:.2f}")
    logger.info(f"Model saved to: {config.data.output_dir}/final-model")
    logger.info("=" * 60)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Train XLM-RoBERTa with HuggingFace Trainer"
    )
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--model_name", type=str, help="Model name override")
    parser.add_argument("--data_dir", type=str, help="Data directory override")
    parser.add_argument("--output_dir", type=str, help="Output directory override")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            config = FullConfig.from_dict(json.load(f))
    else:
        config = get_config()

    # Apply overrides
    if args.model_name:
        config.model.model_name = args.model_name
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.data.output_dir = args.output_dir
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.fp16:
        config.training.fp16 = True
    if args.bf16:
        config.training.bf16 = True
    if args.no_wandb:
        config.monitoring.use_wandb = False

    train(config)


if __name__ == "__main__":
    main()
