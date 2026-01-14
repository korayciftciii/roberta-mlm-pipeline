#!/usr/bin/env python3
"""
Turkish Legal Domain MLM Training Script
=========================================

Continued pre-training of XLM-RoBERTa on Turkish legal court decisions
using biased Masked Language Modeling with legal terminology focus.

Features:
---------
- Biased masking: 70% legal terms, 30% random
- Sliding window chunking with configurable overlap
- Multi-GPU support via Hugging Face Accelerate
- Mixed precision training (FP16/BF16)
- Auto batch size detection (OOM recovery)
- WandB and TensorBoard integration
- Detailed system metrics logging

Usage:
------
Single GPU:
    python train.py

Multi-GPU with Accelerate:
    accelerate launch train.py

With custom config:
    python train.py --config path/to/config.json

Author: Generated for Turkish Legal NLP Project
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm

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


class SystemMetricsLogger:
    """
    Logs system metrics (GPU memory, throughput) during training.

    Tracks:
    - GPU memory usage (allocated, reserved, max allocated)
    - Training throughput (samples/second, tokens/second)
    - Step timing statistics
    """

    def __init__(self, log_interval: int = 30):
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.step_times = []
        self.samples_processed = 0
        self.tokens_processed = 0

    def update(
        self,
        step_time: float,
        batch_size: int,
        seq_length: int,
    ) -> None:
        """Update metrics after a training step."""
        self.step_times.append(step_time)
        self.samples_processed += batch_size
        self.tokens_processed += batch_size * seq_length

    def should_log(self) -> bool:
        """Check if it's time to log metrics."""
        return time.time() - self.last_log_time >= self.log_interval

    def get_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        metrics = {}

        # Timing metrics
        if self.step_times:
            elapsed = time.time() - self.last_log_time
            metrics["throughput/samples_per_sec"] = self.samples_processed / max(elapsed, 1e-6)
            metrics["throughput/tokens_per_sec"] = self.tokens_processed / max(elapsed, 1e-6)
            metrics["timing/avg_step_time_ms"] = sum(self.step_times) / len(self.step_times) * 1000

        # GPU memory metrics (if available)
        if torch.cuda.is_available():
            metrics["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            metrics["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            metrics["gpu/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

            # Per-device metrics for multi-GPU
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / 1e9
                total = props.total_memory / 1e9
                metrics[f"gpu_{i}/memory_allocated_gb"] = allocated
                metrics[f"gpu_{i}/memory_utilization"] = allocated / total

        return metrics

    def reset(self) -> None:
        """Reset counters after logging."""
        self.last_log_time = time.time()
        self.step_times = []
        self.samples_processed = 0
        self.tokens_processed = 0


def setup_accelerator(config: FullConfig) -> Accelerator:
    """
    Configure Accelerate for distributed training.

    Handles:
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Multi-GPU distribution
    - WandB/TensorBoard logging
    """
    project_config = ProjectConfiguration(
        project_dir=config.data.output_dir,
        logging_dir=os.path.join(config.data.output_dir, "logs"),
    )

    # Determine mixed precision mode
    mixed_precision = None
    if config.training.bf16:
        mixed_precision = "bf16"
    elif config.training.fp16:
        mixed_precision = "fp16"

    # Configure logging
    log_with = []
    if config.monitoring.use_wandb:
        log_with.append("wandb")
    if config.monitoring.use_tensorboard:
        log_with.append("tensorboard")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with if log_with else None,
        project_config=project_config,
    )

    return accelerator


def setup_wandb(accelerator: Accelerator, config: FullConfig) -> None:
    """Initialize Weights & Biases tracking."""
    if not config.monitoring.use_wandb:
        return

    run_name = config.monitoring.wandb_run_name or f"legal-mlm-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    accelerator.init_trackers(
        project_name=config.monitoring.wandb_project,
        config=config.to_dict(),
        init_kwargs={
            "wandb": {
                "name": run_name,
                "entity": config.monitoring.wandb_entity,
                "tags": ["mlm", "legal", "turkish", "xlm-roberta"],
            }
        },
    )


def load_model_and_tokenizer(
    config: FullConfig,
    accelerator: Accelerator,
) -> Tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Load pre-trained model and tokenizer.

    Uses AutoClasses for compatibility with various model architectures.
    """
    logger.info(f"Loading model: {config.model.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        use_fast=config.model.use_fast_tokenizer,
        cache_dir=config.data.cache_dir,
    )

    model_config = AutoConfig.from_pretrained(
        config.model.model_name,
        cache_dir=config.data.cache_dir,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        config.model.model_name,
        config=model_config,
        cache_dir=config.data.cache_dir,
    )

    logger.info(f"Model loaded: {model.num_parameters():,} parameters")

    return model, tokenizer


def create_dataloaders(
    config: FullConfig,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.

    Features:
    - Sliding window chunking
    - Biased masking collator
    - Pin memory for faster GPU transfer
    - Multi-worker loading
    """
    logger.info(f"Loading dataset from: {config.data.data_dir}")

    # Create dataset
    full_dataset = get_dataset(
        data_dir=config.data.data_dir,
        tokenizer=tokenizer,
        max_length=config.data.max_seq_length,
        stride=config.data.stride,
        min_chunk_length=config.data.min_chunk_length,
        seed=config.reproducibility.seed,
        streaming=False,  # Use map-style for small datasets
    )

    # Split into train/val
    train_dataset, val_dataset = create_train_val_split(
        full_dataset,
        val_ratio=0.1,
        seed=config.reproducibility.seed,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create collator
    collator = BiasedMLMDataCollator(
        tokenizer=tokenizer,
        terms_file_path=config.data.terms_file,
        mlm_probability=config.masking.mlm_probability,
        legal_bias_ratio=config.masking.legal_bias_ratio,
        mask_token_prob=config.masking.mask_token_prob,
        random_token_prob=config.masking.random_token_prob,
        whole_word_masking=config.masking.whole_word_masking,
        seed=config.reproducibility.seed,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    return train_dataloader, val_dataloader


def train_epoch(
    model: AutoModelForMaskedLM,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    config: FullConfig,
    epoch: int,
    global_step: int,
    metrics_logger: SystemMetricsLogger,
) -> Tuple[float, int]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}",
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress_bar:
        step_start_time = time.time()

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            num_batches += 1

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })

            # Update metrics
            step_time = time.time() - step_start_time
            batch_size = batch["input_ids"].shape[0]
            seq_length = batch["input_ids"].shape[1]
            metrics_logger.update(step_time, batch_size, seq_length)

            # Log to trackers
            if global_step % config.training.logging_steps == 0:
                logs = {
                    "train/loss": loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                }

                # Add system metrics if enabled
                if config.monitoring.log_system_metrics and metrics_logger.should_log():
                    logs.update(metrics_logger.get_metrics())
                    metrics_logger.reset()

                accelerator.log(logs, step=global_step)

            # Save checkpoint
            if global_step % config.training.save_steps == 0:
                save_checkpoint(accelerator, model, config, global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss.item(), global_step


@torch.no_grad()
def evaluate(
    model: AutoModelForMaskedLM,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Returns:
        Dictionary with evaluation metrics (loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(
        val_dataloader,
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    ):
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    # Gather across processes
    avg_loss = accelerator.gather(avg_loss).mean().item()

    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "eval/loss": avg_loss,
        "eval/perplexity": perplexity,
    }


def save_checkpoint(
    accelerator: Accelerator,
    model: AutoModelForMaskedLM,
    config: FullConfig,
    global_step: int,
) -> None:
    """Save model checkpoint."""
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(config.data.output_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            save_function=accelerator.save,
        )

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # Clean up old checkpoints
        cleanup_checkpoints(config.data.output_dir, config.training.save_total_limit)


def cleanup_checkpoints(output_dir: str, limit: int) -> None:
    """Keep only the most recent checkpoints."""
    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            step = int(name.split("-")[1])
            checkpoints.append((step, name))

    checkpoints.sort(reverse=True)

    for step, name in checkpoints[limit:]:
        import shutil
        checkpoint_path = os.path.join(output_dir, name)
        shutil.rmtree(checkpoint_path, ignore_errors=True)
        logger.info(f"Deleted old checkpoint: {name}")


def train(config: FullConfig) -> None:
    """
    Main training loop.

    Orchestrates the entire training process:
    1. Setup accelerator and logging
    2. Load model and tokenizer
    3. Create dataloaders
    4. Configure optimizer and scheduler
    5. Run training loop with evaluation
    6. Save final model
    """
    # Set seed for reproducibility
    set_seed(config.reproducibility.seed)

    # Setup accelerator
    accelerator = setup_accelerator(config)

    # Setup logging
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("Turkish Legal Domain MLM Training")
        logger.info("=" * 60)
        logger.info(f"Model: {config.model.model_name}")
        logger.info(f"Output directory: {config.data.output_dir}")
        logger.info(f"Mixed precision: {'bf16' if config.training.bf16 else 'fp16' if config.training.fp16 else 'no'}")
        logger.info(f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info("=" * 60)

    # Initialize W&B
    setup_wandb(accelerator, config)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, accelerator)

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(config, tokenizer, accelerator)

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.training.gradient_accumulation_steps
    )
    max_train_steps = config.training.num_epochs * num_update_steps_per_epoch
    warmup_steps = int(max_train_steps * config.training.warmup_ratio)

    if accelerator.is_main_process:
        logger.info(f"Steps per epoch: {num_update_steps_per_epoch}")
        logger.info(f"Total training steps: {max_train_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

    # Setup optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.training.learning_rate,
    )

    # Setup scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare for distributed training
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Resume from checkpoint if specified
    global_step = 0
    if config.training.resume_from_checkpoint:
        accelerator.load_state(config.training.resume_from_checkpoint)
        global_step = int(config.training.resume_from_checkpoint.split("-")[-1])
        logger.info(f"Resumed from checkpoint: {config.training.resume_from_checkpoint}")

    # Initialize metrics logger
    metrics_logger = SystemMetricsLogger(
        log_interval=config.monitoring.log_interval_seconds
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(config.training.num_epochs):
        # Train
        train_loss, global_step = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            config=config,
            epoch=epoch,
            global_step=global_step,
            metrics_logger=metrics_logger,
        )

        # Evaluate
        if val_dataloader is not None:
            eval_metrics = evaluate(model, val_dataloader, accelerator)

            if accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={eval_metrics['eval/loss']:.4f}, "
                    f"perplexity={eval_metrics['eval/perplexity']:.2f}"
                )

                accelerator.log(eval_metrics, step=global_step)

                # Save best model
                if eval_metrics["eval/loss"] < best_val_loss:
                    best_val_loss = eval_metrics["eval/loss"]
                    save_checkpoint(accelerator, model, config, global_step)

    # Save final model
    if accelerator.is_main_process:
        final_dir = os.path.join(config.data.output_dir, "final-model")
        os.makedirs(final_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        logger.info(f"Training complete! Final model saved to: {final_dir}")

    # End logging
    accelerator.end_training()


def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train XLM-RoBERTa on Turkish Legal Documents"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per-device batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable BF16 mixed precision",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = FullConfig.from_dict(config_dict)
    else:
        config = get_config()

    # Apply command-line overrides
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
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.fp16:
        config.training.fp16 = True
    if args.bf16:
        config.training.bf16 = True
    if args.no_wandb:
        config.monitoring.use_wandb = False

    # Start training
    train(config)


if __name__ == "__main__":
    main()
