"""
Agricultural Insight Distillation - Step 2: Fine-Tune with LoRA
===============================================================

This script fine-tunes Phi-3 Mini using LoRA (Low-Rank Adaptation) for
efficient training. LoRA is the industry-standard approach for:
- Memory-efficient fine-tuning
- Faster training
- Easy model merging and deployment

Author: [Your Name]
"""

import json
import torch
from pathlib import Path
from datetime import datetime

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# =============================================================================
# CONFIGURATION - Optimized for NVIDIA DGX Spark (Blackwell GPU, 128GB RAM)
# =============================================================================

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "models/phi3-agricultural-analyst"
MAX_SEQ_LENGTH = 2048  # Increased - more context capacity
LORA_R = 64  # Increased rank for better quality with powerful GPU
LORA_ALPHA = 128  # Scaled with rank
LORA_DROPOUT = 0.05

# Training hyperparameters - optimized for DGX Spark
TRAIN_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,  # Large batch - leverages GPU memory
    "gradient_accumulation_steps": 2,  # Effective batch = 32
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",  # Updated for newer transformers
    "bf16": True,  # BF16 - native Blackwell support, better than FP16
    "optim": "adamw_torch_fused",  # Fused optimizer - faster on modern GPUs
    "dataloader_num_workers": 4,  # Parallel data loading
    "dataloader_pin_memory": True,  # Faster GPU transfer
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data():
    """Load the generated training data."""
    train_path = Path("data/train.json")
    eval_path = Path("data/eval.json")
    
    if not train_path.exists():
        raise FileNotFoundError(
            "Training data not found. Run 01_generate_training_data.py first."
        )
    
    train_data = json.loads(train_path.read_text())
    eval_data = json.loads(eval_path.read_text())
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    return train_dataset, eval_dataset

# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model_and_tokenizer():
    """Load model in full precision - optimized for DGX Spark with 128GB RAM."""

    print(f"\n[1/4] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"\n[2/4] Loading model in BF16...")

    # Full precision loading - DGX Spark has plenty of memory
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # BF16 - native Blackwell support
        attn_implementation="eager",  # Eager attention for Phi-3 compatibility
    )

    # Enable gradient checkpointing for memory efficiency during training
    model.gradient_checkpointing_enable()
    
    print(f"\n[3/4] Applying LoRA configuration...")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer

# =============================================================================
# TRAINING
# =============================================================================

def train():
    """Main training function."""
    
    print("=" * 60)
    print("Agricultural Insight Distillation - LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\n[Loading Data]")
    train_dataset, eval_dataset = load_training_data()
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    
    # Setup model
    print("\n[Setting Up Model]")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Training arguments
    print(f"\n[4/4] Configuring trainer...")
    
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        logging_steps=TRAIN_CONFIG["logging_steps"],
        save_strategy=TRAIN_CONFIG["save_strategy"],
        eval_strategy=TRAIN_CONFIG["eval_strategy"],
        bf16=TRAIN_CONFIG["bf16"],  # BF16 for Blackwell GPU
        optim=TRAIN_CONFIG["optim"],
        dataloader_num_workers=TRAIN_CONFIG["dataloader_num_workers"],
        dataloader_pin_memory=TRAIN_CONFIG["dataloader_pin_memory"],
        report_to="none",  # Disable wandb for simplicity
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,  # Memory efficient training
        dataset_text_field="text",  # Field containing training text
        max_length=MAX_SEQ_LENGTH,  # Max sequence length
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Updated API: tokenizer -> processing_class
    )
    
    # Train!
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    
    trainer.train()
    
    # Save the final model
    print("\n[Saving Model]")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    # Save training info
    training_info = {
        "model_name": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "training_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "training_config": TRAIN_CONFIG,
        "completed_at": datetime.now().isoformat(),
    }
    Path(f"{OUTPUT_DIR}/training_info.json").write_text(json.dumps(training_info, indent=2))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {OUTPUT_DIR}/final")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return trainer


if __name__ == "__main__":
    train()
