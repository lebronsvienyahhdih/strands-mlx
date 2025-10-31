"""MLX LoRA Training Tool for Strands Agents.

This tool wraps mlx-lm's LoRA training functionality for fine-tuning models
on training data collected from Strands conversations.
"""

import math
import os
import types
from pathlib import Path
from typing import Dict, Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from strands import tool


@tool
def mlx_trainer(
    action: str,
    model: str = "mlx-community/Qwen3-1.7B-4bit",
    data: str = None,
    adapter_path: str = "./adapters",
    batch_size: int = 4,
    iters: int = 1000,
    learning_rate: float = 1e-5,
    num_layers: int = 16,
    lora_rank: int = 8,
    lora_dropout: float = 0.0,
    lora_scale: float = 20.0,
    val_batches: int = 25,
    test_batches: int = 500,
    steps_per_report: int = 10,
    steps_per_eval: int = 200,
    save_every: int = 100,
    max_seq_length: int = 2048,
    seed: int = 0,
    fine_tune_type: str = "lora",
    optimizer: str = "adam",
    grad_checkpoint: bool = False,
    grad_accumulation_steps: int = 1,
    resume_adapter_file: Optional[str] = None,
    mask_prompt: bool = False,
    lr_schedule: Optional[str] = None,
) -> Dict[str, Any]:
    """MLX LoRA training tool for fine-tuning models.

    Args:
        action: Action to perform:
            - "train": Train model with LoRA
            - "test": Evaluate model on test set
            - "train_and_test": Train then evaluate
        model: Model path or HuggingFace repo (e.g., "mlx-community/Qwen3-1.7B-4bit")
        data: Path to training data directory with train.jsonl, valid.jsonl, test.jsonl
              OR path to single .jsonl file (will auto-split)
        adapter_path: Directory to save/load adapter weights
        batch_size: Training batch size
        iters: Number of training iterations
        learning_rate: Learning rate for optimizer
        num_layers: Number of layers to fine-tune (-1 for all)
        lora_rank: LoRA rank parameter
        lora_dropout: LoRA dropout rate
        lora_scale: LoRA scaling factor
        val_batches: Number of validation batches (-1 for full validation set)
        test_batches: Number of test batches (-1 for full test set)
        steps_per_report: Report loss every N steps
        steps_per_eval: Evaluate every N steps
        save_every: Save checkpoint every N iterations
        max_seq_length: Maximum sequence length
        seed: Random seed
        fine_tune_type: Type of fine-tuning ("lora", "dora", or "full")
        optimizer: Optimizer to use ("adam", "adamw", "muon", "sgd", "adafactor")
        grad_checkpoint: Use gradient checkpointing
        grad_accumulation_steps: Gradient accumulation steps
        resume_adapter_file: Path to resume training from
        mask_prompt: Mask prompt in loss calculation
        lr_schedule: Learning rate schedule config

    Returns:
        Dict containing status and training results
    """
    try:
        from mlx_lm.lora import run, CONFIG_DEFAULTS
        from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset
        from mlx_lm.utils import load as mlx_load

        # Validate action
        if action not in ["train", "test", "train_and_test"]:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Invalid action: {action}. Must be 'train', 'test', or 'train_and_test'."
                    }
                ],
            }

        # Validate data path
        if data is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "Data path is required. Provide path to .jsonl file or directory with train/valid/test.jsonl"
                    }
                ],
            }

        data_path = Path(data).expanduser().resolve()
        if not data_path.exists():
            return {
                "status": "error",
                "content": [{"text": f"Data path does not exist: {data_path}"}],
            }

        # Build args namespace
        args = types.SimpleNamespace(
            model=model,
            train=(action in ["train", "train_and_test"]),
            test=(action in ["test", "train_and_test"]),
            data=str(data_path),
            adapter_path=adapter_path,
            batch_size=batch_size,
            iters=iters,
            learning_rate=learning_rate,
            num_layers=num_layers,
            lora_parameters={
                "rank": lora_rank,
                "dropout": lora_dropout,
                "scale": lora_scale,
            },
            val_batches=val_batches,
            test_batches=test_batches,
            steps_per_report=steps_per_report,
            steps_per_eval=steps_per_eval,
            save_every=save_every,
            max_seq_length=max_seq_length,
            seed=seed,
            fine_tune_type=fine_tune_type,
            optimizer=optimizer,
            optimizer_config=CONFIG_DEFAULTS["optimizer_config"],
            grad_checkpoint=grad_checkpoint,
            grad_accumulation_steps=grad_accumulation_steps,
            resume_adapter_file=resume_adapter_file,
            mask_prompt=mask_prompt,
            lr_schedule=lr_schedule,
            report_to=None,
            project_name=None,
            config=None,
        )

        # Set tokenizers parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # Run training/testing
        print(f"\n🚀 Starting MLX LoRA {action}...")
        print(f"📦 Model: {model}")
        print(f"📊 Data: {data_path}")
        print(f"💾 Adapters: {adapter_path}")
        print(f"🔧 Config: batch_size={batch_size}, iters={iters}, lr={learning_rate}")
        print(f"🎯 LoRA: rank={lora_rank}, dropout={lora_dropout}, scale={lora_scale}\n")

        # Execute
        run(args)

        # Build success response
        adapter_path_obj = Path(adapter_path)
        adapter_file = adapter_path_obj / "adapters.safetensors"
        config_file = adapter_path_obj / "adapter_config.json"

        result_text = f"✅ **MLX LoRA {action} complete!**\n\n"

        if args.train:
            result_text += "**📊 Training Summary:**\n"
            result_text += f"- Model: {model}\n"
            result_text += f"- Iterations: {iters}\n"
            result_text += f"- Batch size: {batch_size}\n"
            result_text += f"- Learning rate: {learning_rate}\n"
            result_text += f"- LoRA rank: {lora_rank}\n\n"

            result_text += "**💾 Saved Files:**\n"
            if adapter_file.exists():
                size_mb = adapter_file.stat().st_size / (1024 * 1024)
                result_text += f"- Adapters: {adapter_file} ({size_mb:.1f}MB)\n"
            if config_file.exists():
                result_text += f"- Config: {config_file}\n"

        if args.test:
            result_text += "\n**🧪 Testing complete** - see logs above for results\n"

        result_text += f"\n**🎯 Next steps:**\n"
        result_text += f"```python\n"
        result_text += f"# Load model with trained adapter\n"
        result_text += f"from strands_mlx import MLXModel\n\n"
        result_text += f"model = MLXModel(\n"
        result_text += f'    model_id="{model}",\n'
        result_text += f'    adapter_path="{adapter_path}"\n'
        result_text += f")\n"
        result_text += f"```"

        return {"status": "success", "content": [{"text": result_text}]}

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return {
            "status": "error",
            "content": [{"text": f"❌ **Training failed:**\n\n```\n{error_trace}\n```"}],
        }
