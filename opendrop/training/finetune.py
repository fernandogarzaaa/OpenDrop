"""Fine-tuning pipeline for OpenDrop.

Supports:
  - lora:  LoRA via PEFT + Transformers (CUDA / CPU)
  - qlora: QLoRA via bitsandbytes + PEFT (low VRAM, CUDA)
  - full:  Full fine-tune for small models
  - mlx:   MLX-LM fine-tuning (Apple Silicon)

After training, produces either:
  a) A merged GGUF (via llama.cpp convert + quantize) for lora/full
  b) A LoRA adapter directory for adapter-only workflows
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from opendrop.core.converter import convert_and_quantize
from opendrop.training.dataset import Dataset, format_sample_for_training, load_dataset

console = Console()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    method: str = "lora"              # lora | qlora | full | mlx
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    fp16: bool = True
    bf16: bool = False
    save_steps: int = 100
    logging_steps: int = 10
    output_quant: str = "Q4_K_M"      # Quant for merged GGUF output


@dataclass
class TrainingResult:
    adapter_dir: Optional[Path]
    merged_gguf: Optional[Path]
    method: str
    base_model_id: str
    epochs: int
    final_loss: float


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch not installed. Run: pip install opendrop[training]"
        ) from exc


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Transformers not installed. Run: pip install opendrop[training]"
        ) from exc


def _require_peft() -> None:
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PEFT not installed. Run: pip install opendrop[training]"
        ) from exc


def _require_mlx() -> None:
    try:
        import mlx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MLX not installed. Run: pip install opendrop[training-apple]"
        ) from exc


# ---------------------------------------------------------------------------
# LoRA / QLoRA training (CUDA / CPU path)
# ---------------------------------------------------------------------------

def _train_lora_peft(
    model_id: str,
    data: Dataset,
    cfg: TrainingConfig,
    output_dir: Path,
    use_qlora: bool = False,
    token: Optional[str] = None,
) -> Path:
    """Run LoRA or QLoRA training via PEFT + TRL SFTTrainer."""
    _require_torch()
    _require_transformers()
    _require_peft()

    import torch
    from datasets import Dataset as HFDataset  # type: ignore[import]
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore

    try:
        from trl import SFTTrainer  # type: ignore[import]
        has_trl = True
    except ImportError:
        has_trl = False

    console.print(f"[bold]Loading tokenizer for {model_id} …[/bold]")
    tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    console.print(f"[bold]Loading model {model_id} …[/bold]")
    quant_cfg = None
    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except ImportError:
            console.print("[yellow]bitsandbytes not available — falling back to fp16 LoRA[/yellow]")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16 if not quant_cfg else None,
    )

    if use_qlora and quant_cfg:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Format samples
    texts = [format_sample_for_training(s, tok) for s in data]
    texts = [t for t in texts if t.strip()]
    hf_ds = HFDataset.from_dict({"text": texts})

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16 and not use_qlora,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to="none",
    )

    if has_trl:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tok,
            args=train_args,
            train_dataset=hf_ds,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
        )
    else:
        from transformers import DataCollatorForLanguageModeling, Trainer  # type: ignore
        tokenized = hf_ds.map(
            lambda x: tok(x["text"], truncation=True, max_length=cfg.max_seq_length),
            batched=True,
            remove_columns=["text"],
        )
        trainer = Trainer(
            model=model,
            tokenizer=tok,
            args=train_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        )

    console.print("[bold green]Training started …[/bold green]")
    train_result = trainer.train()
    final_loss = train_result.training_loss

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    console.print(f"[green]✓ Adapter saved: {adapter_dir}[/green]")
    return adapter_dir


# ---------------------------------------------------------------------------
# MLX fine-tuning (Apple Silicon)
# ---------------------------------------------------------------------------

def _train_mlx(
    model_id: str,
    data: Dataset,
    cfg: TrainingConfig,
    output_dir: Path,
    token: Optional[str] = None,
) -> Path:
    """Run LoRA fine-tuning via mlx-lm (Apple Silicon only)."""
    _require_mlx()

    try:
        import mlx_lm  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm not installed. Run: pip install opendrop[training-apple]"
        ) from exc

    # Write dataset as JSONL for mlx-lm
    data_file = output_dir / "train.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(data_file, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(__import__("json").dumps(sample) + "\n")

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_id,
        "--train",
        "--data", str(data_file),
        "--adapter-path", str(adapter_dir),
        "--num-layers", str(cfg.lora_rank),
        "--iters", str(cfg.num_epochs * len(data) // cfg.batch_size),
        "--batch-size", str(cfg.batch_size),
        "--learning-rate", str(cfg.learning_rate),
    ]
    if token:
        env = os.environ.copy()
        env["HF_TOKEN"] = token
    else:
        env = None

    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError("mlx-lm fine-tuning failed")

    console.print(f"[green]✓ MLX adapter saved: {adapter_dir}[/green]")
    return adapter_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fine_tune(
    model_id: str,
    dataset_source: str,
    output_dir: Path,
    cfg: Optional[TrainingConfig] = None,
    token: Optional[str] = None,
    produce_gguf: bool = True,
) -> TrainingResult:
    """Run the full fine-tuning pipeline.

    Args:
        model_id:       HuggingFace model ID or local path of the base model.
        dataset_source: Path to dataset file or HF dataset ID.
        output_dir:     Directory for all training outputs.
        cfg:            Training configuration. Defaults are used if None.
        token:          HuggingFace token for private models.
        produce_gguf:   Whether to convert the merged model to GGUF afterward.

    Returns:
        TrainingResult with paths to adapter and/or merged GGUF.
    """
    if cfg is None:
        cfg = TrainingConfig()

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading dataset from[/bold] {dataset_source} …")
    data = load_dataset(dataset_source)
    console.print(f"  {len(data)} samples loaded.")

    adapter_dir: Optional[Path] = None
    merged_gguf: Optional[Path] = None
    final_loss: float = 0.0

    # --- MLX ----------------------------------------------------------------
    if cfg.method == "mlx":
        if platform.system() != "Darwin":
            raise RuntimeError("MLX fine-tuning is only supported on macOS (Apple Silicon).")
        adapter_dir = _train_mlx(model_id, data, cfg, output_dir, token=token)

    # --- LoRA / QLoRA -------------------------------------------------------
    elif cfg.method in ("lora", "qlora"):
        use_qlora = cfg.method == "qlora"
        adapter_dir = _train_lora_peft(
            model_id, data, cfg, output_dir, use_qlora=use_qlora, token=token
        )

    # --- Full fine-tune -----------------------------------------------------
    elif cfg.method == "full":
        _require_torch()
        _require_transformers()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # type: ignore
        from datasets import Dataset as HFDataset  # type: ignore[import]
        from transformers import DataCollatorForLanguageModeling  # type: ignore

        console.print("[bold]Full fine-tune — loading model …[/bold]")
        tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto",
        )
        texts = [format_sample_for_training(s, tok) for s in data if format_sample_for_training(s, tok).strip()]
        hf_ds = HFDataset.from_dict({"text": texts})
        train_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            learning_rate=cfg.learning_rate,
            fp16=True,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            report_to="none",
        )
        tokenized = hf_ds.map(
            lambda x: tok(x["text"], truncation=True, max_length=cfg.max_seq_length),
            batched=True,
            remove_columns=["text"],
        )
        trainer = Trainer(
            model=model,
            tokenizer=tok,
            args=train_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        )
        train_result = trainer.train()
        final_loss = train_result.training_loss
        merged_model_dir = output_dir / "merged"
        model.save_pretrained(str(merged_model_dir))
        tok.save_pretrained(str(merged_model_dir))
        adapter_dir = merged_model_dir

    else:
        raise ValueError(f"Unknown training method: '{cfg.method}'. "
                         "Choose from: lora, qlora, full, mlx")

    # --- Produce GGUF -------------------------------------------------------
    if produce_gguf and adapter_dir and cfg.method != "mlx":
        try:
            console.print("[bold]Converting fine-tuned model → GGUF …[/bold]")
            gguf_dir = output_dir / "gguf"
            # For LoRA we need to merge first
            if cfg.method in ("lora", "qlora"):
                _merge_lora_then_convert(
                    base_model_id=model_id,
                    adapter_dir=adapter_dir,
                    gguf_dir=gguf_dir,
                    quant=cfg.output_quant,
                    token=token,
                )
                merged_gguf = next(gguf_dir.glob("*.gguf"), None)
            elif cfg.method == "full":
                merged_gguf = convert_and_quantize(adapter_dir, gguf_dir, cfg.output_quant)
        except Exception as exc:
            console.print(f"[yellow]⚠ GGUF conversion failed: {exc}[/yellow]")
            console.print("[dim]The adapter is still usable directly.[/dim]")

    return TrainingResult(
        adapter_dir=adapter_dir,
        merged_gguf=merged_gguf,
        method=cfg.method,
        base_model_id=model_id,
        epochs=cfg.num_epochs,
        final_loss=final_loss,
    )


def _merge_lora_then_convert(
    base_model_id: str,
    adapter_dir: Path,
    gguf_dir: Path,
    quant: str,
    token: Optional[str],
) -> None:
    """Merge LoRA weights into base model, then convert to GGUF."""
    _require_transformers()
    _require_peft()

    import torch
    from peft import AutoPeftModelForCausalLM  # type: ignore[import]
    from transformers import AutoTokenizer  # type: ignore[import]

    console.print("[bold]Merging LoRA adapter into base model …[/bold]")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        token=token,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    merged = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(str(adapter_dir), token=token)

    merged_dir = adapter_dir.parent / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tok.save_pretrained(str(merged_dir))
    console.print(f"[green]✓ Merged model saved: {merged_dir}[/green]")

    convert_and_quantize(merged_dir, gguf_dir, quant)
