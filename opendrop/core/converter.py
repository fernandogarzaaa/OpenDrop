"""Model format converter for OpenDrop.

Wraps llama.cpp's convert_hf_to_gguf.py and the llama-quantize binary to
convert SafeTensors / PyTorch checkpoints to GGUF and apply quantization.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


class ConversionError(RuntimeError):
    pass


def _find_convert_script() -> Optional[Path]:
    """Locate convert_hf_to_gguf.py from common llama.cpp install locations."""
    candidates = [
        shutil.which("convert_hf_to_gguf.py"),
        shutil.which("convert-hf-to-gguf"),
        "/usr/local/lib/llama.cpp/convert_hf_to_gguf.py",
        "/opt/homebrew/lib/llama.cpp/convert_hf_to_gguf.py",
        str(Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def _find_quantize_bin() -> Optional[Path]:
    """Locate llama-quantize binary."""
    candidates = [
        shutil.which("llama-quantize"),
        shutil.which("llama_quantize"),
        "/usr/local/bin/llama-quantize",
        "/opt/homebrew/bin/llama-quantize",
        str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def _run(cmd: list[str], desc: str) -> None:
    console.print(f"[dim]$ {' '.join(str(c) for c in cmd)}[/dim]")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise ConversionError(f"{desc} failed (exit {result.returncode}).")


def is_safetensors_dir(path: Path) -> bool:
    """Return True if *path* looks like a SafeTensors / HF model directory."""
    if not path.is_dir():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def needs_conversion(path: Path) -> bool:
    """Return True if the model at *path* needs conversion to GGUF."""
    if path.is_file() and path.suffix == ".gguf":
        return False
    if path.is_dir() and any(path.glob("*.gguf")):
        return False
    return is_safetensors_dir(path)


def convert_to_gguf(
    model_dir: Path,
    output_dir: Path,
    outtype: str = "f16",
    extra_args: Optional[list[str]] = None,
) -> Path:
    """Convert a SafeTensors/PyTorch model directory to a fp16 GGUF.

    Args:
        model_dir:  Directory containing the HF checkpoint.
        output_dir: Where to write the output GGUF.
        outtype:    llama.cpp output type ('f16', 'bf16', 'f32').
        extra_args: Additional arguments forwarded to convert_hf_to_gguf.py.

    Returns:
        Path to the output .gguf file.
    """
    script = _find_convert_script()
    if not script:
        raise ConversionError(
            "convert_hf_to_gguf.py not found. "
            "Install llama.cpp: https://github.com/ggml-org/llama.cpp#build"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_dir.name
    out_file = output_dir / f"{model_name}-{outtype}.gguf"

    cmd = [
        sys.executable, str(script),
        str(model_dir),
        "--outtype", outtype,
        "--outfile", str(out_file),
    ]
    if extra_args:
        cmd.extend(extra_args)

    _run(cmd, "SafeTensors → GGUF conversion")

    if not out_file.exists():
        # Some versions write with a different naming scheme
        gguf_files = list(output_dir.glob("*.gguf"))
        if not gguf_files:
            raise ConversionError(
                f"Conversion ran but no .gguf found in {output_dir}"
            )
        out_file = gguf_files[0]

    return out_file


def quantize_gguf(
    input_gguf: Path,
    output_dir: Path,
    quant: str,
) -> Path:
    """Quantize an existing GGUF file to the specified quantization level.

    Args:
        input_gguf: Path to the fp16 / bf16 source GGUF.
        output_dir: Where to write the quantized GGUF.
        quant:      Quantization type, e.g. 'Q4_K_M'.

    Returns:
        Path to the quantized .gguf file.
    """
    quantize = _find_quantize_bin()
    if not quantize:
        raise ConversionError(
            "llama-quantize binary not found. "
            "Build llama.cpp: https://github.com/ggml-org/llama.cpp#build"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_gguf.stem.replace("-f16", "").replace("-bf16", "")
    out_file = output_dir / f"{stem}-{quant}.gguf"

    _run([str(quantize), str(input_gguf), str(out_file), quant], "GGUF quantization")
    return out_file


def convert_and_quantize(
    model_dir: Path,
    output_dir: Path,
    quant: str,
    keep_fp16: bool = False,
) -> Path:
    """Full pipeline: SafeTensors → fp16 GGUF → quantized GGUF.

    Args:
        model_dir:  HF model directory.
        output_dir: Output directory.
        quant:      Target quantization (e.g. 'Q4_K_M').
        keep_fp16:  Keep the intermediate fp16 GGUF.

    Returns:
        Path to the final quantized GGUF.
    """
    console.print(f"[bold]Converting {model_dir.name} → {quant} GGUF...[/bold]")
    fp16 = convert_to_gguf(model_dir, output_dir, outtype="f16")
    console.print(f"[green]✓ fp16 GGUF: {fp16}[/green]")

    quantized = quantize_gguf(fp16, output_dir, quant)
    console.print(f"[green]✓ Quantized GGUF: {quantized}[/green]")

    if not keep_fp16 and quant.lower() not in ("f16", "fp16"):
        fp16.unlink(missing_ok=True)

    return quantized
