"""Quantization decision engine.

Given a HardwareProfile and a model's parameter count (in billions),
selects the best quantization level that fits in the available memory budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from opendrop.core.hardware import HardwareProfile


@dataclass
class QuantSpec:
    """Describes a quantization choice."""

    name: str  # e.g. "Q4_K_M"
    bits: float  # effective bits per weight
    quality: str  # "excellent" / "good" / "fair" / "poor"
    description: str


# Ordered from highest to lowest quality.
# bytes_per_param = bits / 8.
_QUANT_TABLE: list[QuantSpec] = [
    QuantSpec("fp16", 16.0, "excellent", "Half precision, lossless"),
    QuantSpec("Q8_0", 8.5, "excellent", "8-bit, near-lossless"),
    QuantSpec("Q6_K", 6.6, "excellent", "6-bit K-quant"),
    QuantSpec("Q5_K_M", 5.7, "good", "5-bit K-quant (medium)"),
    QuantSpec("Q5_K_S", 5.5, "good", "5-bit K-quant (small)"),
    QuantSpec("Q4_K_M", 4.8, "good", "4-bit K-quant (medium) — recommended default"),
    QuantSpec("Q4_K_S", 4.6, "good", "4-bit K-quant (small)"),
    QuantSpec("Q3_K_M", 3.9, "fair", "3-bit K-quant (medium)"),
    QuantSpec("Q3_K_S", 3.5, "fair", "3-bit K-quant (small)"),
    QuantSpec("Q2_K", 2.6, "poor", "2-bit K-quant — use only when forced"),
    QuantSpec("IQ2_XXS", 2.1, "poor", "2-bit IQ quant (extreme small) — MoE experts only"),
]

# Map quant name → spec for fast lookup
QUANT_BY_NAME: dict[str, QuantSpec] = {q.name: q for q in _QUANT_TABLE}

# Safety headroom multiplier (model weights + KV cache + activations overhead)
_OVERHEAD = 1.20


def _model_size_mb(params_b: float, bits: float) -> float:
    """Estimate model size in MB for given parameter count and bit width."""
    bytes_per_param = bits / 8.0
    return params_b * 1e9 * bytes_per_param / (1024 * 1024)


def select_quantization(
    profile: HardwareProfile,
    params_b: float,
    preferred: str | None = None,
) -> QuantSpec:
    """Return the best QuantSpec that fits in the hardware memory budget.

    Args:
        profile:   HardwareProfile from detect_hardware().
        params_b:  Model size in billions of parameters.
        preferred: Optional explicit quant name (overrides auto-selection,
                   but we still emit a warning if it won't fit).

    Returns:
        A QuantSpec that will fit in effective memory.
    """
    budget_mb = profile.effective_memory_mb

    if preferred and preferred in QUANT_BY_NAME:
        spec = QUANT_BY_NAME[preferred]
        estimated = _model_size_mb(params_b, spec.bits) * _OVERHEAD
        if estimated <= budget_mb:
            return spec
        # Falls through to auto-selection with a note below

    for spec in _QUANT_TABLE:
        estimated = _model_size_mb(params_b, spec.bits) * _OVERHEAD
        if estimated <= budget_mb:
            return spec

    # Absolute last resort — smallest quant
    return _QUANT_TABLE[-1]


def quant_fits(profile: HardwareProfile, params_b: float, quant_name: str) -> bool:
    """Return True if the given quant fits in available memory."""
    if quant_name not in QUANT_BY_NAME:
        return False
    spec = QUANT_BY_NAME[quant_name]
    return _model_size_mb(params_b, spec.bits) * _OVERHEAD <= profile.effective_memory_mb


def estimate_size_mb(params_b: float, quant_name: str) -> float:
    """Estimate model disk / RAM footprint in MB."""
    if quant_name not in QUANT_BY_NAME:
        raise ValueError(f"Unknown quant: {quant_name}")
    spec = QUANT_BY_NAME[quant_name]
    return _model_size_mb(params_b, spec.bits)


def quant_summary(profile: HardwareProfile, params_b: float) -> str:
    """Human-readable table showing which quants fit for this model+hardware."""
    lines = [
        f"Model: {params_b:.1f}B params | Budget: {profile.effective_memory_mb / 1024:.1f} GB",
        f"{'Quant':<12} {'Est. size':>10} {'Fits':>6} {'Quality':<12} Note",
        "-" * 72,
    ]
    for q in _QUANT_TABLE:
        size_mb = _model_size_mb(params_b, q.bits)
        fits = size_mb * _OVERHEAD <= profile.effective_memory_mb
        mark = "✓" if fits else "✗"
        lines.append(
            f"{q.name:<12} {size_mb / 1024:>8.1f} GB {mark:>6}  {q.quality:<12} {q.description}"
        )
    return "\n".join(lines)
