"""Tests for opendrop.core.quantizer."""

from __future__ import annotations

import pytest

from opendrop.core.hardware import GPUInfo, GPUKind, HardwareProfile
from opendrop.core.quantizer import (
    QUANT_BY_NAME,
    QuantSpec,
    estimate_size_mb,
    quant_fits,
    quant_summary,
    select_quantization,
)


def _profile(effective_mb: int) -> HardwareProfile:
    """Helper: profile with a given effective memory budget."""
    p = HardwareProfile()
    p.usable_vram_mb = effective_mb
    p.gpu = GPUInfo(kind=GPUKind.NVIDIA, vram_mb=effective_mb)
    return p


class TestSelectQuantization:
    def test_large_memory_selects_fp16(self):
        profile = _profile(200_000)  # 200 GB
        spec = select_quantization(profile, params_b=8.0)
        assert spec.name == "fp16"

    def test_medium_memory_avoids_fp16(self):
        profile = _profile(16_000)  # 16 GB
        spec = select_quantization(profile, params_b=13.0)
        # fp16 would need ~26 GB — should pick something smaller
        assert spec.bits < 16.0

    def test_small_memory_forces_compression(self):
        profile = _profile(8_000)  # 8 GB
        spec = select_quantization(profile, params_b=13.0)
        # 13B fp16 ~= 26 GB — must compress heavily
        assert spec.bits <= 4.8

    def test_preferred_quant_respected_when_fits(self):
        profile = _profile(200_000)
        spec = select_quantization(profile, params_b=7.0, preferred="Q4_K_M")
        assert spec.name == "Q4_K_M"

    def test_preferred_quant_overridden_when_doesnt_fit(self):
        # 0.5 GB budget — fp16 can never fit for a 7B model
        profile = _profile(500)
        spec = select_quantization(profile, params_b=7.0, preferred="fp16")
        assert spec.bits < 16.0

    def test_returns_quant_spec(self):
        profile = _profile(32_000)
        spec = select_quantization(profile, params_b=8.0)
        assert isinstance(spec, QuantSpec)
        assert spec.name in QUANT_BY_NAME


class TestQuantFits:
    def test_fits_returns_true(self):
        profile = _profile(200_000)
        assert quant_fits(profile, 7.0, "fp16") is True

    def test_fits_returns_false(self):
        profile = _profile(1_000)  # 1 GB
        assert quant_fits(profile, 7.0, "fp16") is False

    def test_unknown_quant_returns_false(self):
        profile = _profile(200_000)
        assert quant_fits(profile, 7.0, "INVALID_QUANT") is False


class TestEstimateSize:
    def test_fp16_8b(self):
        size = estimate_size_mb(8.0, "fp16")
        # 8B × 2 bytes / (1024^2) ≈ 15258 MB
        expected = 8.0 * 1e9 * 2 / (1024 * 1024)
        assert abs(size - expected) < 1

    def test_q4_km_8b(self):
        size = estimate_size_mb(8.0, "Q4_K_M")
        # 8B × 0.6 bytes (4.8 bits) ≈ 4800 MB
        assert 4000 < size < 6000

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            estimate_size_mb(7.0, "NOT_A_QUANT")


class TestQuantSummary:
    def test_summary_contains_quants(self):
        profile = _profile(32_000)
        summary = quant_summary(profile, 8.0)
        assert "Q4_K_M" in summary
        assert "fp16" in summary
        assert "✓" in summary or "✗" in summary
