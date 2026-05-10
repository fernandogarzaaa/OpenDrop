"""Tests for opendrop.core.hardware."""

from __future__ import annotations

import platform
from unittest.mock import MagicMock, patch

import pytest

from opendrop.core.hardware import (
    GPUKind,
    HardwareProfile,
    detect_hardware,
    _probe_apple_silicon,
    _probe_nvidia,
    _probe_cpu_features,
)


class TestHardwareProfile:
    def test_effective_memory_unified(self):
        p = HardwareProfile()
        p.gpu.kind = GPUKind.APPLE_SILICON
        p.gpu.unified = True
        p.ram_mb = 128 * 1024
        assert p.effective_memory_mb == int(128 * 1024 * 0.75)

    def test_effective_memory_vram(self):
        p = HardwareProfile()
        p.usable_vram_mb = 24 * 1024
        assert p.effective_memory_mb == 24 * 1024

    def test_effective_memory_cpu_only(self):
        p = HardwareProfile()
        p.free_ram_mb = 16 * 1024
        assert p.effective_memory_mb == int(16 * 1024 * 0.60)

    def test_backend_priority_apple(self):
        p = HardwareProfile()
        p.gpu.kind = GPUKind.APPLE_SILICON
        assert p.backend_priority[0] == "metal"

    def test_backend_priority_nvidia(self):
        p = HardwareProfile()
        p.gpu.kind = GPUKind.NVIDIA
        assert p.backend_priority[0] == "cuda"

    def test_backend_priority_cpu(self):
        p = HardwareProfile()
        p.gpu.kind = GPUKind.NONE
        assert p.backend_priority == ["cpu"]

    def test_summary_contains_basics(self):
        p = HardwareProfile()
        p.os_name = "TestOS"
        p.ram_mb = 32768
        p.free_ram_mb = 20000
        p.cpu_physical_cores = 8
        p.cpu_cores = 16
        p.cpu_arch = "x86_64"
        summary = p.summary()
        assert "TestOS" in summary
        assert "32.0 GB" in summary


class TestProbes:
    @patch("opendrop.core.hardware.platform.system", return_value="Darwin")
    @patch("opendrop.core.hardware._run", return_value="Apple M3 Max")
    def test_probe_apple_silicon(self, mock_run, mock_sys):
        result = _probe_apple_silicon()
        assert result is not None
        assert result.kind == GPUKind.APPLE_SILICON
        assert result.unified is True

    @patch("opendrop.core.hardware.platform.system", return_value="Linux")
    def test_probe_apple_silicon_linux(self, mock_sys):
        result = _probe_apple_silicon()
        assert result is None

    @patch("opendrop.core.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch(
        "opendrop.core.hardware._run",
        return_value="NVIDIA GeForce RTX 4090, 24576",
    )
    def test_probe_nvidia(self, mock_run, mock_which):
        result = _probe_nvidia()
        assert result is not None
        assert result.kind == GPUKind.NVIDIA
        assert result.vram_mb == 24576

    @patch("opendrop.core.hardware.shutil.which", return_value=None)
    def test_probe_nvidia_not_found(self, mock_which):
        result = _probe_nvidia()
        assert result is None


class TestDetectHardware:
    @patch("opendrop.core.hardware._probe_apple_silicon", return_value=None)
    @patch("opendrop.core.hardware._probe_nvidia", return_value=None)
    @patch("opendrop.core.hardware._probe_amd", return_value=None)
    @patch("opendrop.core.hardware._probe_intel_arc", return_value=None)
    def test_detect_hardware_returns_profile(self, *mocks):
        profile = detect_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.cpu_cores >= 1
        assert profile.ram_mb > 0

    def test_detect_hardware_smoke(self):
        """Smoke test — just verify it doesn't crash."""
        profile = detect_hardware()
        assert profile is not None
        _ = profile.summary()
