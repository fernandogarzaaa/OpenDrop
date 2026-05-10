"""Hardware profiler for OpenDrop.

Detects GPU type, VRAM / unified memory, CPU cores, system RAM, and
estimated SSD throughput.  Results are used by the quantization decision
engine to pick the best quantization level automatically.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import psutil


class GPUKind(str, Enum):
    APPLE_SILICON = "apple_silicon"
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL_ARC = "intel_arc"
    NONE = "none"


@dataclass
class GPUInfo:
    kind: GPUKind
    name: str = ""
    vram_mb: int = 0  # 0 = unified / unknown
    unified: bool = False  # Apple unified memory


@dataclass
class HardwareProfile:
    """Complete hardware snapshot for a host machine."""

    gpu: GPUInfo = field(default_factory=lambda: GPUInfo(kind=GPUKind.NONE))
    # Apple unified memory OR dedicated VRAM in MB
    usable_vram_mb: int = 0
    cpu_cores: int = 1
    cpu_physical_cores: int = 1
    cpu_arch: str = ""          # "x86_64", "arm64", …
    has_avx2: bool = False
    has_avx512: bool = False
    has_neon: bool = False      # Apple Silicon / ARM
    ram_mb: int = 0
    free_ram_mb: int = 0
    ssd_est_mb_per_s: int = 0  # rough estimate, 0 = unknown
    os_name: str = ""           # "Darwin", "Linux", "Windows"

    # Derived helpers
    @property
    def effective_memory_mb(self) -> int:
        """The memory budget for model weights."""
        if self.gpu.unified:
            # Use 75 % of total RAM as conservative estimate for unified
            return int(self.ram_mb * 0.75)
        if self.usable_vram_mb > 0:
            return self.usable_vram_mb
        # CPU only — use 60 % of free RAM
        return int(self.free_ram_mb * 0.60)

    @property
    def backend_priority(self) -> list[str]:
        """Ordered list of preferred llama.cpp build flags / backends."""
        if self.gpu.kind == GPUKind.APPLE_SILICON:
            return ["metal", "cpu"]
        if self.gpu.kind == GPUKind.NVIDIA:
            return ["cuda", "cpu"]
        if self.gpu.kind == GPUKind.AMD:
            return ["rocm", "vulkan", "cpu"]
        if self.gpu.kind == GPUKind.INTEL_ARC:
            return ["vulkan", "cpu"]
        return ["cpu"]

    def summary(self) -> str:
        lines = [
            f"OS          : {self.os_name}",
            f"CPU         : {self.cpu_physical_cores}c/{self.cpu_cores}t  arch={self.cpu_arch}"
            + (" AVX2" if self.has_avx2 else "")
            + (" AVX512" if self.has_avx512 else "")
            + (" NEON" if self.has_neon else ""),
            f"RAM         : {self.ram_mb / 1024:.1f} GB total, {self.free_ram_mb / 1024:.1f} GB free",
            f"GPU         : {self.gpu.name or self.gpu.kind.value}",
        ]
        if self.gpu.unified:
            lines.append(f"Unified mem : {self.ram_mb / 1024:.1f} GB (shared)")
        elif self.usable_vram_mb:
            lines.append(f"VRAM        : {self.usable_vram_mb / 1024:.1f} GB")
        lines.append(f"Budget      : {self.effective_memory_mb / 1024:.1f} GB for weights")
        if self.ssd_est_mb_per_s:
            lines.append(f"SSD         : ~{self.ssd_est_mb_per_s} MB/s (est.)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 5) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _probe_apple_silicon() -> Optional[GPUInfo]:
    if platform.system() != "Darwin":
        return None
    cpu_brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if cpu_brand and ("Apple" in cpu_brand or "M1" in cpu_brand or "M2" in cpu_brand
                      or "M3" in cpu_brand or "M4" in cpu_brand):
        name = cpu_brand.split("Apple ")[-1] if "Apple " in cpu_brand else cpu_brand
        return GPUInfo(kind=GPUKind.APPLE_SILICON, name=name, unified=True)
    return None


def _probe_nvidia() -> Optional[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out:
        return None
    first_line = out.splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 2:
        return None
    name = parts[0]
    try:
        vram_mb = int(parts[1])
    except ValueError:
        vram_mb = 0
    return GPUInfo(kind=GPUKind.NVIDIA, name=name, vram_mb=vram_mb)


def _probe_amd() -> Optional[GPUInfo]:
    for tool in ("rocm-smi", "rocminfo"):
        if not shutil.which(tool):
            continue
        if tool == "rocm-smi":
            out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
            if out:
                # Try to parse vram total from CSV
                for line in out.splitlines():
                    m = re.search(r"(\d+)", line)
                    if m:
                        return GPUInfo(kind=GPUKind.AMD, name="AMD GPU",
                                       vram_mb=int(m.group(1)) // (1024 * 1024))
        return GPUInfo(kind=GPUKind.AMD, name="AMD GPU (ROCm)")
    return None


def _probe_intel_arc() -> Optional[GPUInfo]:
    # Intel GPU detection via clinfo or sycl-ls
    if shutil.which("sycl-ls"):
        out = _run(["sycl-ls"])
        if out and "Intel" in out:
            return GPUInfo(kind=GPUKind.INTEL_ARC, name="Intel Arc GPU")
    return None


def _probe_cpu_features() -> tuple[bool, bool, bool]:
    """Returns (has_avx2, has_avx512, has_neon)."""
    arch = platform.machine().lower()
    # ARM / Apple Silicon
    if arch in ("arm64", "aarch64"):
        return False, False, True

    # x86 — read /proc/cpuinfo on Linux, sysctl on macOS
    flags_str = ""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                flags_str = f.read()
        except OSError:
            pass
    elif platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.features",
                    "machdep.cpu.leaf7_features"])
        flags_str = out or ""

    flags_str = flags_str.lower()
    avx2 = "avx2" in flags_str
    avx512 = "avx512" in flags_str or "avx-512" in flags_str
    return avx2, avx512, False


def _estimate_ssd_speed() -> int:
    """Very rough SSD speed estimate in MB/s.

    Returns 0 when it cannot be determined cheaply.
    On macOS, disk type can be inferred from the 'solidstate' property.
    """
    if platform.system() == "Darwin":
        out = _run(["system_profiler", "SPStorageDataType"], timeout=10)
        if out and ("SSD" in out or "Solid State" in out):
            # Modern Apple SSD — conservatively 2000 MB/s
            return 2000
    # On Linux we can check /sys, but skip for speed — caller can override
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareProfile:
    """Run hardware probes and return a :class:`HardwareProfile`."""
    profile = HardwareProfile()
    profile.os_name = platform.system()
    profile.cpu_arch = platform.machine()
    profile.cpu_cores = psutil.cpu_count(logical=True) or 1
    profile.cpu_physical_cores = psutil.cpu_count(logical=False) or 1

    vm = psutil.virtual_memory()
    profile.ram_mb = vm.total // (1024 * 1024)
    profile.free_ram_mb = vm.available // (1024 * 1024)

    avx2, avx512, neon = _probe_cpu_features()
    profile.has_avx2 = avx2
    profile.has_avx512 = avx512
    profile.has_neon = neon

    # GPU probe — first match wins
    gpu = (
        _probe_apple_silicon()
        or _probe_nvidia()
        or _probe_amd()
        or _probe_intel_arc()
        or GPUInfo(kind=GPUKind.NONE, name="No GPU / CPU only")
    )
    profile.gpu = gpu

    if gpu.unified:
        profile.usable_vram_mb = profile.ram_mb
    elif gpu.vram_mb:
        profile.usable_vram_mb = gpu.vram_mb

    profile.ssd_est_mb_per_s = _estimate_ssd_speed()
    return profile
