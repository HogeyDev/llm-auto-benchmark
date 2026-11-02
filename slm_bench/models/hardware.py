"""
Hardware and runtime environment detection.
"""

import platform
import psutil
import subprocess
import re
from typing import Optional
from .schema import HardwareInfo


def get_hardware_info() -> HardwareInfo:
    """Collect host hardware and system specifications."""
    # OS & CPU
    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    cpu_name = platform.processor() or platform.machine()
    cpu_cores = psutil.cpu_count(logical=True) or 1
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    gpu_name: Optional[str] = None
    gpu_count = 0
    gpu_vram_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None

    # First attempt detection using torch if available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_vram_gb = round(total_bytes / (1024 ** 3), 2)
            cuda_version = torch.version.cuda
    except Exception:
        pass

    # If torch is not initialized or torch.cuda has no driver, fallback to nvidia-smi
    if not gpu_name or not driver_version:
        try:
            smi_proc = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False
            )
            if smi_proc.returncode == 0:
                output = smi_proc.stdout
                # Extract Driver Version & CUDA Version
                driver_match = re.search(r"Driver-Model.*?Driver Version:\s*([0-9.]+)|Driver Version:\s*([0-9.]+)|KMD Version:\s*([0-9.]+)", output, re.IGNORECASE)
                if driver_match:
                    driver_version = next(g for g in driver_match.groups() if g is not None)
                
                cuda_match = re.search(r"CUDA (?:Version|UMD Version):\s*([0-9.]+)", output, re.IGNORECASE)
                if cuda_match and not cuda_version:
                    cuda_version = cuda_match.group(1)

                # Extract GPU Name if not yet found
                if not gpu_name:
                    name_match = re.search(r"\|\s*\d+\s+([A-Za-z0-9\s]+(?:RTX|GeForce|Tesla|A100|H100|GTX)[^|]*?)\s+(?:WDDM|Off|On)", output)
                    if not name_match:
                        name_match = re.search(r"(?:NVIDIA\s+GeForce\s+RTX\s+[0-9A-Za-z\s]+|NVIDIA\s+[A-Za-z0-9\s]+)", output)
                    if name_match:
                        gpu_name = name_match.group(1 if name_match.lastindex else 0).strip()
                        gpu_count = max(gpu_count, 1)

                # Extract Memory if not yet found
                if gpu_vram_gb is None:
                    mem_match = re.search(r"([0-9]+)\s*MiB\s*/\s*([0-9]+)\s*MiB", output)
                    if mem_match:
                        total_mib = float(mem_match.group(2))
                        gpu_vram_gb = round(total_mib / 1024, 2)
        except Exception:
            pass

    return HardwareInfo(
        os=os_name,
        cpu=cpu_name,
        cpu_cores=cpu_cores,
        total_ram_gb=total_ram_gb,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_vram_gb=gpu_vram_gb,
        cuda_version=cuda_version,
        driver_version=driver_version,
    )
