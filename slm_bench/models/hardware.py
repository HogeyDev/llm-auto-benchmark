"""
Hardware and runtime environment detection.
"""

import platform
import psutil
import subprocess
import re
from typing import Optional
from .schema import HardwareInfo


def get_cpu_brand_name() -> str:
    """Retrieve human-friendly CPU model/brand string across OS platforms."""
    # 1. Windows Registry (exact CPU model name)
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val and str(val).strip():
                return str(val).strip()
        except Exception:
            pass
    # 2. Linux /proc/cpuinfo (works on bare metal, VMs, WSL)
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name") or "Hardware" in line:
                        val = line.split(":", 1)[1].strip()
                        if val:
                            return val
        except Exception:
            pass
    # 3. macOS sysctl
    elif platform.system() == "Darwin":
        try:
            val = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if val:
                return val
        except Exception:
            pass

    return platform.processor() or platform.machine()


def is_wsl_environment() -> bool:
    """Check if the current process is running inside WSL."""
    if platform.system() == "Linux":
        try:
            with open("/proc/version", "r", encoding="utf-8") as f:
                ver = f.read().lower()
                return "microsoft" in ver or "wsl" in ver
        except Exception:
            pass
    return False


def is_endpoint_wsl_relayed(endpoint_url: str = "") -> bool:
    """Check if a local endpoint is hosted inside WSL2 relayed by Windows."""
    if platform.system() == "Windows" and any(h in endpoint_url for h in ("localhost", "127.0.0.1", "0.0.0.0")):
        try:
            port_match = re.search(r":(\d+)", endpoint_url)
            target_port = int(port_match.group(1)) if port_match else 8000
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr and conn.laddr.port == target_port and conn.status == psutil.CONN_LISTEN:
                    if conn.pid:
                        proc = psutil.Process(conn.pid)
                        if "wsl" in proc.name().lower():
                            return True
            wsl_procs = [
                p.info.get("name") or ""
                for p in psutil.process_iter(["name"])
                if any(k in (p.info.get("name") or "").lower() for k in ("wslrelay", "vmmem"))
            ]
            if wsl_procs:
                return True
        except Exception:
            pass
    return False


def generate_auto_system_notes(
    hw: HardwareInfo,
    backend_name: str = "",
    endpoint_url: str = "",
    server_runtime: Optional[str] = None,
    user_notes: Optional[str] = None,
) -> str:
    """
    Automatically compose comprehensive system & execution notes based on detected hardware,
    GPU specifications, OS/WSL environment, and backend server telemetry.
    If the user has provided explicit notes, those are respected.
    """
    if user_notes and user_notes.strip():
        return user_notes.strip()

    parts = []

    # 1. Server runtime and environment
    if server_runtime:
        parts.append(server_runtime)
    elif backend_name:
        b_label = backend_name.upper() if backend_name.lower() in ("vllm", "hf") else backend_name.capitalize()
        if is_endpoint_wsl_relayed(endpoint_url) or is_wsl_environment():
            parts.append(f"{b_label} in WSL2")
        else:
            parts.append(f"{b_label}")

    # 2. GPU Model and VRAM
    if hw.gpu_name:
        gpu_str = hw.gpu_name.replace("NVIDIA GeForce ", "GeForce ").replace("NVIDIA ", "")
        if hw.gpu_vram_gb:
            vram_round = round(hw.gpu_vram_gb)
            gpu_str = f"{gpu_str} ({vram_round}GB VRAM)"
        parts.append(f"on {gpu_str}")
    elif hw.cpu:
        parts.append(f"on CPU ({hw.cpu})")

    # 3. Host OS if not WSL
    if not (is_endpoint_wsl_relayed(endpoint_url) or is_wsl_environment()):
        os_short = "Windows" if "Windows" in hw.os else ("Linux" if "Linux" in hw.os else "macOS")
        parts.append(f"({os_short})")

    return " ".join(parts).strip()


def get_hardware_info(model_ram_gb: Optional[float] = None) -> HardwareInfo:
    """Collect host hardware and system specifications."""
    # OS & CPU
    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    cpu_name = get_cpu_brand_name()
    cpu_cores = psutil.cpu_count(logical=True) or 1
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    gpu_name: Optional[str] = None
    gpu_count = 0
    gpu_vram_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None

    # First attempt detection using nvidia-smi CSV query (exact, untruncated name and total memory)
    try:
        csv_proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if csv_proc.returncode == 0 and csv_proc.stdout.strip():
            lines = [l.strip() for l in csv_proc.stdout.strip().splitlines() if l.strip()]
            if lines:
                gpu_count = len(lines)
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 1 and parts[0]:
                    gpu_name = parts[0]
                if len(parts) >= 2 and parts[1]:
                    try:
                        gpu_vram_gb = round(float(parts[1]) / 1024, 2)
                    except ValueError:
                        pass
                if len(parts) >= 3 and parts[2]:
                    driver_version = parts[2]
    except Exception:
        pass

    # Extract CUDA version via nvidia-smi if not set
    if not cuda_version:
        try:
            smi_proc = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False
            )
            if smi_proc.returncode == 0:
                cuda_match = re.search(r"CUDA (?:Version|UMD Version):\s*([0-9.]+)", smi_proc.stdout, re.IGNORECASE)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
        except Exception:
            pass

    # Attempt detection using torch if available
    try:
        import torch
        if torch.cuda.is_available():
            if not gpu_count:
                gpu_count = torch.cuda.device_count()
            if not gpu_name:
                gpu_name = torch.cuda.get_device_name(0)
            if gpu_vram_gb is None:
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_vram_gb = round(total_bytes / (1024 ** 3), 2)
            if not cuda_version:
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
        model_ram_gb=model_ram_gb,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_vram_gb=gpu_vram_gb,
        cuda_version=cuda_version,
        driver_version=driver_version,
    )
