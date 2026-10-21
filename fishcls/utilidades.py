import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def obtener_dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def describir_dispositivo(dispositivo: torch.device) -> str:
    if dispositivo.type == "cuda":
        plataforma = "AMD ROCm" if torch.version.hip else "NVIDIA CUDA"
        return f"{plataforma} · {torch.cuda.get_device_name(dispositivo)}"
    if dispositivo.type == "mps":
        return "Apple Silicon · MPS"
    if dispositivo.type == "xpu":
        return f"Intel XPU · {torch.xpu.get_device_name(dispositivo)}"
    return "CPU"


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def guardar_json(datos: Any, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False))


def cargar_json(ruta: Path) -> Any:
    return json.loads(ruta.read_text())
