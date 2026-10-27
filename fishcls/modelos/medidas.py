from time import perf_counter

import torch
import torch.nn as nn


def contar_parametros(modelo: nn.Module) -> int:
    return sum(parametro.numel() for parametro in modelo.parameters())


@torch.no_grad()
def medir_milisegundos_por_imagen(
    modelo: nn.Module, tamano: tuple[int, int], dispositivo: torch.device, repeticiones: int = 20
) -> float:
    modelo.eval()
    lote = torch.randn(1, 3, *tamano, device=dispositivo)
    for _ in range(5):
        modelo(lote)
    sincronizar(dispositivo)
    comienzo = perf_counter()
    for _ in range(repeticiones):
        modelo(lote)
    sincronizar(dispositivo)
    return 1000 * (perf_counter() - comienzo) / repeticiones


def sincronizar(dispositivo: torch.device) -> None:
    if dispositivo.type in ("cuda", "mps", "xpu"):
        getattr(torch, dispositivo.type).synchronize()
