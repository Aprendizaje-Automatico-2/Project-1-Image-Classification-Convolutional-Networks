from time import perf_counter

import torch
import torch.nn as nn
from torchvision import models

PREENTRENADOS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "classifier"),
}

CAPAS_GRADCAM = {"resnet18": "layer4", "convnext_tiny": "features"}

ACTIVACIONES = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def bloque_convolucional(entrada: int, salida: int, tamano_kernel: int, activacion: str, batchnorm: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel_size=tamano_kernel, padding=tamano_kernel // 2),
        nn.BatchNorm2d(salida) if batchnorm else nn.Identity(),
        ACTIVACIONES[activacion](),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def bloque_denso(entrada: int, salida: int, activacion: str, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(entrada, salida), ACTIVACIONES[activacion](), nn.Dropout(dropout))


class ConvNet(nn.Module):
    def __init__(
        self,
        num_clases: int,
        tamano: tuple[int, int],
        filtros: tuple[int, ...],
        neuronas_fc: tuple[int, ...],
        tamano_kernel: int,
        activacion: str,
        dropout: float,
        batchnorm: bool,
        pooling_global: bool,
    ):
        super().__init__()
        canales = (3, *filtros)
        self.convoluciones = nn.Sequential(
            *(bloque_convolucional(entrada, salida, tamano_kernel, activacion, batchnorm)
              for entrada, salida in zip(canales, canales[1:])),
            nn.AdaptiveAvgPool2d(1) if pooling_global else nn.Identity(),
            nn.Flatten(),
        )
        with torch.no_grad():
            aplanado = self.convoluciones(torch.zeros(1, 3, *tamano)).shape[1]
        neuronas = (aplanado, *neuronas_fc)
        self.clasificador = nn.Sequential(
            *(bloque_denso(entrada, salida, activacion, dropout) for entrada, salida in zip(neuronas, neuronas[1:])),
            nn.Linear(neuronas[-1], num_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clasificador(self.convoluciones(x))


def modelo_preentrenado(nombre: str, num_clases: int, con_pesos: bool = True) -> nn.Module:
    constructor, pesos, nombre_cabeza = PREENTRENADOS[nombre]
    modelo = constructor(weights=pesos if con_pesos else None)
    cabeza = getattr(modelo, nombre_cabeza)
    if isinstance(cabeza, nn.Linear):
        setattr(modelo, nombre_cabeza, nn.Linear(cabeza.in_features, num_clases))
    else:
        cabeza[-1] = nn.Linear(cabeza[-1].in_features, num_clases)
    return modelo


def parametros_cabeza(modelo: nn.Module, nombre: str) -> list[nn.Parameter]:
    if nombre not in PREENTRENADOS:
        return list(modelo.parameters())
    return list(getattr(modelo, PREENTRENADOS[nombre][2]).parameters())


def congelar_extractor(modelo: nn.Module, nombre: str, congelar: bool) -> None:
    cabeza = {id(parametro) for parametro in parametros_cabeza(modelo, nombre)}
    for parametro in modelo.parameters():
        parametro.requires_grad = id(parametro) in cabeza or not congelar


def capa_gradcam(modelo: nn.Module, nombre: str) -> nn.Module:
    if nombre in CAPAS_GRADCAM:
        return getattr(modelo, CAPAS_GRADCAM[nombre])
    return [capa for capa in modelo.convoluciones if isinstance(capa, nn.Sequential)][-1]


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
