import torch
import torch.nn as nn

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


def contar_parametros(modelo: nn.Module) -> int:
    return sum(parametro.numel() for parametro in modelo.parameters() if parametro.requires_grad)
