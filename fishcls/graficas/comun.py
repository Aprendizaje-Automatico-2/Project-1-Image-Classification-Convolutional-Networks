from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.figure import Figure

from fishcls.config import DESVIACION_IMAGENET, FIGURAS, MEDIA_IMAGENET

COLORES = colormaps["tab10"].colors


def desnormalizar(imagen: torch.Tensor) -> np.ndarray:
    media = torch.tensor(MEDIA_IMAGENET).view(3, 1, 1)
    desviacion = torch.tensor(DESVIACION_IMAGENET).view(3, 1, 1)
    return (imagen * desviacion + media).clamp(0, 1).permute(1, 2, 0).numpy()


def guardar_figura(figura: Figure, nombre: str, carpeta: Path = FIGURAS) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    figura.savefig(ruta, dpi=120, bbox_inches="tight")
    return ruta


def mostrar_sin_ejes(eje, imagen: torch.Tensor, titulo: str = "") -> None:
    eje.imshow(desnormalizar(imagen))
    eje.set_title(titulo, fontsize=9)
    eje.set_xticks([])
    eje.set_yticks([])
