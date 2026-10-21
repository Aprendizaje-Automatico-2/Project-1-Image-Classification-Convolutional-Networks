from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from fishcls.aumentado import AUMENTADOS, transformaciones_entrenamiento, transformaciones_evaluacion
from fishcls.config import DESVIACION_IMAGENET, FIGURAS, MEDIA_IMAGENET


def desnormalizar(imagen: torch.Tensor) -> np.ndarray:
    media = torch.tensor(MEDIA_IMAGENET).view(3, 1, 1)
    desviacion = torch.tensor(DESVIACION_IMAGENET).view(3, 1, 1)
    return (imagen * desviacion + media).clamp(0, 1).permute(1, 2, 0).numpy()


def guardar_figura(figura: Figure, nombre: str) -> Path:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / nombre
    figura.savefig(ruta, dpi=120, bbox_inches="tight")
    return ruta


def mostrar_sin_ejes(eje, imagen: torch.Tensor, titulo: str = "") -> None:
    eje.imshow(desnormalizar(imagen))
    eje.set_title(titulo, fontsize=9)
    eje.set_xticks([])
    eje.set_yticks([])


def galeria_aumentado(imagen: Image.Image, tamano: tuple[int, int], muestras: int = 5) -> Figure:
    aumentados = [nombre for nombre in AUMENTADOS if nombre != "ninguno"]
    figura = Figure(figsize=(2.4 * (muestras + 1), 2 * len(aumentados)))
    ejes = figura.subplots(len(aumentados), muestras + 1)
    original = transformaciones_evaluacion(tamano)(imagen)
    for fila, aumentado in zip(ejes, aumentados):
        transformar = transformaciones_entrenamiento(tamano, aumentado)
        mostrar_sin_ejes(fila[0], original, "original")
        fila[0].set_ylabel(aumentado, fontsize=11)
        for eje in fila[1:]:
            mostrar_sin_ejes(eje, transformar(imagen))
    figura.suptitle("Data augmentation por familia de transformaciones")
    return figura


def lote_de_ejemplo(imagenes: torch.Tensor, etiquetas: torch.Tensor, clases: list[str], columnas: int = 4) -> Figure:
    filas = len(imagenes) // columnas
    figura = Figure(figsize=(3 * columnas, 2.4 * filas))
    for eje, imagen, etiqueta in zip(figura.subplots(filas, columnas).flat, imagenes, etiquetas):
        mostrar_sin_ejes(eje, imagen, clases[etiqueta])
    figura.suptitle("Lote de entrenamiento con data augmentation")
    return figura
