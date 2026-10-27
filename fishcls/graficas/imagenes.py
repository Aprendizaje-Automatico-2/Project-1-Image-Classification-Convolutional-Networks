import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from fishcls.datos.aumentado import AUMENTADOS, transformaciones_entrenamiento, transformaciones_evaluacion
from fishcls.graficas.comun import mostrar_sin_ejes


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


def cuadricula_gradcam(
    imagenes: list[torch.Tensor],
    reales: list[int],
    mapas: dict[str, list[tuple[np.ndarray, int]]],
    clases: list[str],
    titulo: str,
) -> Figure:
    figura = Figure(figsize=(2.8 * (len(mapas) + 1), 2.3 * len(imagenes)), layout="constrained")
    ejes = figura.subplots(len(imagenes), len(mapas) + 1, squeeze=False)
    for fila, imagen, real, *celdas in zip(ejes, imagenes, reales, *mapas.values()):
        alto, ancho = imagen.shape[1:]
        mostrar_sin_ejes(fila[0], imagen, f"real: {clases[real]}")
        for eje, modelo, (mapa, prediccion) in zip(fila[1:], mapas, celdas):
            mostrar_sin_ejes(eje, imagen, f"{modelo}: {clases[prediccion]}")
            eje.imshow(mapa, cmap="jet", alpha=0.45, extent=(0, ancho, alto, 0), interpolation="bilinear")
            eje.title.set_color("tab:green" if prediccion == real else "tab:red")
    figura.suptitle(titulo)
    return figura
