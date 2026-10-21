from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay

from fishcls.aumentado import AUMENTADOS, transformaciones_entrenamiento, transformaciones_evaluacion
from fishcls.config import DESVIACION_IMAGENET, FIGURAS, MEDIA_IMAGENET
from fishcls.evaluacion import Evaluacion

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


def figura_perdida_y_accuracy(titulo: str) -> tuple[Figure, list[tuple]]:
    figura = Figure(figsize=(12, 4.5))
    figura.suptitle(titulo)
    return figura, list(zip(figura.subplots(1, 2), ("perdida", "acc"), ("Pérdida", "Accuracy")))


def formatear_eje(eje, titulo: str, nombre: str) -> None:
    eje.set_title(titulo, fontsize=10)
    eje.set_xlabel("Época")
    eje.set_ylabel(nombre)
    eje.legend(fontsize=8)
    eje.grid(True, alpha=0.3)


def curvas_entrenamiento(historial: dict[str, list[float]], mejor_epoca: int, titulo: str) -> Figure:
    figura, paneles = figura_perdida_y_accuracy(titulo)
    epocas = range(1, len(historial["perdida_train"]) + 1)
    for eje, metrica, nombre in paneles:
        eje.plot(epocas, historial[f"{metrica}_train"], label="Entrenamiento")
        eje.plot(epocas, historial[f"{metrica}_val"], label="Validación")
        eje.axvline(mejor_epoca, color="red", linestyle="--", label=f"Mejor época ({mejor_epoca})")
        formatear_eje(eje, f"Evolución de la {nombre.lower()}", nombre)
    return figura


def matriz_confusion(evaluacion: Evaluacion, titulo: str) -> Figure:
    figura = Figure(figsize=(13, 5.5))
    for eje, normalizar, nombre in zip(figura.subplots(1, 2), (None, "true"), ("Conteos", "Normalizada por clase real")):
        ConfusionMatrixDisplay.from_predictions(
            evaluacion.y_real,
            evaluacion.y_pred,
            labels=range(len(evaluacion.clases)),
            display_labels=evaluacion.clases,
            normalize=normalizar,
            values_format=".2f" if normalizar else "d",
            cmap="Blues",
            colorbar=False,
            xticks_rotation=45,
            ax=eje,
        )
        eje.set_title(nombre)
        eje.set_xlabel("Predicción")
        eje.set_ylabel("Etiqueta real")
    figura.suptitle(f"{titulo} · accuracy {evaluacion.accuracy:.3f} · F1 macro {evaluacion.f1_macro:.3f}")
    figura.tight_layout()
    return figura


def comparar_historiales(historiales: dict[str, dict[str, list[float]]], titulo: str) -> Figure:
    figura, paneles = figura_perdida_y_accuracy(titulo)
    for eje, metrica, nombre in paneles:
        for (experimento, historial), color in zip(historiales.items(), COLORES):
            epocas = range(1, len(historial[f"{metrica}_val"]) + 1)
            eje.plot(epocas, historial[f"{metrica}_train"], color=color, linestyle=":", alpha=0.7)
            eje.plot(epocas, historial[f"{metrica}_val"], color=color, label=experimento)
        formatear_eje(eje, f"{nombre} (continua: validación · punteada: entrenamiento)", nombre)
    return figura


def lote_de_ejemplo(imagenes: torch.Tensor, etiquetas: torch.Tensor, clases: list[str], columnas: int = 4) -> Figure:
    filas = len(imagenes) // columnas
    figura = Figure(figsize=(3 * columnas, 2.4 * filas))
    for eje, imagen, etiqueta in zip(figura.subplots(filas, columnas).flat, imagenes, etiquetas):
        mostrar_sin_ejes(eje, imagen, clases[etiqueta])
    figura.suptitle("Lote de entrenamiento con data augmentation")
    return figura
