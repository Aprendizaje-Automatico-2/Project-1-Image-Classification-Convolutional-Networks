from pathlib import Path

import numpy as np
import torch
import pandas as pd
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


def barras_experimentos(tabla: pd.DataFrame, referencia: str, titulo: str) -> Figure:
    figura = Figure(figsize=(12, 0.4 * len(tabla) + 1.5))
    for eje, columna, nombre in zip(figura.subplots(1, 2, sharey=True), ("acc_val", "perdida_val"),
                                    ("Accuracy de validación", "Pérdida de validación")):
        colores = ["tab:red" if experimento == referencia else "tab:blue" for experimento in tabla["nombre"]]
        eje.barh(tabla["nombre"], tabla[columna], color=colores)
        eje.axvline(tabla.loc[tabla["nombre"] == referencia, columna].item(), color="tab:red", linestyle="--")
        for posicion, valor in enumerate(tabla[columna]):
            eje.text(valor, posicion, f" {valor:.3f}", va="center", fontsize=8)
        eje.set_title(nombre)
        eje.invert_yaxis()
        eje.grid(True, axis="x", alpha=0.3)
    figura.suptitle(titulo)
    return figura


def accuracy_frente_a_parametros(
    tabla: pd.DataFrame, titulo: str, columna: str = "acc_val", etiqueta: str = "Accuracy de validación"
) -> Figure:
    figura = Figure(figsize=(8, 5))
    eje = figura.subplots()
    errores = None
    if "ic_inferior" in tabla:
        errores = [tabla[columna] - tabla["ic_inferior"], tabla["ic_superior"] - tabla[columna]]
    eje.errorbar(tabla["parametros"], tabla[columna], yerr=errores, fmt="o", markersize=8, capsize=4, color="tab:blue")
    for _, fila in tabla.iterrows():
        eje.annotate(fila["nombre"], (fila["parametros"], fila[columna]), textcoords="offset points",
                     xytext=(6, 4), fontsize=8)
    eje.set_xscale("log")
    eje.set_xlabel("Parámetros (escala logarítmica)")
    eje.set_ylabel(etiqueta)
    eje.set_title(titulo)
    eje.grid(True, alpha=0.3)
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


def lote_de_ejemplo(imagenes: torch.Tensor, etiquetas: torch.Tensor, clases: list[str], columnas: int = 4) -> Figure:
    filas = len(imagenes) // columnas
    figura = Figure(figsize=(3 * columnas, 2.4 * filas))
    for eje, imagen, etiqueta in zip(figura.subplots(filas, columnas).flat, imagenes, etiquetas):
        mostrar_sin_ejes(eje, imagen, clases[etiqueta])
    figura.suptitle("Lote de entrenamiento con data augmentation")
    return figura
