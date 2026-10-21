import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder

from fishcls.aumentado import transformaciones_entrenamiento, transformaciones_evaluacion
from fishcls.config import (
    ANCHO_MINIMO_KODAK,
    DATOS_ORIGINALES,
    DATOS_PREPARADOS,
    EXTENSIONES_IMAGEN,
    INTENTOS_PARTICION,
    LADO_HUELLA,
    NUM_PROCESOS,
    PARTICION,
    PARTICIONES,
    PROPORCION_TEST,
    PROPORCION_VALIDACION,
    RAIZ,
    SEMILLA,
    TAMANO_PREPARADO,
    UMBRAL_CASI_DUPLICADO,
)
from fishcls.utilidades import guardar_json


class Datos(NamedTuple):
    entrenamiento: DataLoader
    validacion: DataLoader
    test: DataLoader
    clases: list[str]


def listar_imagenes(carpeta: Path) -> list[tuple[Path, str]]:
    return [
        (ruta, clase.name)
        for clase in sorted(carpeta.iterdir())
        if clase.is_dir()
        for ruta in sorted(clase.iterdir())
        if ruta.suffix.lower() in EXTENSIONES_IMAGEN
    ]


def abrir_imagen(ruta: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")


def huella_perceptual(imagen: Image.Image) -> np.ndarray:
    grises = np.asarray(imagen.convert("L").resize((LADO_HUELLA, LADO_HUELLA)), dtype=np.float32)
    return (grises > grises.mean()).ravel()


def agrupar_casi_duplicados(huellas: np.ndarray, clases: np.ndarray) -> np.ndarray:
    distancias = (huellas[:, None, :] != huellas[None, :, :]).sum(axis=2)
    misma_clase = clases[:, None] == clases[None, :]
    _, grupos = connected_components((distancias <= UMBRAL_CASI_DUPLICADO) & misma_clase, directed=False)
    return grupos


def separar_fraccion(
    indices: np.ndarray, etiquetas: np.ndarray, grupos: np.ndarray, fraccion: float, semilla: int
) -> tuple[np.ndarray, np.ndarray]:
    divisor = StratifiedGroupKFold(n_splits=round(1 / fraccion), shuffle=True, random_state=semilla)
    pliegues = list(divisor.split(indices, etiquetas[indices], grupos[indices]))
    resto, separados = pliegues[semilla % len(pliegues)]
    return indices[resto], indices[separados]


def particion_por_grupos(etiquetas: np.ndarray, grupos: np.ndarray, semilla: int) -> np.ndarray:
    todos = np.arange(len(etiquetas))
    resto, test = separar_fraccion(todos, etiquetas, grupos, PROPORCION_TEST, semilla)
    fraccion_validacion = PROPORCION_VALIDACION / (1 - PROPORCION_TEST)
    entrenamiento, validacion = separar_fraccion(resto, etiquetas, grupos, fraccion_validacion, semilla)
    particion = np.empty(len(etiquetas), dtype=object)
    for indices, nombre in zip((entrenamiento, validacion, test), PARTICIONES):
        particion[indices] = nombre
    return particion


def desequilibrio(particion: np.ndarray, etiquetas: np.ndarray) -> float:
    fracciones = (1 - PROPORCION_VALIDACION - PROPORCION_TEST, PROPORCION_VALIDACION, PROPORCION_TEST)
    return sum(
        abs(np.sum(particion[etiquetas == clase] == nombre) - fraccion * np.sum(etiquetas == clase))
        for clase in np.unique(etiquetas)
        for nombre, fraccion in zip(PARTICIONES, fracciones)
    )


def dividir_por_grupos(etiquetas: np.ndarray, grupos: np.ndarray, semilla: int = SEMILLA) -> np.ndarray:
    candidatas = (particion_por_grupos(etiquetas, grupos, semilla + intento) for intento in range(INTENTOS_PARTICION))
    return min(candidatas, key=lambda particion: desequilibrio(particion, etiquetas))


def reducir(ruta: Path) -> tuple[Image.Image, int]:
    imagen = abrir_imagen(ruta)
    alto, ancho = TAMANO_PREPARADO
    return imagen.resize((ancho, alto), Image.Resampling.LANCZOS), imagen.width


def preparar_dataset() -> list[dict]:
    rutas, clases = zip(*listar_imagenes(DATOS_ORIGINALES))
    reducidas, anchos_originales = zip(*(reducir(ruta) for ruta in rutas))
    clases = np.array(clases)
    grupos = agrupar_casi_duplicados(np.stack([huella_perceptual(imagen) for imagen in reducidas]), clases)
    particion = dividir_por_grupos(clases, grupos)

    shutil.rmtree(DATOS_PREPARADOS, ignore_errors=True)
    registros = []
    for ruta, clase, imagen, ancho, grupo, nombre_particion in zip(
        rutas, clases, reducidas, anchos_originales, grupos, particion
    ):
        destino = DATOS_PREPARADOS / nombre_particion / clase / f"{ruta.stem}.jpg"
        destino.parent.mkdir(parents=True, exist_ok=True)
        imagen.save(destino, quality=95)
        registros.append(
            {
                "original": str(ruta.relative_to(RAIZ)),
                "preparada": str(destino.relative_to(RAIZ)),
                "clase": str(clase),
                "grupo": int(grupo),
                "particion": nombre_particion,
                "ancho_original": ancho,
            }
        )
    guardar_json(registros, PARTICION)
    return registros


def resumir_particion(registros: list[dict]) -> pd.DataFrame:
    tabla = pd.DataFrame(registros).assign(kodak=lambda t: t["ancho_original"] >= ANCHO_MINIMO_KODAK)
    por_clase = tabla.groupby("clase")
    resumen = pd.crosstab(tabla["clase"], tabla["particion"])[list(PARTICIONES)]
    resumen["total"] = resumen.sum(axis=1)
    resumen["grupos"] = por_clase["grupo"].nunique()
    resumen["% kodak"] = por_clase["kodak"].mean().mul(100).round().astype(int)
    return resumen


def crear_loader(dataset: Dataset, tamano_lote: int, barajar: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=tamano_lote,
        shuffle=barajar,
        num_workers=NUM_PROCESOS,
        persistent_workers=NUM_PROCESOS > 0,
    )


def crear_dataloaders(
    tamano: tuple[int, int], aumentado: str = "completo", repeticiones: int = 4, tamano_lote: int = 32
) -> Datos:
    entrenamiento = ImageFolder(
        DATOS_PREPARADOS / "entrenamiento", transform=transformaciones_entrenamiento(tamano, aumentado)
    )
    validacion = ImageFolder(DATOS_PREPARADOS / "validacion", transform=transformaciones_evaluacion(tamano))
    test = ImageFolder(DATOS_PREPARADOS / "test", transform=transformaciones_evaluacion(tamano))
    return Datos(
        entrenamiento=crear_loader(ConcatDataset([entrenamiento] * repeticiones), tamano_lote, barajar=True),
        validacion=crear_loader(validacion, tamano_lote, barajar=False),
        test=crear_loader(test, tamano_lote, barajar=False),
        clases=entrenamiento.classes,
    )
