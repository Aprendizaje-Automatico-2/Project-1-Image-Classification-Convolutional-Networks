import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from fishcls.config import INTENTOS_PARTICION, PARTICIONES, PROPORCION_TEST, PROPORCION_VALIDACION, SEMILLA


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
