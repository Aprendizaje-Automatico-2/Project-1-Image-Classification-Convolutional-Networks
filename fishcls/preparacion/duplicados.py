import numpy as np
from PIL import Image
from scipy.sparse.csgraph import connected_components

from fishcls.config import LADO_HUELLA, UMBRAL_CASI_DUPLICADO


def huella_perceptual(imagen: Image.Image) -> np.ndarray:
    grises = np.asarray(imagen.convert("L").resize((LADO_HUELLA, LADO_HUELLA)), dtype=np.float32)
    return (grises > grises.mean()).ravel()


def agrupar_casi_duplicados(huellas: np.ndarray, clases: np.ndarray) -> np.ndarray:
    distancias = (huellas[:, None, :] != huellas[None, :, :]).sum(axis=2)
    misma_clase = clases[:, None] == clases[None, :]
    _, grupos = connected_components((distancias <= UMBRAL_CASI_DUPLICADO) & misma_clase, directed=False)
    return grupos
