from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder

from fishcls.config import ANCHO_MINIMO_KODAK, PARTICION, RAIZ, ZONA_CONTROL, ZONA_FECHA
from fishcls.evaluacion.metricas import evaluar
from fishcls.experimentos.resultados import cargar_experimento
from fishcls.utilidades import cargar_json, obtener_dispositivo

ZONAS_TAPADAS = {"original": None, "fecha tapada": ZONA_FECHA, "esquina izquierda tapada": ZONA_CONTROL}


class TaparZona(nn.Module):
    def __init__(self, zona: tuple[float, float, float, float]):
        super().__init__()
        self.zona = zona

    def forward(self, imagenes: torch.Tensor) -> torch.Tensor:
        alto, ancho = imagenes.shape[-2:]
        izquierda, arriba, derecha, abajo = self.zona
        tapadas = imagenes.clone()
        tapadas[..., round(arriba * alto):round(abajo * alto), round(izquierda * ancho):round(derecha * ancho)] = 0
        return tapadas


def imagenes_con_fecha(dataset: ImageFolder) -> np.ndarray:
    anchos = {RAIZ / registro["preparada"]: registro["ancho_original"] for registro in cargar_json(PARTICION)}
    return np.array([anchos[Path(ruta)] >= ANCHO_MINIMO_KODAK for ruta, _ in dataset.samples])


def ablacion_fecha(nombres: list[str]) -> pd.DataFrame:
    dispositivo = obtener_dispositivo()
    filas = []
    for nombre in nombres:
        _, modelo, datos = cargar_experimento(nombre, dispositivo)
        con_fecha = imagenes_con_fecha(datos.test.dataset)
        evaluaciones = {}
        for condicion, zona in ZONAS_TAPADAS.items():
            tapado = nn.Sequential(TaparZona(zona), modelo) if zona else modelo
            evaluaciones[condicion] = evaluar(tapado, datos.test, dispositivo, datos.clases)
        original = evaluaciones["original"].y_pred
        for condicion, evaluacion in evaluaciones.items():
            for fotos, mascara in {"con fecha": con_fecha, "sin fecha": ~con_fecha}.items():
                filas.append({
                    "nombre": nombre,
                    "condicion": condicion,
                    "fotos": fotos,
                    "imagenes": int(mascara.sum()),
                    "acc_test": round(float((evaluacion.y_pred[mascara] == evaluacion.y_real[mascara]).mean()), 4),
                    "cambios": int((evaluacion.y_pred[mascara] != original[mascara]).sum()),
                })
    return pd.DataFrame(filas)
