from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.figure import Figure
from torchvision.datasets import ImageFolder

from fishcls.config import ANCHO_MINIMO_KODAK, PARTICION, RAIZ, TAMANO_CNN, ZONA_CONTROL, ZONA_FECHA
from fishcls.datos.cargadores import Datos, crear_dataloaders
from fishcls.evaluacion import evaluar
from fishcls.experimento import ConfiguracionExperimento, cargar_experimento
from fishcls.graficas import cuadricula_gradcam
from fishcls.modelos import capa_gradcam
from fishcls.utilidades import cargar_json, obtener_dispositivo

TITULOS_GRADCAM = {
    "fallos": "Grad-CAM de las imágenes de test que falla algún finalista",
    "aciertos": "Grad-CAM de un acierto de todos los finalistas por clase",
}
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


def mapa_gradcam(modelo: nn.Module, capa: nn.Module, imagen: torch.Tensor, clase: int) -> np.ndarray:
    activaciones = []
    gancho = capa.register_forward_hook(lambda _modulo, _entrada, salida: activaciones.append(salida))
    try:
        puntuacion = modelo.eval()(imagen.unsqueeze(0))[0, clase]
    finally:
        gancho.remove()
    (gradiente,) = torch.autograd.grad(puntuacion, activaciones[0])
    mapa = F.relu((gradiente.mean(dim=(2, 3), keepdim=True) * activaciones[0]).sum(dim=1))[0]
    return (mapa / mapa.max().clamp(min=1e-8)).detach().cpu().numpy()


def mapas_gradcam(
    experimento: tuple[ConfiguracionExperimento, nn.Module, Datos],
    prediccion: np.ndarray,
    indices: np.ndarray,
    dispositivo: torch.device,
) -> list[tuple[np.ndarray, int]]:
    config, modelo, datos = experimento
    capa = capa_gradcam(modelo, config.modelo)
    clases = [int(prediccion[i]) for i in indices]
    return [(mapa_gradcam(modelo, capa, datos.test.dataset[i][0].to(dispositivo), clase), clase)
            for i, clase in zip(indices, clases)]


def figuras_gradcam(nombres: list[str]) -> dict[str, Figure]:
    dispositivo = obtener_dispositivo()
    test = crear_dataloaders(TAMANO_CNN).test.dataset
    reales = np.array(test.targets)
    experimentos = {nombre: cargar_experimento(nombre, dispositivo) for nombre in nombres}
    predicciones = {nombre: evaluar(modelo, datos.test, dispositivo, datos.clases).y_pred
                    for nombre, (_, modelo, datos) in experimentos.items()}
    aciertan_todos = np.all([prediccion == reales for prediccion in predicciones.values()], axis=0)
    indices = {
        "fallos": np.flatnonzero(~aciertan_todos),
        "aciertos": np.concatenate([np.flatnonzero(aciertan_todos & (reales == clase))[:1]
                                    for clase in range(len(test.classes))]),
    }
    figuras = {}
    for clave, seleccion in indices.items():
        mapas = {nombre.split("_")[0]: mapas_gradcam(experimentos[nombre], predicciones[nombre], seleccion, dispositivo)
                 for nombre in nombres}
        imagenes = [test[i][0] for i in seleccion]
        figuras[clave] = cuadricula_gradcam(imagenes, reales[seleccion].tolist(), mapas, test.classes,
                                            TITULOS_GRADCAM[clave])
    return figuras


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
