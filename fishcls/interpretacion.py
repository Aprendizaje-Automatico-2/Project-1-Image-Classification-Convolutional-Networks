import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.figure import Figure

from fishcls.config import TAMANO_CNN
from fishcls.datos import Datos, crear_dataloaders
from fishcls.evaluacion import evaluar
from fishcls.experimento import ConfiguracionExperimento, cargar_experimento
from fishcls.graficas import cuadricula_gradcam
from fishcls.modelos import capa_gradcam
from fishcls.utilidades import obtener_dispositivo

TITULOS_GRADCAM = {
    "fallos": "Grad-CAM de las imágenes de test que falla algún finalista",
    "aciertos": "Grad-CAM de un acierto de todos los finalistas por clase",
}


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
