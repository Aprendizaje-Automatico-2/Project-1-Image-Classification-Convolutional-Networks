from collections.abc import Callable
from time import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from fishcls.entrenamiento.parada_temprana import ParadaTemprana

METRICAS_HISTORIAL = ("perdida_train", "perdida_val", "acc_train", "acc_val")


def ejecutar_epoca(
    modelo: nn.Module,
    loader: DataLoader,
    funcion_perdida: nn.Module,
    dispositivo: torch.device,
    optimizador: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    entrenando = optimizador is not None
    modelo.train(entrenando)
    perdida_total = 0.0
    y_real, y_pred = [], []
    with torch.set_grad_enabled(entrenando):
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            predicciones = modelo(imagenes)
            perdida = funcion_perdida(predicciones, etiquetas)
            if entrenando:
                optimizador.zero_grad()
                perdida.backward()
                optimizador.step()
            perdida_total += perdida.item()
            y_pred.append(predicciones.argmax(dim=1).cpu().numpy())
            y_real.append(etiquetas.cpu().numpy())
    return perdida_total / len(loader), np.concatenate(y_real), np.concatenate(y_pred)


def entrenar(
    modelo: nn.Module,
    loader_train: DataLoader,
    loader_val: DataLoader,
    optimizador: torch.optim.Optimizer,
    funcion_perdida: nn.Module,
    parada: ParadaTemprana,
    dispositivo: torch.device,
    epocas: int,
    primera_epoca: int = 1,
    al_terminar_epoca: Callable[[int, float], None] | None = None,
) -> dict[str, list[float]]:
    historial = {metrica: [] for metrica in METRICAS_HISTORIAL}
    ultima_epoca = primera_epoca + epocas - 1
    for epoca in range(primera_epoca, ultima_epoca + 1):
        comienzo = time()
        perdida_train, y_real_train, y_pred_train = ejecutar_epoca(
            modelo, loader_train, funcion_perdida, dispositivo, optimizador
        )
        perdida_val, y_real_val, y_pred_val = ejecutar_epoca(modelo, loader_val, funcion_perdida, dispositivo)
        historial["perdida_train"].append(perdida_train)
        historial["perdida_val"].append(perdida_val)
        historial["acc_train"].append(accuracy_score(y_real_train, y_pred_train))
        historial["acc_val"].append(accuracy_score(y_real_val, y_pred_val))
        detener = parada(perdida_val, modelo, epoca)
        estado = "mejora" if parada.mejor_epoca == epoca else f"sin mejora {parada.epocas_sin_mejora}/{parada.paciencia}"
        print(
            f"Época [{epoca:3d}/{ultima_epoca}] | "
            f"Train Loss: {perdida_train:.4f} | Train Acc: {historial['acc_train'][-1]:.4f} | "
            f"Val Loss: {perdida_val:.4f} | Val Acc: {historial['acc_val'][-1]:.4f} | "
            f"{estado} | {time() - comienzo:.1f}s"
        )
        if al_terminar_epoca:
            al_terminar_epoca(epoca, perdida_val)
        if detener:
            print(f"Parada temprana en la época {epoca}: la mejor fue la {parada.mejor_epoca} "
                  f"(Val Loss: {parada.mejor_perdida_val:.4f})")
            break
    modelo.load_state_dict(parada.mejor_estado)
    return historial
