from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from fishcls.entrenamiento import ejecutar_epoca


class Evaluacion(NamedTuple):
    y_real: np.ndarray
    y_pred: np.ndarray
    clases: list[str]

    @property
    def accuracy(self) -> float:
        return accuracy_score(self.y_real, self.y_pred)

    @property
    def f1_macro(self) -> float:
        return f1_score(self.y_real, self.y_pred, average="macro")

    def informe(self, como_diccionario: bool = False) -> str | dict:
        return classification_report(
            self.y_real,
            self.y_pred,
            labels=range(len(self.clases)),
            target_names=self.clases,
            zero_division=0,
            output_dict=como_diccionario,
        )


def evaluar(modelo: nn.Module, loader: DataLoader, dispositivo: torch.device, clases: list[str]) -> Evaluacion:
    _, y_real, y_pred = ejecutar_epoca(modelo, loader, nn.CrossEntropyLoss(), dispositivo)
    return Evaluacion(y_real, y_pred, clases)
