import copy

import torch.nn as nn


class ParadaTemprana:
    def __init__(self, paciencia: int):
        self.paciencia = paciencia
        self.mejor_perdida_val = float("inf")
        self.mejor_epoca = 0
        self.mejor_estado = None
        self.epocas_sin_mejora = 0

    def __call__(self, perdida_val: float, modelo: nn.Module, epoca: int) -> bool:
        if perdida_val < self.mejor_perdida_val:
            self.mejor_perdida_val = perdida_val
            self.mejor_epoca = epoca
            self.mejor_estado = copy.deepcopy(modelo.state_dict())
            self.epocas_sin_mejora = 0
        else:
            self.epocas_sin_mejora += 1
        return self.epocas_sin_mejora >= self.paciencia

    def reiniciar_paciencia(self) -> None:
        self.epocas_sin_mejora = 0
