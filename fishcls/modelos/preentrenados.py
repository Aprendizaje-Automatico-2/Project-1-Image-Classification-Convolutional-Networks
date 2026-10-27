import torch.nn as nn
from torchvision import models

PREENTRENADOS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "classifier"),
}


def modelo_preentrenado(nombre: str, num_clases: int, con_pesos: bool = True) -> nn.Module:
    constructor, pesos, nombre_cabeza = PREENTRENADOS[nombre]
    modelo = constructor(weights=pesos if con_pesos else None)
    cabeza = getattr(modelo, nombre_cabeza)
    if isinstance(cabeza, nn.Linear):
        setattr(modelo, nombre_cabeza, nn.Linear(cabeza.in_features, num_clases))
    else:
        cabeza[-1] = nn.Linear(cabeza[-1].in_features, num_clases)
    return modelo


def parametros_cabeza(modelo: nn.Module, nombre: str) -> list[nn.Parameter]:
    if nombre not in PREENTRENADOS:
        return list(modelo.parameters())
    return list(getattr(modelo, PREENTRENADOS[nombre][2]).parameters())


def congelar_extractor(modelo: nn.Module, nombre: str, congelar: bool) -> None:
    cabeza = {id(parametro) for parametro in parametros_cabeza(modelo, nombre)}
    for parametro in modelo.parameters():
        parametro.requires_grad = id(parametro) in cabeza or not congelar
