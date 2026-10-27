import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as funciones
from PIL import Image, ImageFilter

from fishcls.config import DESVIACION_IMAGENET, MEDIA_IMAGENET

AUMENTADOS = {
    "ninguno": set(),
    "geometrico": {"geometrico"},
    "fotometrico": {"fotometrico"},
    "completo": {"geometrico", "fotometrico", "oclusion"},
}


class GammaAleatoria:
    def __init__(self, minimo: float, maximo: float):
        self.minimo = minimo
        self.maximo = maximo

    def __call__(self, imagen: Image.Image) -> Image.Image:
        return funciones.adjust_gamma(imagen, random.uniform(self.minimo, self.maximo))


class FiltroMediana:
    def __init__(self, tamano: int):
        self.tamano = tamano

    def __call__(self, imagen: Image.Image) -> Image.Image:
        return imagen.filter(ImageFilter.MedianFilter(self.tamano))


def geometricas(tamano: tuple[int, int]) -> list:
    return [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(tamano, scale=(0.7, 1.0), ratio=(1.2, 1.45)),
    ]


def fotometricas() -> list:
    return [
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02),
        transforms.RandomApply([GammaAleatoria(0.7, 1.4)], p=0.5),
        transforms.RandomApply([FiltroMediana(3)], p=0.3),
    ]


def a_tensor_normalizado() -> list:
    return [transforms.ToTensor(), transforms.Normalize(MEDIA_IMAGENET, DESVIACION_IMAGENET)]


def transformaciones_entrenamiento(tamano: tuple[int, int], aumentado: str) -> transforms.Compose:
    familias = AUMENTADOS[aumentado]
    pasos = geometricas(tamano) if "geometrico" in familias else [transforms.Resize(tamano)]
    if "fotometrico" in familias:
        pasos += fotometricas()
    pasos += a_tensor_normalizado()
    if "oclusion" in familias:
        pasos.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)))
    return transforms.Compose(pasos)


def transformaciones_evaluacion(tamano: tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(tamano), *a_tensor_normalizado()])
