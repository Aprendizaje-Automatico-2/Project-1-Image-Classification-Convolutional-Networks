from typing import NamedTuple

from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder

from fishcls.config import DATOS_PREPARADOS, NUM_PROCESOS
from fishcls.datos.aumentado import transformaciones_entrenamiento, transformaciones_evaluacion


class Datos(NamedTuple):
    entrenamiento: DataLoader
    validacion: DataLoader
    test: DataLoader
    clases: list[str]


def crear_loader(dataset: Dataset, tamano_lote: int, barajar: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=tamano_lote,
        shuffle=barajar,
        num_workers=NUM_PROCESOS,
        persistent_workers=NUM_PROCESOS > 0,
    )


def crear_dataloaders(
    tamano: tuple[int, int], aumentado: str = "completo", repeticiones: int = 4, tamano_lote: int = 32
) -> Datos:
    entrenamiento = ImageFolder(
        DATOS_PREPARADOS / "entrenamiento", transform=transformaciones_entrenamiento(tamano, aumentado)
    )
    validacion = ImageFolder(DATOS_PREPARADOS / "validacion", transform=transformaciones_evaluacion(tamano))
    test = ImageFolder(DATOS_PREPARADOS / "test", transform=transformaciones_evaluacion(tamano))
    return Datos(
        entrenamiento=crear_loader(ConcatDataset([entrenamiento] * repeticiones), tamano_lote, barajar=True),
        validacion=crear_loader(validacion, tamano_lote, barajar=False),
        test=crear_loader(test, tamano_lote, barajar=False),
        clases=entrenamiento.classes,
    )
