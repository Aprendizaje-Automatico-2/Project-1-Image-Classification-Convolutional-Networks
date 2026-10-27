from dataclasses import asdict, dataclass
from pathlib import Path

import torch.nn as nn

from fishcls.config import SEMILLA, TAMANO_CNN
from fishcls.modelos.convnet import ConvNet
from fishcls.modelos.preentrenados import PREENTRENADOS, modelo_preentrenado
from fishcls.utilidades import cargar_json, guardar_json


@dataclass(frozen=True)
class ConfiguracionExperimento:
    nombre: str
    modelo: str = "convnet"
    estrategia: str = "ajuste_fino"
    epocas_congelado: int = 10
    factor_lr_extractor: float = 0.1
    filtros: tuple[int, ...] = (16, 32, 64)
    neuronas_fc: tuple[int, ...] = (128,)
    tamano_kernel: int = 5
    activacion: str = "relu"
    dropout: float = 0.25
    batchnorm: bool = False
    pooling_global: bool = False
    tasa_aprendizaje: float = 1e-3
    regularizacion_l2: float = 0.0
    tamano_lote: int = 32
    epocas: int = 100
    paciencia: int = 10
    aumentado: str = "completo"
    repeticiones: int = 4
    tamano: tuple[int, int] = TAMANO_CNN
    semilla: int = SEMILLA


def cargar_configuracion(ruta: Path) -> ConfiguracionExperimento:
    valores = {clave: tuple(valor) if isinstance(valor, list) else valor for clave, valor in cargar_json(ruta).items()}
    return ConfiguracionExperimento(**{"nombre": ruta.stem, **valores})


def guardar_configuracion(config: ConfiguracionExperimento, ruta: Path) -> None:
    por_defecto = ConfiguracionExperimento(nombre=config.nombre)
    cambios = {campo: valor for campo, valor in asdict(config).items()
               if campo != "nombre" and valor != getattr(por_defecto, campo)}
    guardar_json(cambios, ruta)


def construir_modelo(config: ConfiguracionExperimento, num_clases: int) -> nn.Module:
    if config.modelo in PREENTRENADOS:
        return modelo_preentrenado(config.modelo, num_clases)
    return ConvNet(
        num_clases=num_clases,
        tamano=config.tamano,
        filtros=config.filtros,
        neuronas_fc=config.neuronas_fc,
        tamano_kernel=config.tamano_kernel,
        activacion=config.activacion,
        dropout=config.dropout,
        batchnorm=config.batchnorm,
        pooling_global=config.pooling_global,
    )
