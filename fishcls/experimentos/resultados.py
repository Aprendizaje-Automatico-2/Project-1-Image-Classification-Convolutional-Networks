from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from fishcls.config import EXPERIMENTOS, RESUMEN_EXPERIMENTOS
from fishcls.datos.cargadores import Datos, crear_dataloaders
from fishcls.evaluacion.metricas import Evaluacion
from fishcls.experimentos.configuracion import ConfiguracionExperimento, cargar_configuracion, construir_modelo
from fishcls.graficas.comun import guardar_figura
from fishcls.graficas.curvas import curvas_entrenamiento
from fishcls.graficas.matriz_confusion import matriz_confusion
from fishcls.utilidades import guardar_json


def guardar_experimento(
    config: ConfiguracionExperimento,
    historial: dict[str, list[float]],
    evaluacion: Evaluacion,
    modelo: nn.Module,
    resumen: dict,
) -> None:
    carpeta = EXPERIMENTOS / config.nombre
    guardar_json(asdict(config), carpeta / "configuracion.json")
    guardar_json({**resumen, "informe_val": evaluacion.informe(como_diccionario=True)}, carpeta / "metricas.json")
    pd.DataFrame(historial).rename_axis("epoca").to_csv(carpeta / "historial.csv")
    torch.save(modelo.state_dict(), carpeta / "modelo.pt")
    guardar_figura(curvas_entrenamiento(historial, resumen["mejor_epoca"], config.nombre), "curvas.png", carpeta)
    guardar_figura(matriz_confusion(evaluacion, f"{config.nombre} · validación"), "matriz_confusion_val.png", carpeta)
    actualizar_resumen({**resumen, **{clave: str(valor) for clave, valor in asdict(config).items() if clave != "nombre"}})


def actualizar_resumen(fila: dict) -> None:
    anteriores = pd.read_csv(RESUMEN_EXPERIMENTOS) if RESUMEN_EXPERIMENTOS.exists() else pd.DataFrame()
    if not anteriores.empty:
        anteriores = anteriores[anteriores["nombre"] != fila["nombre"]]
    pd.concat([anteriores, pd.DataFrame([fila])]).sort_values("nombre").to_csv(RESUMEN_EXPERIMENTOS, index=False)


def cargar_resumen(patron: str) -> pd.DataFrame:
    tabla = pd.read_csv(RESUMEN_EXPERIMENTOS)
    return tabla[tabla["nombre"].str.fullmatch(patron)].reset_index(drop=True)


def cargar_historial(nombre: str) -> dict[str, list[float]]:
    return pd.read_csv(EXPERIMENTOS / nombre / "historial.csv", index_col="epoca").to_dict(orient="list")


def cargar_modelo(
    carpeta: Path, config: ConfiguracionExperimento, num_clases: int, dispositivo: torch.device
) -> nn.Module:
    modelo = construir_modelo(config, num_clases)
    modelo.load_state_dict(torch.load(carpeta / "modelo.pt", map_location=dispositivo))
    return modelo.to(dispositivo)


def cargar_experimento(nombre: str, dispositivo: torch.device) -> tuple[ConfiguracionExperimento, nn.Module, Datos]:
    carpeta = EXPERIMENTOS / nombre
    config = cargar_configuracion(carpeta / "configuracion.json")
    datos = crear_dataloaders(config.tamano, tamano_lote=config.tamano_lote)
    return config, cargar_modelo(carpeta, config, len(datos.clases), dispositivo), datos
