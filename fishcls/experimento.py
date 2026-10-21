from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

import pandas as pd
import torch
import torch.nn as nn

from fishcls.config import EXPERIMENTOS, RESUMEN_EXPERIMENTOS, SEMILLA, TAMANO_CNN
from fishcls.datos import crear_dataloaders
from fishcls.entrenamiento import ParadaTemprana, entrenar
from fishcls.evaluacion import Evaluacion, evaluar
from fishcls.graficas import curvas_entrenamiento, guardar_figura, matriz_confusion
from fishcls.modelos import ConvNet, contar_parametros
from fishcls.utilidades import cargar_json, describir_dispositivo, fijar_semilla, guardar_json, obtener_dispositivo


@dataclass(frozen=True)
class ConfiguracionExperimento:
    nombre: str
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
    return ConfiguracionExperimento(nombre=ruta.stem, **valores)


def construir_modelo(config: ConfiguracionExperimento, num_clases: int) -> nn.Module:
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


def ejecutar_experimento(config: ConfiguracionExperimento) -> dict:
    fijar_semilla(config.semilla)
    dispositivo = obtener_dispositivo()
    datos = crear_dataloaders(config.tamano, config.aumentado, config.repeticiones, config.tamano_lote)
    modelo = construir_modelo(config, len(datos.clases)).to(dispositivo)
    optimizador = torch.optim.Adam(
        modelo.parameters(), lr=config.tasa_aprendizaje, weight_decay=config.regularizacion_l2
    )
    parada = ParadaTemprana(config.paciencia)
    print(f"*** {config.nombre} · {contar_parametros(modelo):,} parámetros · {describir_dispositivo(dispositivo)}")

    comienzo = time()
    historial = entrenar(
        modelo, datos.entrenamiento, datos.validacion, optimizador, nn.CrossEntropyLoss(), parada, dispositivo,
        config.epocas,
    )
    segundos = time() - comienzo

    evaluacion = evaluar(modelo, datos.validacion, dispositivo, datos.clases)
    print(f"\n=== Classification report: VALIDACIÓN ({config.nombre}) ===\n{evaluacion.informe()}")
    resumen = {
        "nombre": config.nombre,
        "acc_val": round(evaluacion.accuracy, 4),
        "f1_val": round(evaluacion.f1_macro, 4),
        "perdida_val": round(parada.mejor_perdida_val, 4),
        "mejor_epoca": parada.mejor_epoca,
        "epocas_entrenadas": len(historial["perdida_train"]),
        "parametros": contar_parametros(modelo),
        "segundos": round(segundos),
    }
    guardar_experimento(config, historial, evaluacion, modelo, resumen)
    return resumen


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


def cargar_historial(nombre: str) -> dict[str, list[float]]:
    return pd.read_csv(EXPERIMENTOS / nombre / "historial.csv", index_col="epoca").to_dict(orient="list")
