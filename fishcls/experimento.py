from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

import pandas as pd
import torch
import torch.nn as nn

from fishcls.config import EXPERIMENTOS, RESUMEN_EXPERIMENTOS, SEMILLA, TAMANO_CNN
from fishcls.datos import Datos, crear_dataloaders
from fishcls.entrenamiento import METRICAS_HISTORIAL, ParadaTemprana, entrenar
from fishcls.evaluacion import Evaluacion, evaluar
from fishcls.graficas import curvas_entrenamiento, guardar_figura, matriz_confusion
from fishcls.modelos import (
    PREENTRENADOS,
    ConvNet,
    congelar_extractor,
    contar_parametros,
    medir_milisegundos_por_imagen,
    modelo_preentrenado,
    parametros_cabeza,
)
from fishcls.utilidades import cargar_json, describir_dispositivo, fijar_semilla, guardar_json, obtener_dispositivo


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


def fases_entrenamiento(config: ConfiguracionExperimento) -> list[tuple[bool, int]]:
    return {
        "congelado": [(True, config.epocas)],
        "ajuste_fino": [(False, config.epocas)],
        "dos_fases": [(True, config.epocas_congelado), (False, config.epocas)],
    }[config.estrategia]


def crear_optimizador(modelo: nn.Module, config: ConfiguracionExperimento) -> torch.optim.Optimizer:
    cabeza = parametros_cabeza(modelo, config.modelo)
    de_la_cabeza = {id(parametro) for parametro in cabeza}
    extractor = [p for p in modelo.parameters() if p.requires_grad and id(p) not in de_la_cabeza]
    grupos = [{"params": cabeza, "lr": config.tasa_aprendizaje}]
    if extractor:
        grupos.append({"params": extractor, "lr": config.tasa_aprendizaje * config.factor_lr_extractor})
    return torch.optim.Adam(grupos, weight_decay=config.regularizacion_l2)


def ejecutar_experimento(
    config: ConfiguracionExperimento,
    al_terminar_epoca: Callable[[int, float], None] | None = None,
    guardar: bool = True,
) -> dict:
    fijar_semilla(config.semilla)
    dispositivo = obtener_dispositivo()
    datos = crear_dataloaders(config.tamano, config.aumentado, config.repeticiones, config.tamano_lote)
    modelo = construir_modelo(config, len(datos.clases)).to(dispositivo)
    parada = ParadaTemprana(config.paciencia)
    print(f"*** {config.nombre} · {contar_parametros(modelo):,} parámetros · {describir_dispositivo(dispositivo)}")

    comienzo = time()
    historial = {metrica: [] for metrica in METRICAS_HISTORIAL}
    for congelado, epocas in fases_entrenamiento(config):
        congelar_extractor(modelo, config.modelo, congelado)
        parada.reiniciar_paciencia()
        fase = entrenar(
            modelo, datos.entrenamiento, datos.validacion, crear_optimizador(modelo, config), nn.CrossEntropyLoss(),
            parada, dispositivo, epocas, primera_epoca=len(historial["perdida_train"]) + 1,
            al_terminar_epoca=al_terminar_epoca,
        )
        for metrica in historial:
            historial[metrica] += fase[metrica]
    segundos = time() - comienzo

    evaluacion = evaluar(modelo, datos.validacion, dispositivo, datos.clases)
    print(f"\n=== Classification report: VALIDACIÓN ({config.nombre}) ===\n{evaluacion.informe()}")
    resumen = {
        "nombre": config.nombre,
        "acc_val": round(evaluacion.accuracy, 4),
        "f1_val": round(evaluacion.f1_macro, 4),
        "perdida_val": float(f"{parada.mejor_perdida_val:.3g}"),
        "mejor_epoca": parada.mejor_epoca,
        "epocas_entrenadas": len(historial["perdida_train"]),
        "parametros": contar_parametros(modelo),
        "ms_imagen": round(medir_milisegundos_por_imagen(modelo, config.tamano, dispositivo), 2),
        "segundos": round(segundos),
    }
    if guardar:
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


def evaluar_en_test(nombre: str) -> dict:
    carpeta = EXPERIMENTOS / nombre
    dispositivo = obtener_dispositivo()
    _, modelo, datos = cargar_experimento(nombre, dispositivo)
    evaluacion = evaluar(modelo, datos.test, dispositivo, datos.clases)
    print(f"\n=== Classification report: TEST ({nombre}) ===\n{evaluacion.informe()}")
    inferior, superior = evaluacion.intervalo_wilson
    resumen = {
        "nombre": nombre,
        "acc_test": round(evaluacion.accuracy, 4),
        "ic_inferior": round(inferior, 4),
        "ic_superior": round(superior, 4),
        "f1_test": round(evaluacion.f1_macro, 4),
        "imagenes": len(evaluacion.y_real),
        "parametros": contar_parametros(modelo),
    }
    guardar_json({**resumen, "informe_test": evaluacion.informe(como_diccionario=True)}, carpeta / "metricas_test.json")
    guardar_figura(matriz_confusion(evaluacion, f"{nombre} · test"), "matriz_confusion_test.png", carpeta)
    return resumen
