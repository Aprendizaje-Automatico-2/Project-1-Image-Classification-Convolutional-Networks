from collections.abc import Callable
from time import time

import torch
import torch.nn as nn

from fishcls.datos.cargadores import crear_dataloaders
from fishcls.entrenamiento.bucle import METRICAS_HISTORIAL, entrenar
from fishcls.entrenamiento.parada_temprana import ParadaTemprana
from fishcls.evaluacion.metricas import evaluar
from fishcls.experimentos.configuracion import ConfiguracionExperimento, construir_modelo
from fishcls.experimentos.resultados import guardar_experimento
from fishcls.modelos.medidas import contar_parametros, medir_milisegundos_por_imagen
from fishcls.modelos.preentrenados import congelar_extractor, parametros_cabeza
from fishcls.utilidades import describir_dispositivo, fijar_semilla, obtener_dispositivo


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
