from collections.abc import Callable

import optuna
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from optuna.visualization import matplotlib as visualizacion

from fishcls.config import ESTUDIO_OPTUNA, SEMILLA
from fishcls.experimentos.configuracion import ConfiguracionExperimento
from fishcls.experimentos.ejecucion import ejecutar_experimento

NOMBRE_ESTUDIO = "convnet"
OBJETIVO = "Pérdida de validación"
TEXTOS_FIGURAS = {
    "historia": ("Historia de la búsqueda", "Prueba", OBJETIVO),
    "importancia": ("Importancia de cada hiperparámetro (fANOVA)", "Importancia", "Hiperparámetro"),
    "coordenadas_paralelas": ("Coordenadas paralelas", "", ""),
    "valores_intermedios": ("Pérdida de validación en cada época", "Época", OBJETIVO),
}


def configuracion_sugerida(prueba: optuna.Trial, nombre: str) -> ConfiguracionExperimento:
    bloques = prueba.suggest_int("bloques", 3, 5)
    filtros_base = prueba.suggest_categorical("filtros_base", [8, 16, 32])
    capas_fc = prueba.suggest_int("capas_fc", 1, 2)
    neuronas = prueba.suggest_categorical("neuronas_fc", [64, 128, 256, 512])
    return ConfiguracionExperimento(
        nombre=nombre,
        filtros=tuple(filtros_base * 2**bloque for bloque in range(bloques)),
        neuronas_fc=tuple(neuronas // 2**capa for capa in range(capas_fc)),
        tamano_kernel=prueba.suggest_categorical("tamano_kernel", [3, 5]),
        activacion=prueba.suggest_categorical("activacion", ["relu", "leaky_relu", "gelu", "silu"]),
        dropout=prueba.suggest_float("dropout", 0.0, 0.5),
        batchnorm=prueba.suggest_categorical("batchnorm", [False, True]),
        tasa_aprendizaje=prueba.suggest_float("tasa_aprendizaje", 1e-4, 3e-3, log=True),
        regularizacion_l2=prueba.suggest_float("regularizacion_l2", 1e-6, 1e-3, log=True),
        tamano_lote=prueba.suggest_categorical("tamano_lote", [16, 32, 64]),
    )


def informar_a_optuna(prueba: optuna.Trial) -> Callable[[int, float], None]:
    def informar(epoca: int, perdida_val: float) -> None:
        prueba.report(perdida_val, epoca)
        if prueba.should_prune():
            raise optuna.TrialPruned()

    return informar


def objetivo(prueba: optuna.Trial) -> float:
    config = configuracion_sugerida(prueba, f"optuna_{prueba.number:03d}")
    resumen = ejecutar_experimento(config, al_terminar_epoca=informar_a_optuna(prueba), guardar=False)
    return resumen["perdida_val"]


def crear_estudio() -> optuna.Study:
    return optuna.create_study(
        study_name=NOMBRE_ESTUDIO,
        storage=f"sqlite:///{ESTUDIO_OPTUNA}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEMILLA),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )


def optimizar(pruebas: int) -> optuna.Study:
    ESTUDIO_OPTUNA.parent.mkdir(parents=True, exist_ok=True)
    estudio = crear_estudio()
    estudio.optimize(objetivo, callbacks=[optuna.study.MaxTrialsCallback(pruebas, states=None)], gc_after_trial=True)
    return estudio


def mejor_configuracion(estudio: optuna.Study, nombre: str) -> ConfiguracionExperimento:
    return configuracion_sugerida(optuna.trial.FixedTrial(estudio.best_params), nombre)


def dibujar_estudio(estudio: optuna.Study) -> dict[str, Axes]:
    importancia = optuna.importance.FanovaImportanceEvaluator(seed=SEMILLA)
    return {
        "historia": visualizacion.plot_optimization_history(estudio, target_name=OBJETIVO),
        "importancia": visualizacion.plot_param_importances(estudio, evaluator=importancia, target_name=OBJETIVO),
        "coordenadas_paralelas": visualizacion.plot_parallel_coordinate(estudio, target_name=OBJETIVO),
        "valores_intermedios": visualizacion.plot_intermediate_values(estudio),
    }


def figuras_estudio(estudio: optuna.Study) -> dict[str, Figure]:
    figuras = {}
    for nombre, eje in dibujar_estudio(estudio).items():
        titulo, eje_x, eje_y = TEXTOS_FIGURAS[nombre]
        eje.set_title("", loc="left")
        eje.set(title=titulo, xlabel=eje_x, ylabel=eje_y)
        leyenda = eje.get_legend()
        for texto in leyenda.get_texts() if leyenda else []:
            texto.set_text(texto.get_text().replace("Best Value", "Mejor hasta ahora").replace("Trial", "Prueba "))
        eje.figure.set_size_inches(11, 5.5)
        figuras[nombre] = eje.figure
    return figuras
