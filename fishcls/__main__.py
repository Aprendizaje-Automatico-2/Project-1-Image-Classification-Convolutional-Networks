import argparse
import platform
from pathlib import Path

import torch
import torchvision

from fishcls.ajuste import figuras_estudio, mejor_configuracion, optimizar
from fishcls.config import CONFIGURACIONES, DATOS_PREPARADOS, PRUEBAS_OPTUNA, SEMILLA, TAMANO_CNN
from fishcls.datos import abrir_imagen, crear_dataloaders, listar_imagenes, preparar_dataset, resumir_particion
from fishcls.experimento import (
    cargar_configuracion,
    cargar_historial,
    cargar_resumen,
    ejecutar_experimento,
    guardar_configuracion,
)
from fishcls.graficas import (
    accuracy_frente_a_parametros,
    barras_experimentos,
    comparar_historiales,
    galeria_aumentado,
    guardar_figura,
    lote_de_ejemplo,
)
from fishcls.utilidades import describir_dispositivo, fijar_semilla, obtener_dispositivo


def mostrar_info(_: argparse.Namespace) -> None:
    print(f"python      {platform.python_version()}")
    print(f"torch       {torch.__version__}")
    print(f"torchvision {torchvision.__version__}")
    print(f"dispositivo {describir_dispositivo(obtener_dispositivo())}")


def preparar(_: argparse.Namespace) -> None:
    registros = preparar_dataset()
    resumen = resumir_particion(registros)
    print(resumen.to_string())
    print(f"\n{len(registros)} imágenes en {resumen['grupos'].sum()} grupos → {DATOS_PREPARADOS}")


def ver_aumentado(_: argparse.Namespace) -> None:
    fijar_semilla(SEMILLA)
    ruta, _ = listar_imagenes(DATOS_PREPARADOS / "entrenamiento")[0]
    print(guardar_figura(galeria_aumentado(abrir_imagen(ruta), TAMANO_CNN), "aumentado_familias.png"))
    datos = crear_dataloaders(TAMANO_CNN, tamano_lote=16)
    imagenes, etiquetas = next(iter(datos.entrenamiento))
    print(guardar_figura(lote_de_ejemplo(imagenes, etiquetas, datos.clases), "aumentado_lote.png"))


def entrenar(argumentos: argparse.Namespace) -> None:
    for ruta in argumentos.configuraciones:
        print(ejecutar_experimento(cargar_configuracion(ruta)))


def comparar(argumentos: argparse.Namespace) -> None:
    historiales = {nombre: cargar_historial(nombre) for nombre in argumentos.experimentos}
    print(guardar_figura(comparar_historiales(historiales, argumentos.titulo), f"{argumentos.salida}.png"))


def resumir(argumentos: argparse.Namespace) -> None:
    tabla = cargar_resumen(argumentos.patron)
    columnas = ["nombre", "acc_val", "f1_val", "perdida_val", "mejor_epoca", "epocas_entrenadas", "parametros",
                "ms_imagen", "segundos"]
    print(tabla[[columna for columna in columnas if columna in tabla]].to_string(index=False))
    print(guardar_figura(barras_experimentos(tabla, argumentos.referencia, argumentos.titulo), f"{argumentos.salida}.png"))
    print(guardar_figura(accuracy_frente_a_parametros(tabla, argumentos.titulo), f"{argumentos.salida}_parametros.png"))


def ajustar(argumentos: argparse.Namespace) -> None:
    estudio = optimizar(argumentos.pruebas)
    estudio.trials_dataframe().to_csv(PRUEBAS_OPTUNA, index=False)
    for nombre, figura in figuras_estudio(estudio).items():
        print(guardar_figura(figura, f"fase5_optuna_{nombre}.png"))
    ruta = CONFIGURACIONES / "optuna" / "O01_optuna_mejor.json"
    guardar_configuracion(mejor_configuracion(estudio, ruta.stem), ruta)
    print(f"Mejor prueba {estudio.best_trial.number}: pérdida {estudio.best_value:.3g} · {estudio.best_params}")
    print(f"Configuración ganadora → {ruta}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fishcls")
    comandos = parser.add_subparsers(dest="comando", required=True)
    comandos.add_parser("info", help="versiones y dispositivo").set_defaults(ejecutar=mostrar_info)
    comandos.add_parser("preparar", help="agrupa, divide y redimensiona el dataset").set_defaults(ejecutar=preparar)
    comandos.add_parser("ver-aumentado", help="figuras del data augmentation").set_defaults(ejecutar=ver_aumentado)
    entrenar_parser = comandos.add_parser("entrenar", help="ejecuta uno o varios experimentos")
    entrenar_parser.add_argument("configuraciones", nargs="+", type=Path)
    entrenar_parser.set_defaults(ejecutar=entrenar)
    comparar_parser = comandos.add_parser("comparar", help="superpone las curvas de varios experimentos")
    comparar_parser.add_argument("experimentos", nargs="+")
    comparar_parser.add_argument("--titulo", required=True)
    comparar_parser.add_argument("--salida", required=True)
    comparar_parser.set_defaults(ejecutar=comparar)
    resumir_parser = comandos.add_parser("resumir", help="tabla y barras de los experimentos que cumplen un patrón")
    resumir_parser.add_argument("patron", help="expresión regular sobre el nombre, p. ej. 'C.*'")
    resumir_parser.add_argument("--referencia", required=True)
    resumir_parser.add_argument("--titulo", required=True)
    resumir_parser.add_argument("--salida", required=True)
    resumir_parser.set_defaults(ejecutar=resumir)
    ajustar_parser = comandos.add_parser("ajustar", help="búsqueda automática de hiperparámetros con Optuna")
    ajustar_parser.add_argument("--pruebas", type=int, default=30)
    ajustar_parser.set_defaults(ejecutar=ajustar)
    return parser


def main() -> None:
    argumentos = construir_parser().parse_args()
    argumentos.ejecutar(argumentos)


if __name__ == "__main__":
    main()
