import argparse
import platform

import pandas as pd
import torch
import torchvision

from fishcls.config import (
    ABLACION_FECHA,
    CONFIGURACIONES,
    DATOS_PREPARADOS,
    PRUEBAS_OPTUNA,
    RESUMEN_TEST,
    SEMILLA,
    TAMANO_CNN,
)
from fishcls.datos.cargadores import crear_dataloaders
from fishcls.evaluacion.fecha import ablacion_fecha
from fishcls.evaluacion.finalistas import evaluar_en_test
from fishcls.evaluacion.gradcam import figuras_gradcam
from fishcls.experimentos.ajuste import figuras_estudio, mejor_configuracion, optimizar
from fishcls.experimentos.configuracion import cargar_configuracion, guardar_configuracion
from fishcls.experimentos.ejecucion import ejecutar_experimento
from fishcls.experimentos.resultados import cargar_historial, cargar_resumen
from fishcls.graficas.comparativas import accuracy_frente_a_parametros, barras_ablacion, barras_experimentos
from fishcls.graficas.comun import guardar_figura
from fishcls.graficas.curvas import comparar_historiales
from fishcls.graficas.imagenes import galeria_aumentado, lote_de_ejemplo
from fishcls.preparacion.imagenes import abrir_imagen, listar_imagenes
from fishcls.preparacion.preparar import preparar_dataset, resumir_particion
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


def evaluar(argumentos: argparse.Namespace) -> None:
    tabla = pd.DataFrame([evaluar_en_test(nombre) for nombre in argumentos.experimentos])
    tabla.to_csv(RESUMEN_TEST, index=False)
    print(tabla.to_string(index=False))
    figura = accuracy_frente_a_parametros(tabla, argumentos.titulo, "acc_test", "Accuracy de test (IC 95 % de Wilson)")
    print(guardar_figura(figura, f"{argumentos.salida}.png"))


def gradcam(argumentos: argparse.Namespace) -> None:
    for nombre, figura in figuras_gradcam(argumentos.experimentos).items():
        print(guardar_figura(figura, f"{argumentos.salida}_{nombre}.png"))


def tapar_fecha(argumentos: argparse.Namespace) -> None:
    tabla = ablacion_fecha(argumentos.experimentos)
    tabla.to_csv(ABLACION_FECHA, index=False)
    print(tabla.to_string(index=False))
    print(guardar_figura(barras_ablacion(tabla, argumentos.titulo), f"{argumentos.salida}.png"))
