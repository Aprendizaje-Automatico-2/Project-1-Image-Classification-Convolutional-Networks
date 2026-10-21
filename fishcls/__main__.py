import argparse
import platform

import torch
import torchvision

from fishcls.config import DATOS_PREPARADOS, SEMILLA, TAMANO_CNN
from fishcls.datos import abrir_imagen, crear_dataloaders, listar_imagenes, preparar_dataset, resumir_particion
from fishcls.graficas import galeria_aumentado, guardar_figura, lote_de_ejemplo
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


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fishcls")
    comandos = parser.add_subparsers(dest="comando", required=True)
    comandos.add_parser("info", help="versiones y dispositivo").set_defaults(ejecutar=mostrar_info)
    comandos.add_parser("preparar", help="agrupa, divide y redimensiona el dataset").set_defaults(ejecutar=preparar)
    comandos.add_parser("ver-aumentado", help="figuras del data augmentation").set_defaults(ejecutar=ver_aumentado)
    return parser


def main() -> None:
    argumentos = construir_parser().parse_args()
    argumentos.ejecutar(argumentos)


if __name__ == "__main__":
    main()
