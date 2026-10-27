import argparse
from pathlib import Path

from fishcls.comandos import (
    ajustar,
    comparar,
    entrenar,
    evaluar,
    gradcam,
    mostrar_info,
    preparar,
    resumir,
    tapar_fecha,
    ver_aumentado,
)


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
    evaluar_parser = comandos.add_parser("evaluar", help="evalúa en test los experimentos finalistas")
    evaluar_parser.add_argument("experimentos", nargs="+")
    evaluar_parser.add_argument("--titulo", required=True)
    evaluar_parser.add_argument("--salida", required=True)
    evaluar_parser.set_defaults(ejecutar=evaluar)
    gradcam_parser = comandos.add_parser("gradcam", help="mapas Grad-CAM de los finalistas sobre test")
    gradcam_parser.add_argument("experimentos", nargs="+")
    gradcam_parser.add_argument("--salida", required=True)
    gradcam_parser.set_defaults(ejecutar=gradcam)
    fecha_parser = comandos.add_parser("tapar-fecha", help="test con la fecha impresa tapada y una esquina de control")
    fecha_parser.add_argument("experimentos", nargs="+")
    fecha_parser.add_argument("--titulo", required=True)
    fecha_parser.add_argument("--salida", required=True)
    fecha_parser.set_defaults(ejecutar=tapar_fecha)
    return parser


def main() -> None:
    argumentos = construir_parser().parse_args()
    argumentos.ejecutar(argumentos)


if __name__ == "__main__":
    main()
