import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS_ORIGINALES = RAIZ / "data" / "Fish_Dataset"
DATOS_PREPARADOS = RAIZ / "data" / "preparados"
PARTICION = RAIZ / "particion.json"
CONFIGURACIONES = RAIZ / "configuraciones"
EXPERIMENTOS = RAIZ / "resultados" / "experimentos"
RESUMEN_EXPERIMENTOS = RAIZ / "resultados" / "resumen.csv"
FIGURAS = RAIZ / "resultados" / "figuras"

SEMILLA = 42
EXTENSIONES_IMAGEN = (".png", ".jpg", ".jpeg")
PARTICIONES = ("entrenamiento", "validacion", "test")
PROPORCION_VALIDACION = 0.15
PROPORCION_TEST = 0.15
INTENTOS_PARTICION = 50

TAMANO_PREPARADO = (300, 400)
TAMANO_CNN = (150, 200)
TAMANO_TRANSFERENCIA = (224, 224)
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIACION_IMAGENET = (0.229, 0.224, 0.225)

LADO_HUELLA = 16
UMBRAL_CASI_DUPLICADO = 6
ANCHO_MINIMO_KODAK = 2000

NUM_PROCESOS = min(4, os.cpu_count() or 1)
