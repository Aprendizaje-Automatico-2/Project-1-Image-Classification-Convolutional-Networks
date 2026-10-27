import shutil

import numpy as np
import pandas as pd

from fishcls.config import ANCHO_MINIMO_KODAK, DATOS_ORIGINALES, DATOS_PREPARADOS, PARTICION, PARTICIONES, RAIZ
from fishcls.preparacion.duplicados import agrupar_casi_duplicados, huella_perceptual
from fishcls.preparacion.imagenes import listar_imagenes, reducir
from fishcls.preparacion.particion import dividir_por_grupos
from fishcls.utilidades import guardar_json


def preparar_dataset() -> list[dict]:
    rutas, clases = zip(*listar_imagenes(DATOS_ORIGINALES))
    reducidas, anchos_originales = zip(*(reducir(ruta) for ruta in rutas))
    clases = np.array(clases)
    grupos = agrupar_casi_duplicados(np.stack([huella_perceptual(imagen) for imagen in reducidas]), clases)
    particion = dividir_por_grupos(clases, grupos)

    shutil.rmtree(DATOS_PREPARADOS, ignore_errors=True)
    registros = []
    for ruta, clase, imagen, ancho, grupo, nombre_particion in zip(
        rutas, clases, reducidas, anchos_originales, grupos, particion
    ):
        destino = DATOS_PREPARADOS / nombre_particion / clase / f"{ruta.stem}.jpg"
        destino.parent.mkdir(parents=True, exist_ok=True)
        imagen.save(destino, quality=95)
        registros.append(
            {
                "original": str(ruta.relative_to(RAIZ)),
                "preparada": str(destino.relative_to(RAIZ)),
                "clase": str(clase),
                "grupo": int(grupo),
                "particion": nombre_particion,
                "ancho_original": ancho,
            }
        )
    guardar_json(registros, PARTICION)
    return registros


def resumir_particion(registros: list[dict]) -> pd.DataFrame:
    tabla = pd.DataFrame(registros).assign(kodak=lambda t: t["ancho_original"] >= ANCHO_MINIMO_KODAK)
    por_clase = tabla.groupby("clase")
    resumen = pd.crosstab(tabla["clase"], tabla["particion"])[list(PARTICIONES)]
    resumen["total"] = resumen.sum(axis=1)
    resumen["grupos"] = por_clase["grupo"].nunique()
    resumen["% kodak"] = por_clase["kodak"].mean().mul(100).round().astype(int)
    return resumen
