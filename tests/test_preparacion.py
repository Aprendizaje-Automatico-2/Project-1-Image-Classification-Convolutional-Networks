import unittest
from collections import Counter

import numpy as np

from fishcls.config import DATOS_PREPARADOS, PARTICION, PARTICIONES, RAIZ
from fishcls.preparacion.duplicados import agrupar_casi_duplicados
from fishcls.preparacion.particion import dividir_por_grupos
from fishcls.utilidades import cargar_json


def particiones_por_grupo(grupos, particion) -> dict:
    vistas = {}
    for grupo, nombre in zip(grupos, particion):
        vistas.setdefault(grupo, set()).add(nombre)
    return vistas


class AgruparTest(unittest.TestCase):
    def setUp(self):
        generador = np.random.default_rng(0)
        self.base = generador.random(256) > 0.5
        self.parecida = self.base.copy()
        self.parecida[:3] = ~self.parecida[:3]
        self.distinta = generador.random(256) > 0.5

    def test_casi_duplicados_de_la_misma_clase_comparten_grupo(self):
        grupos = agrupar_casi_duplicados(np.stack([self.base, self.parecida, self.distinta]), np.array(["a", "a", "a"]))
        self.assertEqual(grupos[0], grupos[1])
        self.assertNotEqual(grupos[0], grupos[2])

    def test_no_se_agrupan_clases_distintas(self):
        grupos = agrupar_casi_duplicados(np.stack([self.base, self.parecida]), np.array(["a", "b"]))
        self.assertNotEqual(grupos[0], grupos[1])


class DividirTest(unittest.TestCase):
    def setUp(self):
        generador = np.random.default_rng(0)
        self.etiquetas = np.repeat(np.arange(8), 50)
        self.grupos = np.concatenate([clase * 100 + generador.integers(0, 40, 50) for clase in range(8)])
        self.particion = dividir_por_grupos(self.etiquetas, self.grupos)

    def test_ningun_grupo_aparece_en_dos_particiones(self):
        self.assertTrue(all(len(vistas) == 1 for vistas in particiones_por_grupo(self.grupos, self.particion).values()))

    def test_cada_clase_aparece_en_todas_las_particiones(self):
        for clase in range(8):
            self.assertEqual(set(self.particion[self.etiquetas == clase]), set(PARTICIONES))

    def test_proporciones_cercanas_a_70_15_15(self):
        tamanos = Counter(self.particion)
        self.assertAlmostEqual(tamanos["entrenamiento"] / 400, 0.70, delta=0.05)
        self.assertAlmostEqual(tamanos["validacion"] / 400, 0.15, delta=0.03)
        self.assertAlmostEqual(tamanos["test"] / 400, 0.15, delta=0.03)


@unittest.skipUnless(PARTICION.exists(), "ejecuta antes: python -m fishcls preparar")
class ParticionRealTest(unittest.TestCase):
    def setUp(self):
        self.registros = cargar_json(PARTICION)

    def test_sin_fugas_entre_particiones(self):
        grupos = [registro["grupo"] for registro in self.registros]
        particion = [registro["particion"] for registro in self.registros]
        self.assertTrue(all(len(vistas) == 1 for vistas in particiones_por_grupo(grupos, particion).values()))

    @unittest.skipUnless(DATOS_PREPARADOS.exists(), "ejecuta antes: python -m fishcls preparar")
    def test_las_imagenes_preparadas_existen_en_su_particion(self):
        for registro in self.registros:
            self.assertIn(f"/{registro['particion']}/{registro['clase']}/", registro["preparada"])
            self.assertTrue((RAIZ / registro["preparada"]).exists())


if __name__ == "__main__":
    unittest.main()
