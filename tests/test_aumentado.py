import unittest

import numpy as np
import torch
from PIL import Image

from fishcls.datos.aumentado import AUMENTADOS, transformaciones_entrenamiento, transformaciones_evaluacion
from fishcls.config import TAMANO_CNN, TAMANO_PREPARADO


def imagen_aleatoria() -> Image.Image:
    alto, ancho = TAMANO_PREPARADO
    return Image.fromarray(np.random.default_rng(0).integers(0, 256, (alto, ancho, 3), dtype=np.uint8))


class EvaluacionTest(unittest.TestCase):
    def test_es_determinista_y_respeta_el_tamano(self):
        transformar = transformaciones_evaluacion(TAMANO_CNN)
        imagen = imagen_aleatoria()
        primera, segunda = transformar(imagen), transformar(imagen)
        self.assertEqual(tuple(primera.shape), (3, *TAMANO_CNN))
        self.assertTrue(torch.equal(primera, segunda))


class EntrenamientoTest(unittest.TestCase):
    def test_todos_los_aumentados_respetan_el_tamano(self):
        for aumentado in AUMENTADOS:
            with self.subTest(aumentado=aumentado):
                salida = transformaciones_entrenamiento(TAMANO_CNN, aumentado)(imagen_aleatoria())
                self.assertEqual(tuple(salida.shape), (3, *TAMANO_CNN))

    def test_sin_aumentado_coincide_con_evaluacion(self):
        imagen = imagen_aleatoria()
        sin_aumentado = transformaciones_entrenamiento(TAMANO_CNN, "ninguno")(imagen)
        self.assertTrue(torch.equal(sin_aumentado, transformaciones_evaluacion(TAMANO_CNN)(imagen)))

    def test_el_aumentado_completo_varia_cada_vez(self):
        transformar = transformaciones_entrenamiento(TAMANO_CNN, "completo")
        imagen = imagen_aleatoria()
        self.assertFalse(torch.equal(transformar(imagen), transformar(imagen)))


if __name__ == "__main__":
    unittest.main()
