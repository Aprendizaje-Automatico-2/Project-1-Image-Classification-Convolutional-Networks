import unittest

import numpy as np

from fishcls.evaluacion.metricas import Evaluacion


def evaluacion_con(aciertos: int, imagenes: int) -> Evaluacion:
    y_real = np.zeros(imagenes, dtype=int)
    y_pred = np.where(np.arange(imagenes) < aciertos, 0, 1)
    return Evaluacion(y_real, y_pred, ["a", "b"])


class IntervaloWilsonTest(unittest.TestCase):
    def test_mitad_de_aciertos_es_simetrico(self):
        inferior, superior = evaluacion_con(50, 100).intervalo_wilson
        self.assertAlmostEqual(inferior, 0.4038, places=4)
        self.assertAlmostEqual(superior, 0.5962, places=4)

    def test_todo_aciertos_no_llega_a_cero_de_anchura(self):
        inferior, superior = evaluacion_con(57, 57).intervalo_wilson
        self.assertAlmostEqual(inferior, 57 / (57 + 1.959964**2), places=4)
        self.assertAlmostEqual(superior, 1.0)


if __name__ == "__main__":
    unittest.main()
