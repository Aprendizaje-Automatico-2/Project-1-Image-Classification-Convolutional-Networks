import unittest

import torch

from fishcls.evaluacion.fecha import TaparZona


class TaparZonaTest(unittest.TestCase):
    def test_pone_a_cero_solo_la_zona_y_no_toca_la_entrada(self):
        imagenes = torch.ones(2, 3, 10, 20)
        tapadas = TaparZona((0.5, 0.6, 1.0, 1.0))(imagenes)
        self.assertTrue((tapadas[..., 6:, 10:] == 0).all())
        self.assertEqual(tapadas.sum().item(), imagenes.sum().item() - 2 * 3 * 4 * 10)
        self.assertTrue((imagenes == 1).all())


if __name__ == "__main__":
    unittest.main()
