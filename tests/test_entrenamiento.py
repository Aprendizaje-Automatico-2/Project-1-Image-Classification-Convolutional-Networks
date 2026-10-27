import unittest

import torch
import torch.nn as nn

from fishcls.entrenamiento.parada_temprana import ParadaTemprana


class ParadaTempranaTest(unittest.TestCase):
    def test_para_tras_la_paciencia_y_guarda_el_mejor_estado(self):
        modelo = nn.Linear(1, 1)
        parada = ParadaTemprana(paciencia=2)
        decisiones = []
        for epoca, perdida in enumerate([1.0, 0.5, 0.6, 0.4, 0.7, 0.8], start=1):
            with torch.no_grad():
                modelo.weight.fill_(epoca)
            decisiones.append(parada(perdida, modelo, epoca))
        self.assertEqual(decisiones, [False, False, False, False, False, True])
        self.assertEqual(parada.mejor_epoca, 4)
        self.assertEqual(parada.mejor_estado["weight"].item(), 4)

    def test_reiniciar_paciencia_conserva_el_mejor_estado(self):
        modelo = nn.Linear(1, 1)
        parada = ParadaTemprana(paciencia=2)
        parada(0.5, modelo, 1)
        parada(0.9, modelo, 2)
        parada.reiniciar_paciencia()
        self.assertFalse(parada(0.9, modelo, 3))
        self.assertEqual((parada.mejor_epoca, parada.mejor_perdida_val), (1, 0.5))


if __name__ == "__main__":
    unittest.main()
