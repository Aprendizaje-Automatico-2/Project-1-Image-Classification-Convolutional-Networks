import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from fishcls.entrenamiento import ParadaTemprana
from fishcls.experimento import ConfiguracionExperimento, cargar_configuracion, construir_modelo
from fishcls.modelos import ACTIVACIONES

TAMANO_PEQUENO = (48, 64)


def modelo_pequeno(**cambios) -> nn.Module:
    return construir_modelo(ConfiguracionExperimento(nombre="prueba", tamano=TAMANO_PEQUENO, **cambios), num_clases=8)


class ConvNetTest(unittest.TestCase):
    def test_salida_con_todas_las_variantes(self):
        variantes = [{}, {"batchnorm": True}, {"pooling_global": True}, {"neuronas_fc": (64, 32)},
                     {"filtros": (8, 16, 32, 64)}, *({"activacion": nombre} for nombre in ACTIVACIONES)]
        for cambios in variantes:
            with self.subTest(**cambios):
                self.assertEqual(tuple(modelo_pequeno(**cambios)(torch.randn(2, 3, *TAMANO_PEQUENO)).shape), (2, 8))

    def test_perdida_inicial_cercana_a_azar(self):
        torch.manual_seed(0)
        perdida = nn.CrossEntropyLoss()(modelo_pequeno()(torch.randn(64, 3, *TAMANO_PEQUENO)), torch.randint(0, 8, (64,)))
        self.assertAlmostEqual(perdida.item(), math.log(8), delta=0.3)

    def test_sobreajusta_un_lote(self):
        torch.manual_seed(0)
        modelo = modelo_pequeno(dropout=0.0)
        imagenes, etiquetas = torch.randn(8, 3, *TAMANO_PEQUENO), torch.arange(8)
        optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
        for _ in range(100):
            optimizador.zero_grad()
            perdida = nn.CrossEntropyLoss()(modelo(imagenes), etiquetas)
            perdida.backward()
            optimizador.step()
        self.assertLess(perdida.item(), 0.05)


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


class ConfiguracionTest(unittest.TestCase):
    def test_json_solo_sobrescribe_lo_indicado(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "C99_prueba.json"
            ruta.write_text('{"filtros": [8, 16], "aumentado": "ninguno"}')
            config = cargar_configuracion(ruta)
        self.assertEqual(config.nombre, "C99_prueba")
        self.assertEqual(config.filtros, (8, 16))
        self.assertEqual(config.aumentado, "ninguno")
        self.assertEqual(config.tasa_aprendizaje, ConfiguracionExperimento(nombre="x").tasa_aprendizaje)


if __name__ == "__main__":
    unittest.main()
