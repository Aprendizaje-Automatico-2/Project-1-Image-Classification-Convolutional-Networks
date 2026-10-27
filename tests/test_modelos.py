import math
import unittest

import torch
import torch.nn as nn

from fishcls.experimentos.configuracion import ConfiguracionExperimento, construir_modelo
from fishcls.modelos.convnet import ACTIVACIONES
from fishcls.modelos.preentrenados import PREENTRENADOS, congelar_extractor, modelo_preentrenado, parametros_cabeza

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


class PreentrenadosTest(unittest.TestCase):
    def test_salida_de_ocho_clases_y_congelado_solo_deja_la_cabeza(self):
        for nombre in PREENTRENADOS:
            with self.subTest(modelo=nombre):
                modelo = modelo_preentrenado(nombre, num_clases=8, con_pesos=False)
                self.assertEqual(tuple(modelo.eval()(torch.randn(1, 3, 64, 96)).shape), (1, 8))
                congelar_extractor(modelo, nombre, congelar=True)
                entrenables = [p for p in modelo.parameters() if p.requires_grad]
                self.assertEqual({id(p) for p in entrenables}, {id(p) for p in parametros_cabeza(modelo, nombre)})
                congelar_extractor(modelo, nombre, congelar=False)
                self.assertTrue(all(p.requires_grad for p in modelo.parameters()))


if __name__ == "__main__":
    unittest.main()
