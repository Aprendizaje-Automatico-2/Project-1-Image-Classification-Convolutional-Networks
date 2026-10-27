import unittest

import torch
import torch.nn as nn

from fishcls.evaluacion.fecha import TaparZona
from fishcls.evaluacion.gradcam import capa_gradcam, mapa_gradcam
from fishcls.experimentos.configuracion import ConfiguracionExperimento, construir_modelo
from fishcls.modelos.preentrenados import PREENTRENADOS, modelo_preentrenado


def salida_de(capa: nn.Module, modelo: nn.Module, imagenes: torch.Tensor) -> torch.Tensor:
    salidas = []
    gancho = capa.register_forward_hook(lambda _modulo, _entrada, salida: salidas.append(salida))
    modelo.eval()(imagenes)
    gancho.remove()
    return salidas[0]


class CapaGradcamTest(unittest.TestCase):
    def test_la_capa_elegida_tiene_salida_espacial(self):
        convnet = construir_modelo(ConfiguracionExperimento(nombre="t", tamano=(48, 64), filtros=(8, 16, 32)), 8)
        self.assertEqual(tuple(salida_de(capa_gradcam(convnet, "convnet"), convnet, torch.randn(1, 3, 48, 64)).shape),
                         (1, 32, 6, 8))
        for nombre in PREENTRENADOS:
            with self.subTest(modelo=nombre):
                modelo = modelo_preentrenado(nombre, 8, con_pesos=False)
                self.assertEqual(salida_de(capa_gradcam(modelo, nombre), modelo, torch.randn(1, 3, 64, 64)).dim(), 4)


class MapaGradcamTest(unittest.TestCase):
    def setUp(self):
        self.capa = nn.Conv2d(2, 2, kernel_size=1)
        lineal = nn.Linear(2, 2)
        with torch.no_grad():
            self.capa.weight.copy_(torch.eye(2).view(2, 2, 1, 1))
            self.capa.bias.zero_()
            lineal.weight.copy_(torch.eye(2))
            lineal.bias.zero_()
        self.modelo = nn.Sequential(self.capa, nn.AdaptiveAvgPool2d(1), nn.Flatten(), lineal)
        self.imagen = torch.zeros(2, 4, 4)
        self.imagen[0, :, :2] = 1
        self.imagen[1, :, 2:] = 1

    def test_cada_clase_ilumina_la_zona_de_la_que_depende(self):
        izquierda = mapa_gradcam(self.modelo, self.capa, self.imagen, clase=0)
        derecha = mapa_gradcam(self.modelo, self.capa, self.imagen, clase=1)
        self.assertEqual(izquierda.shape, (4, 4))
        self.assertTrue((izquierda[:, :2] == 1).all() and (izquierda[:, 2:] == 0).all())
        self.assertTrue((derecha[:, :2] == 0).all() and (derecha[:, 2:] == 1).all())


class TaparZonaTest(unittest.TestCase):
    def test_pone_a_cero_solo_la_zona_y_no_toca_la_entrada(self):
        imagenes = torch.ones(2, 3, 10, 20)
        tapadas = TaparZona((0.5, 0.6, 1.0, 1.0))(imagenes)
        self.assertTrue((tapadas[..., 6:, 10:] == 0).all())
        self.assertEqual(tapadas.sum().item(), imagenes.sum().item() - 2 * 3 * 4 * 10)
        self.assertTrue((imagenes == 1).all())


if __name__ == "__main__":
    unittest.main()
