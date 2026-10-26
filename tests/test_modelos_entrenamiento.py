import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from fishcls.entrenamiento import ParadaTemprana
from fishcls.experimento import (
    ConfiguracionExperimento,
    cargar_configuracion,
    cargar_modelo,
    construir_modelo,
    crear_optimizador,
    fases_entrenamiento,
)
from fishcls.modelos import ACTIVACIONES, PREENTRENADOS, congelar_extractor, modelo_preentrenado, parametros_cabeza
from fishcls.utilidades import guardar_json

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

    def test_ajuste_fino_usa_tasa_reducida_en_el_extractor(self):
        config = ConfiguracionExperimento(nombre="t", modelo="resnet18", tasa_aprendizaje=1e-3, factor_lr_extractor=0.1)
        optimizador = crear_optimizador(modelo_preentrenado("resnet18", 8, con_pesos=False), config)
        self.assertEqual([grupo["lr"] for grupo in optimizador.param_groups], [1e-3, 1e-4])

    def test_fases_por_estrategia(self):
        base = {"nombre": "t", "epocas": 30, "epocas_congelado": 5}
        self.assertEqual(fases_entrenamiento(ConfiguracionExperimento(**base, estrategia="congelado")), [(True, 30)])
        self.assertEqual(fases_entrenamiento(ConfiguracionExperimento(**base, estrategia="dos_fases")),
                         [(True, 5), (False, 30)])


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

    def test_recarga_un_experimento_guardado(self):
        torch.manual_seed(0)
        config = ConfiguracionExperimento(nombre="C99_prueba", tamano=TAMANO_PEQUENO, filtros=(8, 16))
        modelo = construir_modelo(config, num_clases=8).eval()
        with tempfile.TemporaryDirectory() as temporal:
            carpeta = Path(temporal)
            guardar_json(asdict(config), carpeta / "configuracion.json")
            torch.save(modelo.state_dict(), carpeta / "modelo.pt")
            recargada = cargar_configuracion(carpeta / "configuracion.json")
            recargado = cargar_modelo(carpeta, recargada, 8, torch.device("cpu")).eval()
        imagenes = torch.randn(2, 3, *TAMANO_PEQUENO)
        self.assertEqual(recargada, config)
        self.assertTrue(torch.equal(recargado(imagenes), modelo(imagenes)))


if __name__ == "__main__":
    unittest.main()
