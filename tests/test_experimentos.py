import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from fishcls.experimentos.configuracion import ConfiguracionExperimento, cargar_configuracion, construir_modelo
from fishcls.experimentos.ejecucion import crear_optimizador, fases_entrenamiento
from fishcls.experimentos.resultados import cargar_modelo
from fishcls.modelos.preentrenados import modelo_preentrenado
from fishcls.utilidades import guardar_json

TAMANO_PEQUENO = (48, 64)


class EjecucionTest(unittest.TestCase):
    def test_ajuste_fino_usa_tasa_reducida_en_el_extractor(self):
        config = ConfiguracionExperimento(nombre="t", modelo="resnet18", tasa_aprendizaje=1e-3, factor_lr_extractor=0.1)
        optimizador = crear_optimizador(modelo_preentrenado("resnet18", 8, con_pesos=False), config)
        self.assertEqual([grupo["lr"] for grupo in optimizador.param_groups], [1e-3, 1e-4])

    def test_fases_por_estrategia(self):
        base = {"nombre": "t", "epocas": 30, "epocas_congelado": 5}
        self.assertEqual(fases_entrenamiento(ConfiguracionExperimento(**base, estrategia="congelado")), [(True, 30)])
        self.assertEqual(fases_entrenamiento(ConfiguracionExperimento(**base, estrategia="dos_fases")),
                         [(True, 5), (False, 30)])


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
