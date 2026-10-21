import tempfile
import unittest
from pathlib import Path

import torch

from fishcls.utilidades import cargar_json, describir_dispositivo, fijar_semilla, guardar_json, obtener_dispositivo


class SemillaTest(unittest.TestCase):
    def test_misma_semilla_mismos_numeros(self):
        fijar_semilla(0)
        primeros = torch.rand(3)
        fijar_semilla(0)
        segundos = torch.rand(3)
        self.assertTrue(torch.equal(primeros, segundos))


class JsonTest(unittest.TestCase):
    def test_ida_y_vuelta_crea_carpetas(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "anidada" / "datos.json"
            guardar_json({"clases": ["Shrimp", "Sea Bass"]}, ruta)
            self.assertEqual(cargar_json(ruta), {"clases": ["Shrimp", "Sea Bass"]})


class DispositivoTest(unittest.TestCase):
    def test_el_dispositivo_ejecuta_tensores(self):
        self.assertEqual(torch.ones(2, device=obtener_dispositivo()).sum().item(), 2)

    def test_descripcion_legible(self):
        self.assertTrue(describir_dispositivo(obtener_dispositivo()))
        self.assertEqual(describir_dispositivo(torch.device("cpu")), "CPU")


if __name__ == "__main__":
    unittest.main()
