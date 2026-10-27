import unittest

from fishcls.config import DATOS_PREPARADOS, PARTICION, TAMANO_CNN
from fishcls.datos.cargadores import crear_dataloaders
from fishcls.utilidades import cargar_json


@unittest.skipUnless(DATOS_PREPARADOS.exists(), "ejecuta antes: python -m fishcls preparar")
class DataloadersTest(unittest.TestCase):
    def test_repeticiones_multiplican_solo_entrenamiento(self):
        datos = crear_dataloaders(TAMANO_CNN, repeticiones=4, tamano_lote=8)
        entrenamiento = sum(registro["particion"] == "entrenamiento" for registro in cargar_json(PARTICION))
        self.assertEqual(len(datos.entrenamiento.dataset), 4 * entrenamiento)
        self.assertEqual(len(datos.validacion.dataset) + len(datos.test.dataset), 400 - entrenamiento)
        self.assertEqual(len(datos.clases), 8)


if __name__ == "__main__":
    unittest.main()
