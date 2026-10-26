import tempfile
import unittest
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fishcls.ajuste import configuracion_sugerida, informar_a_optuna
from fishcls.entrenamiento import ParadaTemprana, entrenar
from fishcls.experimento import cargar_configuracion, guardar_configuracion
from fishcls.utilidades import cargar_json

PARAMETROS = {
    "bloques": 4,
    "filtros_base": 16,
    "capas_fc": 2,
    "neuronas_fc": 256,
    "tamano_kernel": 5,
    "activacion": "relu",
    "dropout": 0.25,
    "batchnorm": False,
    "tasa_aprendizaje": 1e-3,
    "regularizacion_l2": 1e-4,
    "tamano_lote": 32,
}


class ConfiguracionSugeridaTest(unittest.TestCase):
    def test_traduce_bloques_y_capas_a_tuplas(self):
        config = configuracion_sugerida(optuna.trial.FixedTrial(PARAMETROS), "O99")
        self.assertEqual(config.filtros, (16, 32, 64, 128))
        self.assertEqual(config.neuronas_fc, (256, 128))

    def test_el_json_solo_guarda_cambios_y_se_recarga_igual(self):
        config = configuracion_sugerida(optuna.trial.FixedTrial(PARAMETROS), "O99_prueba")
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "O99_prueba.json"
            guardar_configuracion(config, ruta)
            self.assertEqual(set(cargar_json(ruta)), {"filtros", "neuronas_fc", "regularizacion_l2"})
            self.assertEqual(cargar_configuracion(ruta), config)


class AlTerminarEpocaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        datos = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
        self.loader = DataLoader(datos, batch_size=8)
        self.modelo = nn.Linear(4, 2)

    def entrenar_tres_epocas(self, al_terminar_epoca):
        return entrenar(
            self.modelo, self.loader, self.loader, torch.optim.SGD(self.modelo.parameters(), lr=0.1),
            nn.CrossEntropyLoss(), ParadaTemprana(paciencia=10), torch.device("cpu"), epocas=3,
            al_terminar_epoca=al_terminar_epoca,
        )

    def test_se_llama_una_vez_por_epoca(self):
        llamadas = []
        self.entrenar_tres_epocas(lambda epoca, perdida: llamadas.append(epoca))
        self.assertEqual(llamadas, [1, 2, 3])

    def test_optuna_poda_cuando_el_podador_lo_decide(self):
        estudio = optuna.create_study(pruner=optuna.pruners.ThresholdPruner(upper=-1.0))
        with self.assertRaises(optuna.TrialPruned):
            self.entrenar_tres_epocas(informar_a_optuna(estudio.ask()))


if __name__ == "__main__":
    unittest.main()
