# Clasificación de peces con redes convolucionales

Práctica 1 de Aprendizaje Automático II (2026-27): clasificador de imágenes en PyTorch sobre 8 especies del [A Large-Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) (Ulucan et al., ASYU 2020). Se usan solo las 400 fotos originales, no las 9000 aumentadas: esas son 20 rotaciones de cada foto y filtrarían el test.

## Entorno

```bash
conda activate Aprendizaje_Automatico
python -m fishcls info
```

Detecta automáticamente NVIDIA CUDA, AMD ROCm, Apple Silicon (MPS), Intel XPU o CPU. Las versiones exactas están en [`requirements.txt`](requirements.txt).

## Uso

```bash
python -m fishcls preparar         # agrupa casi duplicados, divide 70/15/15 por grupos y redimensiona a 300×400
python -m fishcls ver-aumentado    # figuras del data augmentation en resultados/figuras/
python -m fishcls entrenar configuraciones/cnn/C01_base.json   # uno o varios experimentos
python -m fishcls comparar C00_sin_aumentado C01_base --titulo "Con y sin DA" --salida comparativa
python -m fishcls resumir 'C.*' --referencia C01_base --titulo "Búsqueda manual" --salida busqueda
python -m fishcls entrenar configuraciones/transferencia/T02_convnext_tiny.json   # transfer learning
python -m fishcls ajustar --pruebas 30   # Optuna; escribe configuraciones/optuna/O01_optuna_mejor.json
python -m unittest discover tests
```

## Datos

- **Origen:** las imágenes originales van en `data/Fish_Dataset/<clase>/` y no se versionan. Trout queda excluida porque solo tiene 30 imágenes y el enunciado pide al menos 50 por clase.
- **Partición:** [`particion.json`](particion.json) guarda, para cada imagen, su grupo de casi duplicados y su partición. Ningún grupo aparece en dos particiones.
- **Data augmentation:** se aplica al vuelo y solo a entrenamiento.

## Estructura

```
fishcls/
  config.py       rutas y constantes
  utilidades.py   dispositivo, semillas, JSON
  datos.py        catálogo, agrupado, partición, preparación y DataLoaders
  aumentado.py    transformaciones de entrenamiento y evaluación
  modelos.py      ConvNet configurable (parte de la red de la asignatura) y ResNet18 / ConvNeXt-Tiny preentrenadas
  entrenamiento.py  ParadaTemprana, bucle de entrenamiento y validación
  evaluacion.py   métricas y classification report
  experimento.py  configuración → datos → modelo → entrenamiento → resultados
  ajuste.py       búsqueda automática con Optuna (TPE + poda por mediana) sobre la ConvNet
  graficas.py     figuras
configuraciones/  un JSON por experimento (cnn/, transferencia/ y optuna/), solo con lo que cambia respecto a los valores por defecto
tests/            unittest
```

Cada experimento guarda en `resultados/experimentos/<nombre>/` su configuración, su historial por época, sus métricas, el mejor modelo, las curvas de pérdida y accuracy y la matriz de confusión de validación. `resultados/resumen.csv` acumula una fila por experimento.
