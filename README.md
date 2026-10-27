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
python -m fishcls preparar         # agrupa casi duplicados, divide 70/15/15 por grupos y redimensiona a 400×300
python -m fishcls ver-aumentado    # figuras del data augmentation en resultados/figuras/
python -m fishcls entrenar configuraciones/cnn/*.json   # búsqueda manual; acepta uno o varios experimentos
python -m fishcls comparar C00_sin_aumentado C01_base --titulo "Efecto del data augmentation en el sobreajuste" --salida fase2_con_y_sin_aumentado
python -m fishcls resumir 'C.*' --referencia C01_base --titulo "Búsqueda manual" --salida fase3_busqueda_manual
python -m fishcls ajustar --pruebas 30   # Optuna; escribe configuraciones/optuna/O01_optuna_mejor.json
python -m fishcls entrenar configuraciones/optuna/O01_optuna_mejor.json configuraciones/transferencia/*.json
python -m fishcls evaluar C10a_cuatro_bloques_dos_fc O01_optuna_mejor T01c_resnet18_dos_fases T02_convnext_tiny \
    --titulo "Finalistas en test" --salida fase6_test   # una sola vez, al final
python -m fishcls gradcam C10a_cuatro_bloques_dos_fc O01_optuna_mejor T01c_resnet18_dos_fases T02_convnext_tiny \
    --salida fase6_gradcam   # qué mira cada red en test
python -m fishcls tapar-fecha C10a_cuatro_bloques_dos_fc O01_optuna_mejor T01c_resnet18_dos_fases T02_convnext_tiny \
    --titulo "Test con la fecha impresa tapada" --salida fase6_tapar_fecha   # ¿usan las redes la fecha impresa?
python -m unittest discover tests
```

## Datos

- **Origen:** las fotos originales son las de la carpeta `NA_Fish_Dataset` del dataset de Kaggle. Cada clase va en `data/Fish_Dataset/<clase>/`, que no se versiona. Trout se deja fuera, en `data/excluded/`, porque solo tiene 30 imágenes y el enunciado pide al menos 50 por clase.
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
  evaluacion.py   métricas, classification report e intervalo de Wilson
  experimento.py  configuración → datos → modelo → entrenamiento → resultados; recarga y evaluación en test
  ajuste.py       búsqueda automática con Optuna (TPE + poda por mediana) sobre la ConvNet y figuras del estudio
  interpretacion.py  Grad-CAM sobre la última capa convolucional y test con la fecha impresa tapada
  graficas.py     figuras
configuraciones/  un JSON por experimento (cnn/, transferencia/ y optuna/), solo con lo que cambia respecto a los valores por defecto
tests/            unittest
```

Cada experimento guarda en `resultados/experimentos/<nombre>/` su configuración, su historial por época, sus métricas, el mejor modelo, las curvas de pérdida y accuracy y la matriz de confusión de validación; `evaluar` añade las métricas y la matriz de confusión de test. `resultados/resumen.csv` acumula una fila por experimento.
