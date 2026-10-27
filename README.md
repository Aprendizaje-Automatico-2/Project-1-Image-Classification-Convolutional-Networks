# Clasificación de peces con redes convolucionales

Práctica 1 de Aprendizaje Automático II (2026-27): clasificador de imágenes en PyTorch sobre 8 especies del [A Large-Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) (Ulucan et al., ASYU 2020). Se usan solo las 400 fotos originales, no las 9000 aumentadas: esas son 20 rotaciones de cada foto y filtrarían el test.

## Entorno

```bash
pip install -r requirements.txt   # con Python 3.14
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
  __main__.py            argumentos de cada comando
  comandos.py            lo que hace cada comando
  config.py              rutas y constantes
  utilidades.py          dispositivo, semillas y JSON
  preparacion/           de las fotos originales a data/preparados/ y particion.json (comando preparar)
    imagenes.py          listar, abrir con la orientación EXIF en RGB y reducir a 400×300
    duplicados.py        huella perceptual y grupos de casi duplicados
    particion.py         partición 70/15/15 estratificada y por grupos
    preparar.py          ejecuta los pasos anteriores y resume la partición
  datos/                 de las imágenes preparadas a los DataLoader
    aumentado.py         data augmentation de entrenamiento y transformaciones de evaluación
    cargadores.py        DataLoader de entrenamiento, validación y test
  modelos/
    convnet.py           ConvNet configurable (parte de la red de la asignatura)
    preentrenados.py     ResNet18 y ConvNeXt-Tiny con cabeza nueva y extractor congelable
    medidas.py           número de parámetros y milisegundos por imagen
  entrenamiento/
    parada_temprana.py   ParadaTemprana
    bucle.py             bucle de entrenamiento y validación
  experimentos/
    configuracion.py     ConfiguracionExperimento, su JSON y el modelo que describe
    ejecucion.py         entrena un experimento de principio a fin
    resultados.py        guarda y carga los resultados y resultados/resumen.csv
    ajuste.py            búsqueda con Optuna (TPE y poda por mediana) y sus figuras
  evaluacion/
    metricas.py          accuracy, F1 macro, classification report e intervalo de Wilson
    finalistas.py        evaluación en test
    gradcam.py           Grad-CAM sobre la última capa convolucional
    fecha.py             test con la fecha impresa tapada
  graficas/
    comun.py             guardar figuras y mostrar imágenes
    curvas.py            curvas de pérdida y accuracy
    comparativas.py      barras y accuracy frente a parámetros
    imagenes.py          data augmentation y cuadrículas de Grad-CAM
    matriz_confusion.py  matriz de confusión
configuraciones/         un JSON por experimento (cnn/, transferencia/ y optuna/), solo con lo que cambia
tests/                   un fichero de unittest por módulo
```

Cada experimento guarda en `resultados/experimentos/<nombre>/` su configuración, su historial por época, sus métricas, el mejor modelo, las curvas de pérdida y accuracy y la matriz de confusión de validación; `evaluar` añade las métricas y la matriz de confusión de test. `resultados/resumen.csv` acumula una fila por experimento.
