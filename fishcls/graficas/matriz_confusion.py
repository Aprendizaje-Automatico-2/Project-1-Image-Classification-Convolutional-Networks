from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from fishcls.evaluacion.metricas import Evaluacion


def matriz_confusion(evaluacion: Evaluacion, titulo: str) -> Figure:
    figura = Figure(figsize=(13, 5.5))
    for eje, normalizar, nombre in zip(figura.subplots(1, 2), (None, "true"), ("Conteos", "Normalizada por clase real")):
        ConfusionMatrixDisplay.from_predictions(
            evaluacion.y_real,
            evaluacion.y_pred,
            labels=range(len(evaluacion.clases)),
            display_labels=evaluacion.clases,
            normalize=normalizar,
            values_format=".2f" if normalizar else "d",
            cmap="Blues",
            colorbar=False,
            xticks_rotation=45,
            ax=eje,
        )
        eje.set_title(nombre)
        eje.set_xlabel("Predicción")
        eje.set_ylabel("Etiqueta real")
    figura.suptitle(f"{titulo} · accuracy {evaluacion.accuracy:.3f} · F1 macro {evaluacion.f1_macro:.3f}")
    figura.tight_layout()
    return figura
