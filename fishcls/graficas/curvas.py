from matplotlib.figure import Figure

from fishcls.graficas.comun import COLORES


def figura_perdida_y_accuracy(titulo: str) -> tuple[Figure, list[tuple]]:
    figura = Figure(figsize=(12, 4.5))
    figura.suptitle(titulo)
    return figura, list(zip(figura.subplots(1, 2), ("perdida", "acc"), ("Pérdida", "Accuracy")))


def formatear_eje(eje, titulo: str, nombre: str) -> None:
    eje.set_title(titulo, fontsize=10)
    eje.set_xlabel("Época")
    eje.set_ylabel(nombre)
    eje.legend(fontsize=8)
    eje.grid(True, alpha=0.3)


def curvas_entrenamiento(historial: dict[str, list[float]], mejor_epoca: int, titulo: str) -> Figure:
    figura, paneles = figura_perdida_y_accuracy(titulo)
    epocas = range(1, len(historial["perdida_train"]) + 1)
    for eje, metrica, nombre in paneles:
        eje.plot(epocas, historial[f"{metrica}_train"], label="Entrenamiento")
        eje.plot(epocas, historial[f"{metrica}_val"], label="Validación")
        eje.axvline(mejor_epoca, color="red", linestyle="--", label=f"Mejor época ({mejor_epoca})")
        formatear_eje(eje, f"Evolución de la {nombre.lower()}", nombre)
    return figura


def comparar_historiales(historiales: dict[str, dict[str, list[float]]], titulo: str) -> Figure:
    figura, paneles = figura_perdida_y_accuracy(titulo)
    for eje, metrica, nombre in paneles:
        for (experimento, historial), color in zip(historiales.items(), COLORES):
            epocas = range(1, len(historial[f"{metrica}_val"]) + 1)
            eje.plot(epocas, historial[f"{metrica}_train"], color=color, linestyle=":", alpha=0.7)
            eje.plot(epocas, historial[f"{metrica}_val"], color=color, label=experimento)
        formatear_eje(eje, f"{nombre} (continua: validación · punteada: entrenamiento)", nombre)
    return figura
