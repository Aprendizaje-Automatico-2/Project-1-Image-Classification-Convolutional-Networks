import pandas as pd
from matplotlib.figure import Figure

from fishcls.graficas.comun import COLORES


def barras_experimentos(tabla: pd.DataFrame, referencia: str, titulo: str) -> Figure:
    figura = Figure(figsize=(12, 0.4 * len(tabla) + 1.5))
    for eje, columna, nombre in zip(figura.subplots(1, 2, sharey=True), ("acc_val", "perdida_val"),
                                    ("Accuracy de validación", "Pérdida de validación")):
        colores = ["tab:red" if experimento == referencia else "tab:blue" for experimento in tabla["nombre"]]
        eje.barh(tabla["nombre"], tabla[columna], color=colores)
        eje.axvline(tabla.loc[tabla["nombre"] == referencia, columna].item(), color="tab:red", linestyle="--")
        for posicion, valor in enumerate(tabla[columna]):
            eje.text(valor, posicion, f" {valor:.3f}", va="center", fontsize=8)
        eje.set_title(nombre)
        eje.invert_yaxis()
        eje.grid(True, axis="x", alpha=0.3)
    figura.suptitle(titulo)
    return figura


def accuracy_frente_a_parametros(
    tabla: pd.DataFrame, titulo: str, columna: str = "acc_val", etiqueta: str = "Accuracy de validación"
) -> Figure:
    figura = Figure(figsize=(8, 5))
    eje = figura.subplots()
    errores = None
    if "ic_inferior" in tabla:
        errores = [tabla[columna] - tabla["ic_inferior"], tabla["ic_superior"] - tabla[columna]]
    eje.errorbar(tabla["parametros"], tabla[columna], yerr=errores, fmt="o", markersize=8, capsize=4, color="tab:blue")
    for _, fila in tabla.iterrows():
        eje.annotate(fila["nombre"].split("_")[0], (fila["parametros"], fila[columna]), textcoords="offset points",
                     xytext=(6, 4), fontsize=8)
    eje.set_xscale("log")
    eje.set_xlabel("Parámetros (escala logarítmica)")
    eje.set_ylabel(etiqueta)
    eje.set_title(titulo)
    eje.grid(True, alpha=0.3)
    return figura


def barras_ablacion(tabla: pd.DataFrame, titulo: str) -> Figure:
    figura = Figure(figsize=(13, 4.5), layout="constrained")
    for eje, (fotos, grupo) in zip(figura.subplots(1, 2, sharey=True), tabla.groupby("fotos", sort=False)):
        barras = grupo.pivot(index="nombre", columns="condicion", values="acc_test")
        barras = barras.reindex(index=grupo["nombre"].unique(), columns=grupo["condicion"].unique())
        barras.rename(index=lambda nombre: nombre.split("_")[0]).plot.bar(ax=eje, rot=0, color=COLORES, legend=False)
        for barra in eje.containers:
            eje.bar_label(barra, fmt="%.2f", fontsize=7)
        eje.set_title(f"Fotos {fotos} ({grupo['imagenes'].iloc[0]} imágenes de test)")
        eje.set_xlabel("")
        eje.set_ylabel("Accuracy de test")
        eje.grid(True, axis="y", alpha=0.3)
    figura.legend(*eje.get_legend_handles_labels(), loc="outside lower center", ncols=3, fontsize=9)
    figura.suptitle(titulo)
    return figura
