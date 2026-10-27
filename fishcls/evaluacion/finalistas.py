from fishcls.config import EXPERIMENTOS
from fishcls.evaluacion.metricas import evaluar
from fishcls.experimentos.resultados import cargar_experimento
from fishcls.graficas.comun import guardar_figura
from fishcls.graficas.matriz_confusion import matriz_confusion
from fishcls.modelos.medidas import contar_parametros
from fishcls.utilidades import guardar_json, obtener_dispositivo


def evaluar_en_test(nombre: str) -> dict:
    carpeta = EXPERIMENTOS / nombre
    dispositivo = obtener_dispositivo()
    _, modelo, datos = cargar_experimento(nombre, dispositivo)
    evaluacion = evaluar(modelo, datos.test, dispositivo, datos.clases)
    print(f"\n=== Classification report: TEST ({nombre}) ===\n{evaluacion.informe()}")
    inferior, superior = evaluacion.intervalo_wilson
    resumen = {
        "nombre": nombre,
        "acc_test": round(evaluacion.accuracy, 4),
        "ic_inferior": round(inferior, 4),
        "ic_superior": round(superior, 4),
        "f1_test": round(evaluacion.f1_macro, 4),
        "imagenes": len(evaluacion.y_real),
        "parametros": contar_parametros(modelo),
    }
    guardar_json({**resumen, "informe_test": evaluacion.informe(como_diccionario=True)}, carpeta / "metricas_test.json")
    guardar_figura(matriz_confusion(evaluacion, f"{nombre} · test"), "matriz_confusion_test.png", carpeta)
    return resumen
