# ------------------------- FUNCIÓN 4 ------------------------- #
def guardar_csv(df, nombre_archivo):
    """Guarda el DataFrame en un archivo CSV"""
    df.to_csv(nombre_archivo, index=False)
    print(f"\n📁 Archivo '{nombre_archivo}' guardado con éxito.")

guardar_csv(df_transformado, 'vino_transformado.csv')
# Ejecutar



def knn_predict(x_entrenamiento, y_entrenamiento, x_prueba, k, ponderado=False):
    """
    Predice la clase de cada ejemplo en x_prueba usando KNN.
    Si ponderado=False: voto simple.
    Si ponderado=True: voto ponderado con peso = 1/d^2.
    """
    # 1) Estandarizar datos
    x_train_std, x_test_std = estandarizar_datos(x_entrenamiento, x_prueba)
    
    predicciones = []
    for u in x_test_std:
        # 2) Calcular distancias
        distancias = [
            (calcular_distancia(u, x_train_std[j]), y_entrenamiento[j])
            for j in range(len(x_train_std))
        ]
        distancias.sort(key=lambda t: t[0])
        vecinos = distancias[:k]
        
        if not ponderado:
            # Voto simple
            clases = [et for _, et in vecinos]
            predicciones.append(Counter(clases).most_common(1)[0][0])
        else:
            # Voto ponderado
            pesos = {}
            for d, et in vecinos:
                d2 = d**2
                peso = 1/(d2 + 1e-8)
                pesos[et] = pesos.get(et, 0) + peso
            predicciones.append(max(pesos, key=pesos.get))
    
    return np.array(predicciones)


def evaluar_clusters(asignaciones, y_reales):
    """
    Evalúa los clusters comparando contra las clases reales (y_reales).
    """
    etiquetas_unicas = np.unique(y_reales)
    clusters_unicos = np.unique(asignaciones)
    
    # Mapeamos cada cluster a la clase mayoritaria dentro de él
    mapeo = {}
    for c in clusters_unicos:
        clases_en_cluster = y_reales[asignaciones == c]
        if len(clases_en_cluster) == 0:
            mapeo[c] = None
        else:
            clase_mayoritaria = Counter(clases_en_cluster).most_common(1)[0][0]
            mapeo[c] = clase_mayoritaria

    # Predecimos usando el mapeo
    predichos = np.array([mapeo[c] for c in asignaciones])
    tasa_aciertos = calcular_tasa_aciertos(y_reales, predichos)
    return tasa_aciertos, mapeo
    # Evaluar qué tan bien se alinean los clusters con las clases reales
    tasa, mapeo_clusters = evaluar_clusters(asignaciones_entrenamiento, y_entrenamiento)