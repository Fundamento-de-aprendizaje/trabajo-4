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
