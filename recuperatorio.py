# Librerías
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------- FUNCIÓN 1 ------------------------- #

def descargar_dataset(url):
    """Convierte el enlace de Google Drive en uno de descarga directa y carga el CSV completo"""
    return pd.read_csv(url, encoding='latin1')


# ------------------------- Ej 1 - PUNTO 1 ------------------------- #
def describir_dataset(df):
    """Muestra información general del dataset"""
    print(f"\nCantidad de registros: {df.shape[0]}")
    print(f"Cantidad de columnas: {df.shape[1]}\n")

    print("Tipos de datos por columna:")
    # print(df.dtypes)
    for col in df.columns:
        if("quality" != col):
            print(col,": ",df[col].dtype, "||", "Numerica continua")
        else:
            print(col,": ",df[col].dtype, "||", "Categorica ordinal")

    print("\nCantidad de valores faltantes por columna:")
    print(df.isnull().sum())

    if df.isnull().sum().sum() > 0:
        print("\n Se encontraron valores faltantes.")
        print("\n En este caso se podria calcular el promedio para los datos faltantes de las columnas numéricas " \
        "o la Moda para las Categóricas ")
    else:
        print("\n No se encontraron valores faltantes.")

# ------------------------- Ej 1 - Punto 2 ------------------------- #

def transformar_quality(df):
    """
    Elimina registros con quality = 6 y crea nueva columna categórica 'objetivo'
    con valores: REGULAR, BUENO, EXCELENTE
    """
    df_filtrado = df[df['quality'] != 6].copy()

    print("Cantidad de datos sin calidad igual a 6 : ",df_filtrado.shape[0])

    df_filtrado['objetivo'] = df_filtrado['quality'].apply(clasificar)

    return df_filtrado

def clasificar(x):
        if x < 5:
            return "REGULAR"
        elif x == 5:
            return "BUENO"
        else:
            return "EXCELENTE"

# ------------------------- Ej 2 - Punto 1 ------------------------- #

# === 2. DIVISIÓN ENTRE ENTRENAMIENTO Y PRUEBA ===#

def dividir_entrenamiento_prueba(x, y, prueba_size=0.2, random_state=100):
    np.random.seed(random_state)
    indices = np.random.permutation(len(x))

    n_train = int(len(x) * (1 - prueba_size))
    x_entrenamiento = x[indices[:n_train]]
    x_prueba = x[indices[n_train:]]
    y_entrenamiento = y[indices[:n_train]]
    y_prueba = y[indices[n_train:]]
    print(f"Entrenamiento: {len(x_entrenamiento)}, Prueba: {len(x_prueba)}")
    return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba
# ------------------------- Ej 2 - Punto 2 ------------------------- #
def calcular_distancia(x1, x2):
    """
    Calcula la distancia euclidiana entre dos vectores (dos filas de datos).
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def estandarizar_datos(x_entrenamiento, x_prueba):
    """
    Estandariza los datos usando media y desvío del entrenamiento.
    """
    medias = x_entrenamiento.mean(axis=0)
   # print("medias",medias)
    desvios = x_entrenamiento.std(axis=0)

    x_entrenamiento_std = (x_entrenamiento - medias) / desvios
    x_prueba_std = (x_prueba - medias) / desvios

    return x_entrenamiento_std, x_prueba_std

# ------------------- 2. FUNCIÓN PRINCIPAL DE KNN -------------------
def knn_predict(x_entrenamiento, y_entrenamiento, x_prueba, k):
    """
    Predice la clase de cada ejemplo en x_prueba usando KNN con voto simple.
    """
    predicciones = []
     #estandarizar datos
    x_entrenamiento_std, x_prueba_std = estandarizar_datos(x_entrenamiento, x_prueba)
    for i in range(len(x_prueba)):
        distancias = []
        # Calcular la distancia de este x_prueba[i] con cada x_entrenamiento[j]
        for j in range(len(x_entrenamiento)):
            distancia = calcular_distancia(x_prueba_std[i], x_entrenamiento_std[j])
            etiqueta = y_entrenamiento[j]
            distancias.append((distancia, etiqueta))
        # Ordenar las distancias de menor a mayor
        distancias.sort(key=lambda tupla: tupla[0])
        # Tomar los k vecinos más cercanos
        vecinos = distancias[:k]
        # Votar la clase más frecuente
        clases = [etiqueta for _, etiqueta in vecinos]
        clase_mas_comun = Counter(clases).most_common(1)[0][0]
        # Guardar predicción
        predicciones.append(clase_mas_comun)

    return np.array(predicciones)


######## KNN PONDERADO ########

def knn_predict_ponderado(x_entrenamiento, y_entrenamiento, x_prueba, k):
    """
    Predice la clase usando k vecinos ponderados con peso = 1/d^2.
    """
    predicciones = []
    # estandariza igual que en knn_predict
    x_entrenamiento_std, x_prueba_std = estandarizar_datos(x_entrenamiento, x_prueba)

    for i in range(len(x_prueba_std)):
        # calculo distancias al cuadrado
        distancias = []
        for j in range(len(x_entrenamiento_std)):
            distanciaAlCuadrado = calcular_distancia(x_prueba_std[i], x_entrenamiento_std[j])**2
            etiqueta = y_entrenamiento[j]
            distancias.append((distanciaAlCuadrado, etiqueta))


        distancias.sort(key=lambda t: t[0])
        vecinos = distancias[:k]
        # acumula peso por clase
        pesos = {}
        for distanciaAlCuadrado, etiqueta in vecinos:
            peso = 1/(distanciaAlCuadrado if distanciaAlCuadrado > 0 else 1e-8)
            pesos[etiqueta] = pesos.get(etiqueta, 0) + peso
        # elige la clase de mayor peso
        predicciones.append(max(pesos, key=pesos.get))

    return np.array(predicciones)


# ------------------- 3. FUNCIÓN PARA CALCULAR OBSERVACIONES BIEN ETI -------------------
def calcular_tasa_aciertos(y_real, y_predicho):
    """
    Calcula el porcentaje de predicciones correctas.
    """
    return np.mean(y_real == y_predicho)

## PUNTO 3 ####################################################################
def k_means_cluster_random(x, k, max_iter=100):
    """
    Variante de K-Means: inicialización aleatoria por asignación de clusters (según teoría).
    x: datos de entrada (matriz numpy)
    k: cantidad de clusters
    max_iter: número máximo de iteraciones permitidas
    """
    contador_de_iteraciones = 0  # Inicializa un contador para mostrar en qué iteración se detiene el algoritmo

    np.random.seed(45)  # Fija una semilla aleatoria para que los resultados sean reproducibles

    # Paso 1: asignación inicial aleatoria
    # Asigna un número de cluster (de 0 a k-1) a cada observación del conjunto x
    asignaciones = np.random.randint(0, k, size=len(x))

    # Comienza el ciclo de iteración (algoritmo iterativo)
    for _ in range(max_iter):
        contador_de_iteraciones = contador_de_iteraciones + 1  # Cuenta la cantidad de iteraciones
        centroides = []  # Lista vacía donde se almacenarán los centroides actualizados
        # Paso 2.1: calcular el nuevo centroide de cada cluster
        for i in range(k):
            # Selecciona todos los puntos que fueron asignados al cluster i
            puntos_cluster = x[asignaciones == i]
            if len(puntos_cluster) > 0:
                # Calcula el centroide como el promedio de los puntos del cluster
                centroides.append(puntos_cluster.mean(axis=0))
            else:
                # Si un cluster no tiene puntos asignados, elige aleatoriamente un nuevo punto como centroide
                centroides.append(x[np.random.randint(0, len(x))])

        # Convierte la lista de centroides en un array numpy para facilitar cálculos posteriores
        centroides = np.array(centroides)

        # Paso 2.2: asignar cada observación al centroide más cercano
        nuevas_asignaciones = []  # Lista para guardar las nuevas asignaciones de cada punto

        for fila in x:
            # Calcula la distancia de esta observación a cada centroide
            distancias = [calcular_distancia(fila, centroide) for centroide in centroides]

            # Busca el índice del centroide más cercano (mínima distancia)
            cluster_cercano = np.argmin(distancias)

            # Asigna este punto al cluster más cercano
            nuevas_asignaciones.append(cluster_cercano)

        # Convierte la lista a array para poder comparar con las asignaciones anteriores
        nuevas_asignaciones = np.array(nuevas_asignaciones)

        # Verifica si las asignaciones no cambiaron respecto a la iteración anterior
        if np.array_equal(asignaciones, nuevas_asignaciones):
            break  # Si no hubo cambios, se alcanzó la convergencia y se sale del ciclo

        # Si hubo cambios, actualiza las asignaciones para la próxima iteración
        asignaciones = nuevas_asignaciones

    # Muestra cuántas iteraciones fueron necesarias para llegar a la convergencia
    print(f"Iteración {contador_de_iteraciones} de K-Means con k = {k}")

    # Retorna las asignaciones finales y los centroides encontrados
    return asignaciones, centroides



def accuracy_clusters_vs_reales(y_reales, asignaciones, k):
    from itertools import permutations
    mejor_accuracy = 0
    mejores_labels = None
    for perm in permutations(range(k)):
        reetiquetado = np.array([perm[i] for i in asignaciones])
        acc = np.mean(reetiquetado == y_reales)
        if acc > mejor_accuracy:
            mejor_accuracy = acc
            mejores_labels = reetiquetado
    return mejor_accuracy, mejores_labels

def codificar_etiquetas(etiquetas):
    clases = sorted(list(set(etiquetas)))
    mapa = {nombre: i for i, nombre in enumerate(clases)}
    mapa_inv = {i: nombre for nombre, i in mapa.items()}
    etiquetas_codificadas = np.array([mapa[et] for et in etiquetas])
    return etiquetas_codificadas, mapa, mapa_inv

def asignar_a_centroides(x, centroides):
    """Asigna cada punto de x al cluster más cercano según los centroides dados"""
    asignaciones = []
    for fila in x:
        distancias = [calcular_distancia(fila, c) for c in centroides]
        asignaciones.append(np.argmin(distancias))
    return np.array(asignaciones)


##########################################################################################
print("PUNTO 1 \n")
url_drive = "https://drive.google.com/uc?export=download&id=1XWqzPATPdb_AvN3vP60InDwQc7cPtKSP"

# Paso 1: Descargar y cargar el dataset completo
df = descargar_dataset(url_drive)
# Paso 2: Describir columnas, tipos y faltantes
describir_dataset(df)
# Paso 3: Transformar el atributo quality
print("PUNTO 2 \n")
df_transformado = transformar_quality(df)
# Mostrar primeras filas del resultado
print("Primeras filas luego de aplicar la función objetivo a cada una de las instancias de x: \n")
print(df_transformado[['quality', 'objetivo']].head())

# Entradas (X) y salida (y)
x = df_transformado[['fixed acidity','volatile acidity','citric acid','residual sugar', 'chlorides', 'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']].values
y = df_transformado['objetivo'].values

x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = dividir_entrenamiento_prueba(x, y)

# Evaluar los tres valores de k
print(f"\n🔸 KNN Sin Ponderado")
resultados = {}
for k in [3, 5, 7]:
    y_pred = knn_predict(x_entrenamiento, y_entrenamiento, x_prueba, k)
    tasa = calcular_tasa_aciertos(y_prueba, y_pred)
    resultados[k] = tasa
    print(f"Tasa de aciertos para k = {k}: {tasa:.4f}")
mejor_k = max(resultados, key=resultados.get)
print(f"\n✅ k recomendado: {mejor_k} (tasa de aciertos = {resultados[mejor_k]:.4f})")


# ítem 3: KNN ponderado con el k recomendado
y_pred_pond = knn_predict_ponderado(x_entrenamiento, y_entrenamiento, x_prueba, mejor_k)
tasa_pond = calcular_tasa_aciertos(y_prueba, y_pred_pond)
print(f"\n🔸 KNN ponderado (k={mejor_k}) → tasa de aciertos = {tasa_pond:.4f}")

# comparar
if tasa_pond > resultados[mejor_k]:
    print("\n ✅ El KNN ponderado MEJORA el rendimiento frente al no ponderado.")
elif tasa_pond < resultados[mejor_k]:
    print("❌ El KNN ponderado empeora el rendimiento.")
else:
    print("➖ El KNN ponderado ofrece el mismo rendimiento.")



# ---------------- EJERCICIO 3 ---------------- #

# Estandarizar datos
print("\n🔸 EJERCICIO 3: K-MEANS — Clustering no supervisado con evaluación por clase real\n")

# Estandarizar datos
x_entrenamiento_std, x_prueba_std = estandarizar_datos(x_entrenamiento, x_prueba)

for k in [2, 3, 4]:
    print(f"🔹 K-MEANS con k = {k}")
    
    # Paso 1: Clustering sobre datos de entrenamiento
    asignaciones_entrenamiento, centroides = k_means_cluster_random(x_entrenamiento_std, k)
    
    # Paso 2: Comparar clusters con clases reales (entrenamiento)
    y_entrenamiento_codificado, mapa, mapa_inv = codificar_etiquetas(y_entrenamiento)
    acc_entrenamiento, mejor_asignacion_entrenamiento = accuracy_clusters_vs_reales(y_entrenamiento_codificado, asignaciones_entrenamiento, k)
    aciertos_entrenamiento = np.sum(mejor_asignacion_entrenamiento == y_entrenamiento_codificado)

    print(f"  ✅ Entrenamiento → Accuracy: {acc_entrenamiento:.2f} ({aciertos_entrenamiento}/{len(y_entrenamiento)})")

    # Paso 3: Aplicar centroides a prueba y comparar con clases reales
    prueba_asignaciones = asignar_a_centroides(x_prueba_std, centroides)
    y_prueba_codificado, _, _ = codificar_etiquetas(y_prueba)
    acc_prueba, mejor_asignacion_prueba = accuracy_clusters_vs_reales(y_prueba_codificado, prueba_asignaciones, k)
    aciertos_prueba = np.sum(mejor_asignacion_prueba == y_prueba_codificado)

    print(f"  📊 Prueba → Accuracy: {acc_prueba:.2f} ({aciertos_prueba}/{len(y_prueba)})\n")
