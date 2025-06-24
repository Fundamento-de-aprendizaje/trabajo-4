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
def k_means(x, k, max_iter=1000):
    """
    Implementa el algoritmo de K-Means.
    x: matriz de datos (numpy)
    k: número de clusters
    """
    np.random.seed(87) #Fija la semilla aleatoria para resultados reproducibles.
    # Elegir k puntos aleatorios como centroides iniciales
    indices_iniciales = np.random.choice(len(x), k, replace=False)
   
    centroides = x[indices_iniciales]

    for _ in range(max_iter):
        # Asignar cada punto al cluster más cercano
        asignaciones = []
        for fila in x: #A cada punto del dataset (fila) le calcula la distancia a cada centroide.
            distancias = [calcular_distancia(fila, centroide) for centroide in centroides]
            cluster_cercano = np.argmin(distancias) #Lo asigna al centroide más cercano.
            asignaciones.append(cluster_cercano)#Guarda el índice del cluster correspondiente.
        asignaciones = np.array(asignaciones)
        

        # Recalcular centroides
        nuevos_centroides = [] #Agrupa los puntos por cluster.
        for i in range(k):
            puntos_del_cluster = x[asignaciones == i]
            if len(puntos_del_cluster) > 0:
                nuevos_centroides.append(puntos_del_cluster.mean(axis=0))#Calcula el nuevo centroide como la media de los puntos del grupo.
            else:
                nuevos_centroides.append(centroides[i])  # Si un grupo quedó vacío, mantiene su centroide anterior.
        nuevos_centroides = np.array(nuevos_centroides)

        # Verificar convergencia. Compara si los nuevos centroides son casi iguales a los anteriores.
        if np.allclose(centroides, nuevos_centroides):
            break #Si sí, se detiene porque el algoritmo converge.

        centroides = nuevos_centroides #Si no, actualiza los centroides para la siguiente iteración.

    return asignaciones, centroides


# def evaluar_clusters(asignaciones, y_reales):
#     """
#     Evalúa los clusters comparando contra las clases reales (y_reales).
#     """
#     etiquetas_unicas = np.unique(y_reales)
#     clusters_unicos = np.unique(asignaciones)
    
#     # Mapeamos cada cluster a la clase mayoritaria dentro de él
#     mapeo = {}
#     for c in clusters_unicos:
#         clases_en_cluster = y_reales[asignaciones == c]
#         if len(clases_en_cluster) == 0:
#             mapeo[c] = None
#         else:
#             clase_mayoritaria = Counter(clases_en_cluster).most_common(1)[0][0]
#             mapeo[c] = clase_mayoritaria

#     # Predecimos usando el mapeo
#     predichos = np.array([mapeo[c] for c in asignaciones])
#     tasa_aciertos = calcular_tasa_aciertos(y_reales, predichos)
#     return tasa_aciertos, mapeo

def graficar_clusters_pca(x_std, y_reales, asignaciones, centroides, k):
    """
    Aplica PCA a los datos para reducir a 2D y grafica los clusters.
    Color = clase real, forma = cluster asignado
    """
    # PCA manual
    #Se centra la matriz (se le resta la media).
    media = x_std.mean(axis=0)
    x_centered = x_std - media
    cov = np.cov(x_centered.T)#Se calcula la matriz de covarianza.
    valores, vectores = np.linalg.eig(cov)#Se extraen sus autovalores y autovectores.
       
    idx = np.argsort(valores)[::-1]#Se ordenan los vectores según los dos autovalores más grandes (los que más varianza explican).
    componentes = vectores[:, idx[:2]] #Resultado: componentes es una matriz 2D para proyectar a 2 dimensiones.
   
    x_2d = x_centered @ componentes  #Se proyectan los datos x_std y los centroides 
    centroides_2d = (centroides - media) @ componentes #al nuevo espacio 2D usando los componentes principales.

    colores_clase = {"REGULAR": "red", "BUENO": "blue", "EXCELENTE": "green"}
    formas_cluster = ['s', 'o', '^', 'D', 'v']

    plt.figure(figsize=(8, 6))

    for i in range(len(x_2d)):
        clase = y_reales[i]
        cluster = asignaciones[i]
        plt.scatter(
            x_2d[i, 0], x_2d[i, 1],
            color=colores_clase[clase],
            marker=formas_cluster[cluster % len(formas_cluster)],
            edgecolor='black',
            s=70,
            alpha=0.7
        )

    # Centroides en X negras
    plt.scatter(
        centroides_2d[:, 0], centroides_2d[:, 1],
        marker='X', color="#FF7301", s=150, label='Centroides'
    )

    # Leyenda
    leyenda_colores = [Line2D([0], [0], marker='o', color='w', label=clase,
                              markerfacecolor=color, markersize=8) for clase, color in colores_clase.items()]
    leyenda_formas = [Line2D([0], [0], marker=forma, color='black', label=f'Cluster {i}',
                             linestyle='None', markersize=8) for i, forma in enumerate(formas_cluster[:k])]
    plt.legend(handles=leyenda_colores + leyenda_formas, loc='best')

    plt.title(f"K-Means con k = {k} (PCA 2D)\nColor = clase real, Forma = cluster")
    plt.xlabel("Estandarizada X")
    plt.ylabel("Estandarizada Y")
    plt.grid(True)
    ax = plt.gca()
    ax.set_facecolor("#A8D6AA")  # Fondo gris claro, podés usar cualquier color

    plt.tight_layout()
    plt.show()



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
print("\n🔸 EJERCICIO 3: K-MEANS ")

# Estandarizar datos
x_entrenamiento_std, x_prueba_std = estandarizar_datos(x_entrenamiento, x_prueba)

# Probar con distintos valores de k
for k in [2, 3, 4]:
   # print(f"\n--- K-Means con k = {k} ---")
    asignaciones, centroides = k_means(x_entrenamiento_std, k)
    
    # Asignar los datos de prueba al cluster más cercano
    asignaciones_entrenamiento = []
    for fila in x_entrenamiento_std:
        distancias = [calcular_distancia(fila, centroide) for centroide in centroides]
        cluster = np.argmin(distancias)
        asignaciones_entrenamiento.append(cluster)
    asignaciones_entrenamiento = np.array(asignaciones_entrenamiento)

    # Evaluar qué tan bien se alinean los clusters con las clases reales
    # tasa, mapeo_clusters = evaluar_clusters(asignaciones_entrenamiento, y_entrenamiento)
    
    graficar_clusters_pca(x_entrenamiento_std, y_entrenamiento, asignaciones, centroides, k)

 
