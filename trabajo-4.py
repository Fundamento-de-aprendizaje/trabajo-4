# Librerías
import pandas as pd
import numpy as np
from collections import Counter
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
    df_filtrado = df[df['quality'] != 6].copy()  # <- importante usar .copy() para evitar warnings
    
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

def dividir_entrenamiento_prueba(x, y, prueba_size=0.2, random_state=42):
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
    x_prueba_std = (x_prueba - medias) / desvios  # Usar solo estadísticas del entrenamiento

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
            peso = 1/(distanciaAlCuadrado + 1e-8)
            pesos[etiqueta] = pesos.get(etiqueta, 0) + peso
        # elige la clase de mayor peso
        predicciones.append(max(pesos, key=pesos.get))
       
    return np.array(predicciones)



# ------------------- 3. FUNCIÓN PARA CALCULAR ACCURACY -------------------
def calcular_tasa_aciertos(y_real, y_predicho):
    """
    Calcula el porcentaje de predicciones correctas.
    """
    return np.mean(y_real == y_predicho)


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
