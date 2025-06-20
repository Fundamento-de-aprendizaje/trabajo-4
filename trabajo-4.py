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

# === 2. DIVISIÓN ENTRE ENTRENAMIENTO Y PRUEBA ===#la buena es la 20
def dividir_entrenamiento_prueba(x, y, prueba_size=0.2, random_state=20):
    np.random.seed(random_state)
    indices = np.random.permutation(len(x))#que es?
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


# np.sqrt(np.sum((x1 - x2), (x1 - x2) ))

# ------------------- 2. FUNCIÓN PRINCIPAL DE KNN -------------------
def knn_predict(x_entrenamiento, y_train, x_prueba, k):
    """
    Predice la clase de cada ejemplo en x_prueba usando KNN con voto simple.
    """
    predicciones = []

    for i in range(len(x_prueba)):
        distancias = []

        # Calcular la distancia de este x_prueba[i] con cada x_entrenamiento[j]
        for j in range(len(x_entrenamiento)):
            distancia = calcular_distancia(x_prueba[i], x_entrenamiento[j])
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


# ------------------- 3. FUNCIÓN PARA CALCULAR ACCURACY -------------------
def calcular_accuracy(y_real, y_predicho):
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
x = df_transformado[['residual sugar', 'chlorides', 'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']].values
y = df_transformado['quality'].values

x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = dividir_entrenamiento_prueba(x, y)

# Evaluar los tres valores de k
for k in [3, 5, 7]:
    y_pred = knn(x_entrenamiento, y_entrenamiento, x_prueba, k, ponderado=False)
    acc = evaluar_accuracy(y_prueba, y_pred)
    print(f"🔹 Accuracy para k = {k}: {acc:.4f}")
