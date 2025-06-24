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

def dividir_entrenamiento_prueba(x, y, prueba_size=0.2, random_state=87):
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

def calcular_distancia_minkowski(x1, x2, p=2):
    """
    Calcula la distancia de Minkowski entre dos vectores.
    """
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

def knn_predict_minkowski(x_entrenamiento, y_entrenamiento, x_prueba, k, p=2):
    """
    Predice usando KNN variando el parámetro p de la distancia de Minkowski.
    """
    predicciones = []
    x_entrenamiento_std, x_prueba_std = estandarizar_datos(x_entrenamiento, x_prueba)

    for i in range(len(x_prueba_std)):
        distancias = []
        for j in range(len(x_entrenamiento_std)):
            distancia = calcular_distancia_minkowski(x_prueba_std[i], x_entrenamiento_std[j], p)
            etiqueta = y_entrenamiento[j]
            distancias.append((distancia, etiqueta))
        
        distancias.sort(key=lambda tupla: tupla[0])
        vecinos = distancias[:k]
        clases = [etiqueta for _, etiqueta in vecinos]
        clase_mas_comun = Counter(clases).most_common(1)[0][0]
        predicciones.append(clase_mas_comun)
    
    return np.array(predicciones)

def evaluar_knn_para_distancias(x_ent, y_ent, x_pru, y_pru, k, p_values):
    resultados = []

    for p in p_values:
        y_pred = knn_predict_minkowski(x_ent, y_ent, x_pru, k, p)
        accuracy = calcular_tasa_aciertos(y_pru, y_pred)
        resultados.append({'p': p, 'accuracy': accuracy})
    
    return pd.DataFrame(resultados)

def evaluar_knn_minkowski_completo(x_ent, y_ent, x_pru, y_pru, k_values, p_values):
    resultados = []

    for k in k_values:
        for p in p_values:
            y_pred = knn_predict_minkowski(x_ent, y_ent, x_pru, k, p)
            accuracy = calcular_tasa_aciertos(y_pru, y_pred)
            resultados.append({'k': k, 'p': p, 'accuracy': accuracy})
    
    return pd.DataFrame(resultados)


# ------------------- 3. FUNCIÓN PARA CALCULAR OBSERVACIONES BIEN ETIQUETADAS -------------------
def calcular_tasa_aciertos(y_real, y_predicho):
    """
    Calcula el porcentaje de predicciones correctas.
    """
    return np.mean(y_real == y_predicho)

def graficar_accuracy_por_k_y_p(resultados_df):
    plt.figure(figsize=(10, 6))

    for p in sorted(resultados_df['p'].unique()):
        subset = resultados_df[resultados_df['p'] == p]
        plt.plot(subset['k'], subset['accuracy'], marker='o', label=f'p = {p}')

    plt.xlabel('Número de vecinos (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy de KNN según k y p (Minkowski)')
    plt.legend(title='Valor de p')
    plt.grid(True)
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

k_vals = [3, 5, 7, 9, 13, 27]
p_vals = [1, 2, 3, 6]

resultados = evaluar_knn_minkowski_completo(x_entrenamiento, y_entrenamiento, x_prueba, y_prueba, k_vals, p_vals)

print(resultados)  # Para ver la tabla
graficar_accuracy_por_k_y_p(resultados)  # Para visualizar

def graficar_accuracy_vs_k(resultados_df):
    plt.figure(figsize=(8, 5))
    plt.plot(resultados_df['k'], resultados_df['accuracy'], marker='o', linestyle='-')
    plt.xlabel('Número de vecinos (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy del KNN en función de k')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

k_vals = [3, 5, 7]
accuracies = []

for k in k_vals:
    y_pred = knn_predict_minkowski(x_entrenamiento, y_entrenamiento, x_prueba, k, p=2)  # p fijo
    acc = calcular_tasa_aciertos(y_prueba, y_pred)
    accuracies.append(acc)

resultados_df = pd.DataFrame({'k': k_vals, 'accuracy': accuracies})
graficar_accuracy_vs_k(resultados_df)
