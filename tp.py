# Librerías
import pandas as pd
import numpy as np
from collections import Counter

# ------------------------- FUNCIONES BASE ------------------------- #

def descargar_dataset(url):
    return pd.read_csv(url, encoding='latin1')

def describir_dataset(df):
    print(f"\nCantidad de registros: {df.shape[0]}")
    print(f"Cantidad de columnas: {df.shape[1]}\n")
    print("Tipos de datos por columna:")
    for col in df.columns:
        if("quality" != col):
            print(col,": ",df[col].dtype, "||", "Numérica continua")
        else:
            print(col,": ",df[col].dtype, "||", "Categorica ordinal")   
    print("\nCantidad de valores faltantes por columna:")
    print(df.isnull().sum())
    if df.isnull().sum().sum() > 0:
        print("\nSe encontraron valores faltantes.")
    else:
        print("\nNo se encontraron valores faltantes.")

def clasificar(x):
    if x < 5:
        return "REGULAR"
    elif x == 5:
        return "BUENO"
    else:
        return "EXCELENTE"

def transformar_quality(df):
    df_filtrado = df[df['quality'] != 6].copy()
    print("Cantidad de datos sin calidad igual a 6 : ", df_filtrado.shape[0])
    df_filtrado['objetivo'] = df_filtrado['quality'].apply(clasificar)
    return df_filtrado

# def dividir_entrenamiento_prueba(x, y, prueba_size=0.2, random_state=20):
#     np.random.seed(random_state)
#     indices = np.random.permutation(len(x))
    
#     n_train = int(len(x) * (1 - prueba_size))
#     x_entrenamiento = x[indices[:n_train]]
#     x_prueba = x[indices[n_train:]]
#     y_entrenamiento = y[indices[:n_train]]
#     y_prueba = y[indices[n_train:]]
#     print(f"Entrenamiento: {len(x_entrenamiento)}, Prueba: {len(x_prueba)}")
#     return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba

def dividir_entrenamiento_prueba(x, y, prueba_size=0.2, random_state=20):
    """
    Divide los datos en entrenamiento y prueba sin mezclar (manteniendo el orden original).
    La semilla se incluye para compatibilidad, pero no se usa ya que no se aleatoriza.
    """
    np.random.seed(random_state)  # No se usa en esta versión, pero queda para compatibilidad
    n_train = int(len(x) * (1 - prueba_size))
    
    x_entrenamiento = x[:n_train]
    x_prueba = x[n_train:]
    y_entrenamiento = y[:n_train]
    y_prueba = y[n_train:]
    
    print(f"Entrenamiento: {len(x_entrenamiento)}, Prueba: {len(x_prueba)}")
    return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba


def calcular_distancia(x_prueba, x_entrenamiento):    
    return np.sqrt(np.sum((x_prueba - x_entrenamiento) ** 2))

# # Convertir a arrays de numpy si todavía no lo hiciste
# x_entrenamiento = np.array(x_entrenamiento)
# x_prueba = np.array(x_prueba)

# # Estandarización
# x_entrenamiento_std, x_prueba_std = estandarizar_datos(x_entrenamiento, x_prueba)

def estandarizar_datos(x_entrenamiento, x_prueba):
    """
    Estandariza los datos usando media y desvío del entrenamiento.
    """
    medias = x_entrenamiento.mean(axis=0)
    print("medias",medias)
    desvios = x_entrenamiento.std(axis=0)

    x_entrenamiento_std = (x_entrenamiento - medias) / desvios
    x_prueba_std = (x_prueba - medias) / desvios  # Usar solo estadísticas del entrenamiento

    return x_entrenamiento_std, x_prueba_std

# ------------------- KNN SIN PONDERAR -------------------
# def estandarizar(x_train, x_test):
#     medias = np.mean(x_train, axis=0)
#     desvios = np.std(x_train, axis=0)
#     x_train_std = (x_train - medias) / desvios
#     x_test_std = (x_test - medias) / desvios
#     return x_train_std, x_test_std
def knn_predict(x_entrenamiento, y_entrenamiento, x_prueba, k):
    predicciones = []
    #estandarizar datos
    x_entrenamiento_std, x_prueba_std = estandarizar_datos(x_entrenamiento, x_prueba)
    for i in range(len(x_prueba)):
        distancias = []
        for j in range(len(x_entrenamiento)):
            distancia = calcular_distancia(x_prueba_std[i], x_entrenamiento_std[j])
            etiqueta = y_entrenamiento[j]
            distancias.append((distancia, etiqueta))
        distancias.sort(key=lambda tupla: tupla[0])
        vecinos = distancias[:k]
        clases = [etiqueta for _, etiqueta in vecinos]
        clase_mas_comun = Counter(clases).most_common(1)[0][0]
        predicciones.append(clase_mas_comun)
    return np.array(predicciones)

# ------------------- KNN PONDERADO -------------------
def knn_predict_ponderado(x_entrenamiento, y_entrenamiento, x_prueba, k):
    predicciones = []
    for i in range(len(x_prueba)):
        distancias = []
        for j in range(len(x_entrenamiento)):
            distancia = calcular_distancia(x_prueba[i], x_entrenamiento[j])
            etiqueta = y_entrenamiento[j]
            distancias.append((distancia, etiqueta))
        distancias.sort(key=lambda tupla: tupla[0])
        vecinos = distancias[:k]

        pesos = {}
        for distancia, etiqueta in vecinos:
            peso = 1 / (distancia + 1e-5)
            if etiqueta in pesos:
                pesos[etiqueta] += peso
            else:
                pesos[etiqueta] = peso

        clase_ponderada = max(pesos.items(), key=lambda item: item[1])[0]
        predicciones.append(clase_ponderada)
    return np.array(predicciones)

# ------------------- ACCURACY -------------------
def calcular_accuracy(y_real, y_predicho):
    return np.mean(y_real == y_predicho)

# ------------------- ESTADÍSTICAS -------------------
def calcular_estadisticas_numpy(arr):
    medias = np.mean(arr, axis=0)
    desvios = np.std(arr, axis=0)
    estadisticas = []
    for i in range(len(medias)):
        print(f"Columna {i} → Media: {medias[i]:.4f} | Desvío estándar: {desvios[i]:.4f}")
        estadisticas.append((medias[i], desvios[i]))
    return estadisticas

# ------------------- PROCESAMIENTO COMPLETO -------------------

print("PUNTO 1 \n")
url_drive = "https://drive.google.com/uc?export=download&id=1XWqzPATPdb_AvN3vP60InDwQc7cPtKSP"
df = descargar_dataset(url_drive)
describir_dataset(df)

print("\nPUNTO 2 \n")
df_transformado = transformar_quality(df)
print(df_transformado[['quality', 'objetivo']].head())

# Variables independientes y dependiente
x = df_transformado[['fixed acidity','volatile acidity','citric acid','residual sugar',
                     'chlorides','free sulfur dioxide','total sulfur dioxide','density',
                     'pH','sulphates','alcohol']].values
y = df_transformado['objetivo'].values
# x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = dividir_entrenamiento_prueba(x, y)
# print ("SIN Estandarizado ENTRENAMIENTO", x_entrenamiento)
# === ESTANDARIZACIÓN ===
# medias = np.mean(x, axis=0)
# desvios = np.std(x, axis=0)
# x = (x - medias) / desvios
# print("medias",medias)
# print("desvios",desvios)
# # Dividir entrenamiento/prueba
x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = dividir_entrenamiento_prueba(x, y) 
print ("Estandarizado ENTRENAMIENTO", x_entrenamiento)


# === ESTANDARIZACIÓN === SOLO con datos de entrenamiento
# medias = np.mean(x_entrenamiento, axis=0)
# desvios = np.std(x_entrenamiento, axis=0)

# # Estandarizar entrenamiento
# x_entrenamiento = (x_entrenamiento - medias) / desvios


# # Estandarizar prueba con los mismos valores
# x_prueba = (x_prueba - medias) / desvios
# print ("Estandarizado", x)
# Evaluar valores de k
accuracies = {}
for k in [3, 5, 7]:
    y_pred = knn_predict(x_entrenamiento, y_entrenamiento, x_prueba, k)
    acc = calcular_accuracy(y_prueba, y_pred)
    print(f"🔹 Accuracy para k = {k}: {acc:.4f}")
    accuracies[k] = acc

# Elegir mejor k
k_recomendado = max(accuracies, key=accuracies.get)
print(f"\n✅ k recomendado: {k_recomendado} (mayor accuracy)")

# KNN ponderado
print("\nPUNTO 3: KNN ponderado\n")
y_pred_ponderado = knn_predict_ponderado(x_entrenamiento, y_entrenamiento, x_prueba, k_recomendado)
acc_ponderado = calcular_accuracy(y_prueba, y_pred_ponderado)
print(f"🔸 Accuracy con KNN ponderado para k = {k_recomendado}: {acc_ponderado:.4f}")

# Comparación
if acc_ponderado > accuracies[k_recomendado]:
    print("✅ El método ponderado mejora el accuracy.")
elif acc_ponderado < accuracies[k_recomendado]:
    print("❌ El método ponderado obtiene peor resultado.")
else:
    print("➖ El método ponderado da el mismo resultado.")
