
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt  # Asegurate de tener esto también

# def graficar_pairplot(x, asignaciones, columnas):
#     """
#     Visualiza múltiples combinaciones de variables y clusters usando seaborn.pairplot.
#     x: matriz de datos (original o estandarizada)
#     asignaciones: array con número de cluster por cada fila
#     columnas: lista con nombres de las columnas de x
#     """
#     df = pd.DataFrame(x, columns=columnas)  # Crea un DataFrame con nombres de columnas
#     df["cluster"] = asignaciones            # Agrega columna de cluster asignado

#     # Muestra combinación de todas las variables por pares coloreadas por cluster
#     sns.pairplot(df, hue="cluster", palette="Set2", plot_kws={"alpha": 0.6})
#     plt.suptitle("Pairplot de Clusters K-Means", y=1.02)
#     plt.show()


# # --- Configuración para usar pairplot ---
# columnas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#             'pH', 'sulphates', 'alcohol']

# # Ejecutar agrupamiento
# k = 4
# asignaciones, centroides = k_means_cluster_random(x_entrenamiento_std, k)

# # Visualizar con pairplot
# graficar_pairplot(x_entrenamiento_std, asignaciones, columnas)



# def graficar_clusters_pca(x_std, y_reales, asignaciones, centroides, k):
#     """
#     Aplica PCA a los datos para reducir a 2D y grafica los clusters.
#     Color = clase real, forma = cluster asignado
#     """
#     # PCA manual
#     #Se centra la matriz (se le resta la media).
#     media = x_std.mean(axis=0)
#     x_centered = x_std - media
#     cov = np.cov(x_centered.T)#Se calcula la matriz de covarianza.
#     valores, vectores = np.linalg.eig(cov)#Se extraen sus autovalores y autovectores.
       
#     idx = np.argsort(valores)[::-1]#Se ordenan los vectores según los dos autovalores más grandes (los que más varianza explican).
#     componentes = vectores[:, idx[:2]] #Resultado: componentes es una matriz 2D para proyectar a 2 dimensiones.
   
#     x_2d = x_centered @ componentes  #Se proyectan los datos x_std y los centroides 
#     centroides_2d = (centroides - media) @ componentes #al nuevo espacio 2D usando los componentes principales.

#     colores_clase = {"REGULAR": "red", "BUENO": "blue", "EXCELENTE": "green"}
#     formas_cluster = ['s', 'o', '^', 'D', 'v']

#     plt.figure(figsize=(8, 6))

#     for i in range(len(x_2d)):
#         clase = y_reales[i]
#         cluster = asignaciones[i]
#         plt.scatter(
#             x_2d[i, 0], x_2d[i, 1],
#             color=colores_clase[clase],
#             marker=formas_cluster[cluster % len(formas_cluster)],
#             edgecolor='black',
#             s=70,
#             alpha=0.7
#         )

#     # Centroides en X negras
#     plt.scatter(
#         centroides_2d[:, 0], centroides_2d[:, 1],
#         marker='X', color="#FF7301", s=150, label='Centroides'
#     )

#     # Leyenda
#     leyenda_colores = [Line2D([0], [0], marker='o', color='w', label=clase,
#                               markerfacecolor=color, markersize=8) for clase, color in colores_clase.items()]
#     leyenda_formas = [Line2D([0], [0], marker=forma, color='black', label=f'Cluster {i}',
#                              linestyle='None', markersize=8) for i, forma in enumerate(formas_cluster[:k])]
#     plt.legend(handles=leyenda_colores + leyenda_formas, loc='upper left')

#     plt.title(f"K-Means con k = {k} (PCA 2D)\nColor = clase real, Forma = cluster")
#     plt.xlabel("Estandarizada X")
#     plt.ylabel("Estandarizada Y")
#     plt.grid(True)
#     ax = plt.gca()
#     ax.set_facecolor("#A8D6AA")  # Fondo gris claro, podés usar cualquier color

#     plt.tight_layout()
#     plt.show()





    #(x_entrenamiento_std, y_entrenamiento, asignaciones, centroides, k)
