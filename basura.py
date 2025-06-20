# ------------------------- FUNCIÓN 4 ------------------------- #
def guardar_csv(df, nombre_archivo):
    """Guarda el DataFrame en un archivo CSV"""
    df.to_csv(nombre_archivo, index=False)
    print(f"\n📁 Archivo '{nombre_archivo}' guardado con éxito.")

guardar_csv(df_transformado, 'vino_transformado.csv')
# Ejecutar