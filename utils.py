import numpy as np
import pandas as pd
import seaborn as sns



#----DETECCION OUTLIERS----#
def calcular_outliers_iqr(df):
    """
    Calcula información de outliers para cada columna numérica de un DataFrame usando el criterio del IQR (Interquartile Range).

    Parámetros:
    -----------
    df (pd.DataFrame): DataFrame con los datos numéricos a analizar.

    Retorna:
    --------
    dict: Diccionario donde cada clave es una columna numérica y cada valor es otro diccionario con:
        - 'count': número de outliers
        - 'percent': porcentaje de outliers
        - 'lower_bound': límite inferior para detectar outliers
        - 'upper_bound': límite superior para detectar outliers
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
        
    outliers_info = {}
        
    for columna in numeric_cols:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
            
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
            
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        porcentaje_outliers = (len(outliers) / len(df)) * 100
            
        outliers_info[columna] = {
            'count': len(outliers),
            'percent': porcentaje_outliers,
            'lower_bound': limite_inferior,
            'upper_bound': limite_superior
        }

    return outliers_info


def generar_df_estadisticas_outliers(df):
    """
    Genera un DataFrame con estadísticas de outliers para cada columna numérica, listo para usar en visualizaciones o análisis adicionales.

    Parámetros:
    -----------
    - df (pd.DataFrame): DataFrame con los datos numéricos a analizar.

    Retorna:
    --------
    pd.DataFrame: DataFrame con las columnas:
        - count: número de outliers
        - percent: porcentaje de outliers
        - lower_bound: límite inferior IQR
        - upper_bound: límite superior IQR
        Ordenado de menor a mayor porcentaje de outliers.
    """
    outliers_info = calcular_outliers_iqr(df)

    #Añadimos la informacion para detectar outliers
    resultados_df = pd.DataFrame(outliers_info).T
    resultados_df = resultados_df.sort_values('percent', ascending=True)
    
    return resultados_df





#----MATRIZ CORRELACION----#
def calculo_matriz_correlacion(df, numeric_cols = None):
    """
    Calcula la matriz de correlación de un DataFrame.
    
    Parámetros:
    -----------
        - df (pd.DataFrame): DataFrame con los datos.
        - numeric_cols (list, opcional): Lista de columnas numéricas a usar para la correlación. Si es None, se usan todas las columnas numéricas.

    Retorna:
    --------
        - pd.DataFrame: Matriz de correlación.
    """

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    corr_matrix = df[numeric_cols].corr()
    return corr_matrix




def calcular_medianas_por_fallo(df, fallos, sensores_principales):
    """
    Calcula la mediana de cada sensor para cada tipo de fallo.

    Parámetros:
    - df (pd.DataFrame): DataFrame con los datos.
    - fallos (list): lista de tipos de fallo.
    - sensores_principales (list): lista de nombres de columnas de sensores.

    Retorna:
    - pd.DataFrame: matriz de medianas (fallos x sensores).
    """
    median_matrix = pd.DataFrame(index=fallos, columns=sensores_principales)
    for fallo in fallos:
        for sensor in sensores_principales:
            median_matrix.loc[fallo, sensor] = df[df['Fault_Label'] == fallo][sensor].median()
    return median_matrix.astype(float)





#----OTROS----#
def definicion_paleta(df, columna):
    """
    Genera un diccionario que asigna un color a cada valor único de la columna.
    """
    variables = df[columna].unique().tolist()
    paleta = sns.color_palette("tab20", len(variables))
    color_map = {valor: paleta[i] for i, valor in enumerate(variables)}
    return color_map




def analisis_patrones_por_fallo(fallos, median_matrix):
    """
    Analiza los patrones de los sensores por fallo.

    Parámetros:
    - fallos (list): Lista de fallos a analizar. 
    - median_matrix (pd.DataFrame): Matriz de medianas (fallos x sensores).
    """

    print("Análisis de patrones por fallo:")
    print("--------------------------------")

    for fallo in fallos:
        print(f"\n{fallo}:")
        medians = median_matrix.loc[fallo]
        
        #Sensores más altos que el promedio
        altos = medians[medians > 0.5].sort_values(ascending=False)
        if len(altos) > 0:
            print(f"  Sensores ALTOS: {', '.join([f'{s}:{v:.2f}' for s,v in altos.items()])}")
        
        #Sensores más bajos que el promedio
        bajos = medians[medians < -0.5].sort_values()
        if len(bajos) > 0:
            print(f"  Sensores BAJOS: {', '.join([f'{s}:{v:.2f}' for s,v in bajos.items()])}")
        
        #Sensor más característico
        caracteristico = medians.abs().idxmax()
        print(f"  Sensor más CARACTERÍSTICO: {caracteristico} ({medians[caracteristico]:.2f})")