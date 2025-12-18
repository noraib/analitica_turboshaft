
from sklearn.utils import compute_class_weight
import torch
from utils import preprocesar_datos
from utils import split_data
from utils import split_data
from utils import evaluar_modelo_scikit, mostrar_resultados_notebook
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
import bentoml
import json


def preparar_datos(df):
    X, y, le = preprocesar_datos(df)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Distribución de clases en entrenamiento: {np.bincount(y_train)}")
    print(f"Distribución de clases en prueba: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, le


def aplicar_smote(X_train, y_train):
    smote = SMOTE(random_state=111)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"Distribucion original: {Counter(y_train)}")
    print(f"Distribución de clases después de SMOTE: {np.bincount(y_res)}")

    return X_res, y_res


def medias_moviles(df, ventana=5):
    #creamos "medias móviles" (el promedio de los últimos 5 valores del sensor) para suavizar el ruido de los sensores
    #esto ayuda al modelo a ver la tendencia general en lugar de picos sueltos.

    #identificamos las columnas de sensores (excluyendo la label y timestamp)
    cols_sensores = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Fault_Label' in cols_sensores: cols_sensores.remove('Fault_Label')

    #creamos nuevas variables con una ventana de tiempo
    #(el promedio de los últimos 5 datos)
    ventana = ventana
    
    df_ingenieria = df.copy()

    for col in cols_sensores:
        #Tendencia
        df_ingenieria[f'{col}_mean_{ventana}'] = df_ingenieria[col].rolling(window=ventana).mean()
        
        #std
        df_ingenieria[f'{col}_std_{ventana}'] = df_ingenieria[col].rolling(window=ventana).std()

    #para la limpieza (al crear ventanas, las primeras filas quedan vacías osea que las rellenamos).
    df_ingenieria = df_ingenieria.fillna(method='bfill').fillna(method='ffill')

    print(f"Dimensiones nuevas del dataset: {df_ingenieria.shape}")
    # ----------------------------------------------
    return df_ingenieria


    

def seleccion_caracteristicas(X, y):
    print(f"Características originales ({X.shape[1]}): {X.columns.tolist()}")

    #Usamos Random Forest para ver qué sensores importan de verdad
    sel_model = RandomForestClassifier(n_estimators=50, random_state=111, n_jobs=-1)
    sel_model.fit(X, y)

    #Obtenemos la importancia y seleccionamos las mejores
    #cogemos los 9 mejores
    importancias = pd.Series(sel_model.feature_importances_, index=X.columns)
    mejores_features = importancias.nlargest(9).index.tolist()

    #tambiém podríamos elegir las que superen un umbral
    #mejores_features = importancias[importancias > 0.05].index.tolist()

    print(f"Características seleccionadas ({len(mejores_features)}): {mejores_features}")

    return X[mejores_features]







def selecccion_caracteristicas(X, y):
    print(f"Características originales ({X.shape[1]}): {X.columns.tolist()}")

    #Usamos Random Forest para ver qué sensores importan de verdad
    sel_model = RandomForestClassifier(n_estimators=50, random_state=111, n_jobs=-1)
    sel_model.fit(X, y)

    #Obtenemos la importancia y seleccionamos las mejores
    #cogemos los 7 mejores
    importancias = pd.Series(sel_model.feature_importances_, index=X.columns)
    mejores_features = importancias.nlargest(7).index.tolist()

    #tambiém podríamos elegir las que superen un umbral
    #mejores_features = importancias[importancias > 0.05].index.tolist()

    print(f"Características seleccionadas ({len(mejores_features)}): {mejores_features}")

    X = X[mejores_features]
    

    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Distribución de clases en entrenamiento: {np.bincount(y_train)}")
    print(f"Distribución de clases en prueba: {np.bincount(y_test)}")


def random_forest(X_train, X_test, y_train, y_test, le):
    rf_model = RandomForestClassifier(n_estimators=10, random_state=111, class_weight='balanced')
    resultados_rf = evaluar_modelo_scikit(rf_model, X_train, X_test, y_train, y_test, "Random Forest", le)

    bentoml.sklearn.save_model(
        "modelo_RandomForest", 
        rf_model,
        signatures={"predict": {"batchable": True}}
    )
    
    bentoml.sklearn.save_model("turboshaft_le", le)
    
    if hasattr(X_train, 'columns'):
        bentoml.picklable_model.save_model(
            "turboshaft_features", 
            X_train.columns.tolist()
        )
        
    print(f"Modelo guardado en BentoML.")
    
    return resultados_rf


def gbm(X_train, X_test, y_train, y_test, le):
    gbm_model = GradientBoostingClassifier(n_estimators=50, random_state=111)
    resultados_gbm = evaluar_modelo_scikit(gbm_model, X_train, X_test, y_train, y_test, "Gradient Boosting", le)

    bentoml.sklearn.save_model(
        "modelo_GBM",
        gbm_model,
        signatures={"predict": {"batchable": True}}
    )
    
    bentoml.sklearn.save_model("turboshaft_le", le)
    
    if hasattr(X_train, 'columns'):
        bentoml.picklable_model.save_model(
            "turboshaft_features", 
            X_train.columns.tolist()
        )
        
    print(f"Modelo guardado en BentoML.")
    
    return resultados_gbm  



def xgboost(X_train, X_test, y_train, y_test, le):
    counts = np.bincount(y_train)
    ratio = counts[0] / counts[1] if len(counts) > 1 else 1
    
    xgb_model = XGBClassifier(
        n_estimators=100, 
        random_state=111, 
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        scale_pos_weight=ratio
    )

    resultados_xgb = evaluar_modelo_scikit(xgb_model, X_train, X_test, y_train, y_test, "XGBoost", le)

    bentoml.sklearn.save_model(
        "modelo_XGBoost", 
        xgb_model,
        signatures={"predict": {"batchable": True}}
    )
    
    bentoml.sklearn.save_model("turboshaft_le", le)
    
    if hasattr(X_train, 'columns'):
        bentoml.picklable_model.save_model(
            "turboshaft_features", 
            X_train.columns.tolist()
        )
        
    print(f"Modelo guardado en BentoML.")
    
    return resultados_xgb

def svm(X_train, X_test, y_train, y_test, le):
    svm_model = SVC(kernel='rbf', probability=True, random_state=111, class_weight='balanced')
    resultados_svm = evaluar_modelo_scikit(svm_model, X_train, X_test, y_train, y_test, "SVM", le)
    return resultados_svm

def svm(X_train, X_test, y_train, y_test, le):
    svm_model = SVC(kernel='rbf', probability=True, random_state=111, class_weight='balanced')
    resultados_svm = evaluar_modelo_scikit(svm_model, X_train, X_test, y_train, y_test, "SVM", le)
    
    bentoml.sklearn.save_model(
        "modelo_SVM", 
        svm_model,
        signatures={"predict": {"batchable": True}}
    )
    
    bentoml.sklearn.save_model("turboshaft_le", le)
    
    if hasattr(X_train, 'columns'):
        bentoml.picklable_model.save_model(
            "turboshaft_features", 
            X_train.columns.tolist()
        )
        
    print(f"Modelo guardado en BentoML.")
    
    return resultados_svm