
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
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, cohen_kappa_score,
                             balanced_accuracy_score, confusion_matrix)




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




def preparar_datos_lstm(df, features, device):
    X_raw = df[features].values
    y_raw = df['target'].values

    split = int(len(X_raw) * 0.8)
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

    return X_train_raw, X_test_raw, y_train_raw, y_test_raw




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











































#----------------------------------------------#

def entrenar_evaluar_pytorch(model, X_train, X_test, y_train, y_test, le,
                             epochs=50, batch_size=32, lr=0.001, device='cpu',
                             class_weights=None, 
                             nombre_bento=None):
    
    #Si nos pasan un DataFrame, guardamos sus columnas y extraemos los valores
    feature_names = None
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train_data = X_train.values
    else:
        X_train_data = X_train

    if isinstance(X_test, pd.DataFrame):
        X_test_data = X_test.values
    else:
        X_test_data = X_test

    if isinstance(y_train, pd.Series): y_train = y_train.values
    if isinstance(y_test, pd.Series): y_test = y_test.values

    #Convertir a tensores
    X_train_tensor = torch.tensor(X_train_data, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_data, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = model.to(device)
    
    # Definir el criterio
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # --- ENTRENAMIENTO ---
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    
    # --- EVALUACIÓN ---
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy()) # .cpu() por si acaso
    
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    
    # Metricas
    acc = accuracy_score(all_true, all_preds)
    f1_global = f1_score(all_true, all_preds, average='weighted')
    
    if nombre_bento:
        bentoml.pytorch.save_model(
            nombre_bento, 
            model,
            signatures={"__call__": {"batchable": True, "batch_dim": 0}}
        )
        
        bentoml.sklearn.save_model("turboshaft_le", le)
        
        if feature_names:
            bentoml.picklable_model.save_model("turboshaft_features", feature_names)
        
        print(f"Modelo {nombre_bento} guardado correctamente.")

    resultados = {
        'accuracy': acc,
        'f1_global': f1_global,
        'f1_por_clase': f1_score(all_true, all_preds, average=None),
        'recall_por_clase': recall_score(all_true, all_preds, average=None), 
        'precision_por_clase': precision_score(all_true, all_preds, average=None),  
        'kappa': cohen_kappa_score(all_true, all_preds),  
        'balanced_accuracy': balanced_accuracy_score(all_true, all_preds),  
        'y_pred': all_preds,
        'y_true': all_true,
        'model': model,
        'classes': le.classes_,  
        'class_distribution': {clase: np.sum(all_true == i) for i, clase in enumerate(le.classes_)},  
        'confusion_matrix': confusion_matrix(all_true, all_preds)  
    }
    
    return resultados






def entrenar_evaluar_lstm(model, 
                          X_train=None, X_test=None, 
                          y_train=None, y_test=None, 
                          le=None, 
                          train_dataset=None, test_dataset=None,
                          epochs=50, batch_size=32, lr=0.001, 
                          device='cpu', class_weights=None, 
                          ventana_variable=False, collate_fn=None,
                          ventana=50,
                          nombre_bento=None):
    """
    Entrena y evalúa un modelo LSTM (con soporte BentoML).
    - ventana_variable=False: usa LSTM_basico y tensores 3D.
    - ventana_variable=True: usa LSTMModel con pack_padded_sequence.
    """

    #PREPARACIÓN BENTOML
    feature_names = None
    if X_train is not None and hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()

    #Crear DataLoaders
    if ventana_variable:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        #Crear secuencias fijas
        X_train_seq, y_train_seq = crear_secuencias(X_train, y_train, ventana)
        X_test_seq, y_test_seq = crear_secuencias(X_test, y_test, ventana)

        train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                      torch.tensor(y_train_seq, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                                     torch.tensor(y_test_seq, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #Preparar modelo y criterio
    model = model.to(device)
    
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #Entrenamiento
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            
            if ventana_variable:
                xb, yb, lengths = batch
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb, lengths)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)

            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    #Evaluación
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            if ventana_variable:
                xb, yb, lengths = batch
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb, lengths)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy())

    #Métricas
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_labels = np.arange(len(le.classes_))

    #BENTOML
    if nombre_bento:
        bentoml.pytorch.save_model(
            nombre_bento, 
            model,
            signatures={
                "__call__": {
                    "batchable": True, 
                    "batch_dim": 0
                }
            }
        )
        
        if le is not None:
            bentoml.sklearn.save_model("turboshaft_le", le)
        
        if feature_names:
            bentoml.picklable_model.save_model("turboshaft_features", feature_names)
        
        print(f"Modelo {nombre_bento} guardado.")

    resultados = {
        'accuracy': accuracy_score(all_true, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_true, all_preds),
        'f1_global': f1_score(all_true, all_preds, average='weighted'),
        'f1_por_clase': f1_score(all_true, all_preds, average=None, labels=all_labels, zero_division=0),
        'recall_por_clase': recall_score(all_true, all_preds, average=None, labels=all_labels, zero_division=0),
        'precision_por_clase': precision_score(all_true, all_preds, average=None, labels=all_labels, zero_division=0),
        'kappa': cohen_kappa_score(all_true, all_preds),
        'y_true': all_true,
        'y_pred': all_preds,
        'model': model,
        'classes': le.classes_,
        'class_distribution': {clase: np.sum(all_true == i) for i, clase in enumerate(le.classes_)},
        'confusion_matrix': confusion_matrix(all_true, all_preds, labels=all_labels),
        'ventana_usada': ventana,
        'ventana_variable': ventana_variable
    }

    return resultados



def crear_secuencias_variables(df, features_cols, target_col, sequence_id_col=None):
    """
    Divide el DataFrame en una lista de secuencias de longitud variable.

    """
    X_seqs = []
    y_seqs = []
    
    if sequence_id_col and sequence_id_col in df.columns:
        for _, group in df.groupby(sequence_id_col):
            X_seqs.append(group[features_cols].values)
            y_seqs.append(group[target_col].iloc[-1])
            
    else:
        current_seq = []
        
        for i, row in df.iterrows():
            current_seq.append(row[features_cols].values)
            
            if row[target_col] != 0: 
                X_seqs.append(np.array(current_seq))
                y_seqs.append(row[target_col])
                current_seq = []
                
        if len(current_seq) > 0:
            pass 

    return X_seqs, y_seqs








#----EXTRAS LSTM----#
def crear_secuencias(X, y, ventana):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - ventana):
        # Para cada posicion, tomar una secuencia de longitud de la ventana
        X_seq.append(X[i:i+ventana])
        y_seq.append(y[i+ventana-1])  # Etiqueta del ultimo paso
    return np.array(X_seq), np.array(y_seq)



# def crear_dataloaders_ventana(X_train, y_train, X_test, y_test, batch_size=64, ventana=50, ventana_variable=False):
#     """
#     Crea DataLoaders para LSTM con ventana fija.
#     """
#     # Secuencias fijas
#     X_train_seq, y_train_seq = crear_secuencias(X_train, y_train, ventana)
#     X_test_seq, y_test_seq = crear_secuencias(X_test, y_test, ventana)
#     train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
#                                       torch.tensor(y_train_seq, dtype=torch.long))
#     test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32),
#                                      torch.tensor(y_test_seq, dtype=torch.long))
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     return train_loader, test_loader, X_train_seq, X_test_seq, y_train_seq, y_test_seq


