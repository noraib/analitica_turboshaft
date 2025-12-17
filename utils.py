from IPython.display import display
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report,recall_score, precision_score, cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from modelos.ventana_lstm import SequenceDataset, collate_fn

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




#----MODELOS----#
def preprocesar_datos(df, target_col="Fault_Label"):
    le = LabelEncoder()
    df[f"{target_col}_Encoded"] = le.fit_transform(df[target_col])
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if f"{target_col}_Encoded" in numeric_cols:
        numeric_cols.remove(f"{target_col}_Encoded")
    
    X = df[numeric_cols]
    y = df[f"{target_col}_Encoded"]
    return X, y, le


def split_data(X, y, test_size=0.3, random_state=111):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluar_modelo_scikit(modelo, X_train, X_test, y_train, y_test, nombre_modelo, le):
    """Entrena y evalúa un modelo, mostrando métricas clave"""
    
    # Entrenar modelo
    modelo.fit(X_train, y_train)
    
    # Predecir
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test) if hasattr(modelo, "predict_proba") else None
    
    # Métricas globales
    acc = accuracy_score(y_test, y_pred)
    f1_global = f1_score(y_test, y_pred, average='weighted')
    
    # Métricas por clase 
    f1_por_clase = f1_score(y_test, y_pred, average=None)
    recall_por_clase = recall_score(y_test, y_pred, average=None)
    precision_por_clase = precision_score(y_test, y_pred, average=None)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Métricas extra pa desbalanceo
    # Cohen's Kappa (mejor que accuracy pa desbalanceo)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Balanced Accuracy (promedio de recall por clase)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    return {
        'modelo': modelo,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1_global': f1_global,
        'f1_por_clase': f1_por_clase,
        'recall_por_clase': recall_por_clase,
        'precision_por_clase': precision_por_clase,
        'kappa': kappa,
        'y_true': y_test,       
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'classes': le.classes_,
        'class_distribution': {clase: np.sum(y_test == i) for i, clase in enumerate(le.classes_)}
    }
    
def mostrar_resultados_notebook(resultados, le):
    """Muestra los resultados de un modelo en un notebook con el layout especificado."""
    import matplotlib.pyplot as plt
    from plots import plot_confusion_matrix
    
    print("Resultados del Modelo")
    print("---------------------")
    
    # Metricas globales
    print(f"Accuracy: {resultados['accuracy']:.3f}")
    print(f"Balanced Accuracy: {resultados['balanced_accuracy']:.3f}")
    print(f"F1 Global: {resultados['f1_global']:.3f}")
    print(f"Kappa: {resultados['kappa']:.3f}")
    print(f"Muestras en test: {len(resultados['y_true'])}\n")
    
    # Metricas por clase
    print("Métricas por Clase:")
    metrics_data = []
    for i, clase in enumerate(resultados['classes']):
        metrics_data.append({
            'Clase': clase,
            'Precision': resultados['precision_por_clase'][i],
            'Recall': resultados['recall_por_clase'][i],
            'F1': resultados['f1_por_clase'][i],
        })
    
    df_class_metrics = pd.DataFrame(metrics_data)
    display(df_class_metrics.style.format({
        'Precision': '{:.3f}',
        'Recall': '{:.3f}',
        'F1': '{:.3f}'
    }))
    
    # Distribucion de clases
    print("Distribución de clases en el conjunto de test:")
    for clase, count in resultados['class_distribution'].items():
        porcentaje = (count / len(resultados['y_true'])) * 100
        print(f"  {clase}: {count} muestras ({porcentaje:.2f}%)")
    
    # Matriz de confusion
    print("\nMatriz de Confusión:")
    fig = plot_confusion_matrix(
        y_true=resultados['y_true'],
        y_pred=resultados['y_pred'],
        class_names=le.classes_
    )
    plt.show()
    
    
    

def mostrar_resultados_modelo(resultados, le):
    """Muestra los resultados de un modelo en Streamlit con el layout especificado."""
    import streamlit as st
    import pandas as pd
    from plots import plot_confusion_matrix
    
    # Crear dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resultados del Modelo")
        
        # Pestañas para organizar
        tab_global, tab_clases, tab_detalle = st.tabs(["Global", "Por Clase", "Detalle"])
        
        with tab_global:
            # Métricas globales
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.metric("Accuracy", f"{resultados['accuracy']:.3f}")
            with g2:
                st.metric("Balanced Acc", f"{resultados['balanced_accuracy']:.3f}")
            with g3:
                st.metric("F1 Global", f"{resultados['f1_global']:.3f}")
            with g4:
                st.metric("Kappa", f"{resultados['kappa']:.3f}")
            
            # Distribución
            st.caption(f"Muestras en test: {len(resultados['y_true'])}")
        
        with tab_clases:
            # Metricas por clase
            metrics_data = []
            for i, clase in enumerate(resultados['classes']):
                metrics_data.append({
                    'Clase': clase,
                    'Precision': resultados['precision_por_clase'][i],
                    'Recall': resultados['recall_por_clase'][i],
                    'F1': resultados['f1_por_clase'][i],
                })
            
            df_class_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_class_metrics.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1': '{:.3f}'
            }))
        
        with tab_detalle:
            # Informacion adicional sobre la distribucion (desbalanceo)
            st.write("**Distribución de clases:**")
            for clase, count in resultados['class_distribution'].items():
                st.progress(count / len(resultados['y_true']), 
                          text=f"{clase}: {count} muestras")
    
    with col2:
        st.subheader("Matríz de Confusión")
        fig = plot_confusion_matrix(
            y_true=resultados['y_true'],
            y_pred=resultados['y_pred'],
            class_names=le.classes_
        )
        st.pyplot(fig)
    
    return col1, col2



def entrenar_evaluar_pytorch(model, X_train, X_test, y_train, y_test, le,
                             epochs=50, batch_size=32, lr=0.001, device='cpu',
                             class_weights=None):
    
    #Convertir a tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = model.to(device)
    
    #Definir el criterio (usar pesos solo si se pasan)
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    #Entrenamiento
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    
    #Evaluación
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())
    
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    
    #Metricas
    acc = accuracy_score(all_true, all_preds)
    f1_global = f1_score(all_true, all_preds, average='weighted')
    f1_por_clase = f1_score(all_true, all_preds, average=None)
    recall_por_clase = recall_score(all_true, all_preds, average=None)  
    precision_por_clase = precision_score(all_true, all_preds, average=None)  
    kappa = cohen_kappa_score(all_true, all_preds)  
    balanced_acc = balanced_accuracy_score(all_true, all_preds)  

    #Mostrar resultados
    resultados = {
        'accuracy': acc,
        'f1_global': f1_global,
        'f1_por_clase': f1_por_clase,
        'recall_por_clase': recall_por_clase, 
        'precision_por_clase': precision_por_clase,  
        'kappa': kappa,  
        'balanced_accuracy': balanced_acc,  
        'y_pred': all_preds,
        'y_true': all_true,
        'model': model,
        'classes': le.classes_,  
        'class_distribution': {clase: np.sum(all_true == i) for i, clase in enumerate(le.classes_)},  
        'confusion_matrix': confusion_matrix(all_true, all_preds)  
    }
    
    return resultados



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, cohen_kappa_score, balanced_accuracy_score, confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def entrenar_evaluar_lstm(model, 
                          X_train=None, X_test=None, 
                          y_train=None, y_test=None, 
                          le=None, 
                          train_dataset=None, test_dataset=None,
                          epochs=50, batch_size=32, lr=0.001, 
                          device='cpu', class_weights=None, 
                          ventana_variable=False, collate_fn=None,
                          ventana=50):
    """
    Entrena y evalúa un modelo LSTM.

    - ventana_variable=False: usa LSTM_basico y tensores 3D.
    - ventana_variable=True: usa LSTMModel con pack_padded_sequence.
    """

    # --- 1. Crear DataLoaders ---
    if ventana_variable:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        # Crear secuencias fijas
        X_train_seq, y_train_seq = crear_secuencias(X_train, y_train, ventana)
        X_test_seq, y_test_seq = crear_secuencias(X_test, y_test, ventana)

        train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                      torch.tensor(y_train_seq, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                                     torch.tensor(y_test_seq, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # --- 2. Preparar modelo y criterio ---
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device)) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- 3. Entrenamiento ---
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

    # --- 4. Evaluación ---
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

    # --- 5. Métricas ---
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

    # 1. Obtenemos la lista completa de índices de clases (0, 1, ..., 5)
    all_labels = np.arange(len(le.classes_))

    resultados = {
        'accuracy': accuracy_score(all_true, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_true, all_preds),
        'f1_global': f1_score(all_true, all_preds, average='weighted'),
        
        # 2. MODIFICAMOS ESTAS TRES LÍNEAS:
        # Añadimos 'labels=all_labels' y 'zero_division=0'
        'f1_por_clase': f1_score(all_true, all_preds, average=None, labels=all_labels, zero_division=0),
        'recall_por_clase': recall_score(all_true, all_preds, average=None, labels=all_labels, zero_division=0),
        'precision_por_clase': precision_score(all_true, all_preds, average=None, labels=all_labels, zero_division=0),
        
        'kappa': cohen_kappa_score(all_true, all_preds),
        'y_true': all_true,
        'y_pred': all_preds,
        'model': model,
        'classes': le.classes_,
        'class_distribution': {clase: np.sum(all_true == i) for i, clase in enumerate(le.classes_)},
        'confusion_matrix': confusion_matrix(all_true, all_preds, labels=all_labels), # Opcional: asegurar matriz completa también
        'ventana_usada': ventana,
        'ventana_variable': ventana_variable
    }

    return resultados






#----EXTRAS LSTM----#
def crear_secuencias(X, y, ventana):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - ventana):
        # Para cada posicion, tomar una secuencia de longitud de la ventana
        X_seq.append(X[i:i+ventana])
        y_seq.append(y[i+ventana-1])  # Etiqueta del ultimo paso
    return np.array(X_seq), np.array(y_seq)



def crear_dataloaders_ventana(X_train, y_train, X_test, y_test, batch_size=64, ventana=50, ventana_variable=False):
    """
    Crea DataLoaders para LSTM con ventana fija o variable (padding).
    """
    if ventana_variable:
        # Secuencias con longitud variable usando collate_fn
        train_dataset = SequenceDataset(X_train, y_train)
        test_dataset = SequenceDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, test_loader, None, None, None, None

    else:
        # Secuencias fijas
        X_train_seq, y_train_seq = crear_secuencias(X_train, y_train, ventana)
        X_test_seq, y_test_seq = crear_secuencias(X_test, y_test, ventana)
        train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                      torch.tensor(y_train_seq, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                                     torch.tensor(y_test_seq, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, test_loader, X_train_seq, X_test_seq, y_train_seq, y_test_seq



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
        altos_dict = {sensor: valor for sensor, valor in altos.items()}

        if len(altos) > 0:
            print(f"  Sensores ALTOS: {', '.join([f'{s}:{v:.2f}' for s,v in altos.items()])}")
        
        #Sensores más bajos que el promedio
        bajos = medians[medians < -0.5].sort_values()
        bajos_dict = {sensor: valor for sensor, valor in bajos.items()}

        if len(bajos) > 0:
            print(f"  Sensores BAJOS: {', '.join([f'{s}:{v:.2f}' for s,v in bajos.items()])}")
        
        #Sensor más característico
        caracteristico = medians.abs().idxmax()
        
        print(f"  Sensor más CARACTERÍSTICO: {caracteristico} ({medians[caracteristico]:.2f})")


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