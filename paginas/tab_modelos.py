import streamlit  as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import torch

from utils import preprocesar_datos, split_data, evaluar_modelo_scikit, entrenar_evaluar_pytorch
from modelos.mlp import MLP
from modelos.mlp_mejorado import MLP_mejorado
from modelos.lstm import LSTM_basico
from plots import plot_confusion_matrix



from utils import *
from modelos.mlp import MLP

def run():
    st.header("🤖 Modelado de Machine Learning")

    # Cargar datos desde sesión
    df = st.session_state['df']
    
    # Filtro de fechas en sidebar (solo para este tab)
    if 'Timestamp' in df.columns:
        st.sidebar.subheader("Filtro Temporal")
        
        # Convertir a datetime si no lo está
        if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Obtener fechas mínimas y máximas
        min_date = df['Timestamp'].min()
        max_date = df['Timestamp'].max()
        
        # Widget de selección de rango de fechas
        selected_dates = st.sidebar.date_input(
            "Selecciona el intervalo de fechas:",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        
        # Aplicar filtro si se seleccionaron dos fechas
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date)
            
            # Calcular diferencia en días
            dias_diferencia = (end_datetime - start_datetime).days + 1  # +1 para incluir ambos extremos
            
            # Validar intervalo mínimo de 7 días
            if dias_diferencia < 7:
                st.sidebar.error(f"Intervalo insuficiente: {dias_diferencia} días. Se requiere al menos 7 días de datos para entrenar modelos.")
                st.sidebar.info(f"Selecciona un intervalo de al menos 7 días.")
                
                # Mostrar dataset completo como fallback
                df_filtered = df
                st.sidebar.warning("Usando dataset completo por intervalo insuficiente")
            else:
                # Filtrar el dataframe
                df_filtered = df[(df['Timestamp'] >= start_datetime) & 
                                (df['Timestamp'] <= end_datetime)]
                st.sidebar.success(f"Intervalo válido: {dias_diferencia} días")
                st.sidebar.success(f"Filtrado: {len(df_filtered)} registros de {start_date} a {end_date}")
                
        else:
            df_filtered = df
            st.sidebar.info("Selecciona un intervalo de fechas válido (inicio y fin)")
    else:
        df_filtered = df
        st.sidebar.info("No hay columna de fechas disponible para filtrar")

    # Parámetros interactivos
    st.sidebar.subheader("Parámetros de entrenamiento")
    test_size = st.sidebar.slider("Tamaño del test (%)", 10, 30, 20)
    
    # Eliminar encoder viejo porque puede que hayan cambiado las dimensiones
    if 'le' in st.session_state:
        del st.session_state['le']
    
    

    # Preprocesamiento y split
    X, y, le = preprocesar_datos(df_filtered)
    # Guardar el nuevo encoder
    st.session_state['le'] = le
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size/100)

    # Crear pestañas para los modelos
    tab1, tab2, tab3, tab8 = st.tabs([
        "Modelos Shallow Learning",
        "MLP Básico",
        "MLP Avanzado",
        "LSTM"
    ])

    # ----- TAB 1: Shallow Learning -----
    with tab1:
        tab4, tab5, tab6, tab7 = st.tabs([
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "Support Vector Machine (SVM)"
        ])
        
        with tab4:
            st.subheader("Random Forest")
            st.markdown("En este caso, Random Forest es un buen modelo porque combina múltiples árboles de decisión para **capturar relaciones complejas entre los sensores**. Es robusto frente a sobreajuste si se limita la profundidad de los árboles y puede manejar de manera eficiente el desbalance de clases utilizando el parámetro ***class_weight='balanced***'. Además, es capaz de detectar interacciones no lineales entre las variables, algo frecuente en datos de fallos mecánicos.")
            
            st.markdown("---")

            st.write("Parámetros seleccionados:")
            st.write(f"- Tamaño del test: {test_size}%")
            
            st.markdown("---")

            st.write("Parámetros por defecto:")
            st.write("- Modelo: Random Forest")
            st.write("- n_estimators: 10")
            st.write("- random_state: 111")
            st.write(f"- class_weight: 'balanced'")
            
            st.markdown("---")

            if st.button("Entrenar modelo Random Forest"):
                col1, col2 = st.columns(2)

                rf_model = RandomForestClassifier(n_estimators=10, random_state=111, class_weight='balanced')
                resultados_rf = evaluar_modelo_scikit(rf_model, X_train, X_test, y_train, y_test, "Random Forest", le)
                mostrar_resultados_modelo(resultados_rf, le)
                
        with tab5:
            st.subheader("Gradient Boosting")
            st.markdown("Gradient Boosting es adecuado porque construye árboles de manera secuencial, corrigiendo los errores de los anteriores. Esto le permite capturar patrones más complejos que un único árbol y mejorar la **precisión en problemas multiclas**. También puede ajustarse para manejar desequilibrio en las clases y es **menos propenso a predecir únicamente la clase mayoritaria** en datasets desbalanceados.")
            
            st.markdown("---")

            st.write("Parámetros seleccionados:")
            st.write(f"- Tamaño del test: {test_size}%")
            
            st.markdown("---")

            st.write("Parámetros por defecto:")
            st.write("- Modelo: Gradient Boosting")
            st.write("- max_iter: 100")
            st.write("- random_state: 111")
            st.write(f"- class_weight: 'balanced'")
            
            st.markdown("---")
            
            if st.button("Entrenar modelo Gradient Boosting"):
                col1, col2 = st.columns(2)
                
                gb_model = HistGradientBoostingClassifier(max_iter=100, random_state=111, class_weight='balanced')
                resultados_gb = evaluar_modelo_scikit(gb_model, X_train, X_test, y_train, y_test, "HistGradientBoosting", le)
                mostrar_resultados_modelo(resultados_gb, le)           
                
        with tab6:
            st.subheader("XGBoost")
            st.markdown("XGBoost es una versión optimizada de Gradient Boosting, diseñada para ser más rápida y eficiente. Permite un control fino sobre regularización, profundidad de los árboles y balance de clases, lo que lo hace especialmente útil cuando algunas **fallas son minoritarias y difíciles de detectar**. Su eficiencia computacional permite entrenar modelos robustos sin comprometer el rendimiento.")
            
            st.markdown("---")
            
            st.write("Parámetros seleccionados:")
            st.write(f"- Tamaño del test: {test_size}%")
            st.markdown("---")

            st.write("Parámetros por defecto:")
            st.write("- Modelo: XGBoost")
            st.write("- n_estimators: 100")
            st.write("- random_state: 111")
            st.write("- use_label_encoder: False")
            st.write("- eval_metric: mlogloss")
            
            st.markdown("---")

            if st.button("Entrenar modelo XGBoost"):
                col1, col2 = st.columns(2)
                ratio = np.bincount(y_train)[0] / np.bincount(y_train)[1]

                xgb_model = XGBClassifier(n_estimators=100, random_state=111, use_label_encoder=False, eval_metric='mlogloss', scale_pos_weight=ratio)
                resultados_xgb = evaluar_modelo_scikit(xgb_model, X_train, X_test, y_train, y_test, "XGBoost", le)
                mostrar_resultados_modelo(resultados_xgb, le)
                
        with tab7:
            st.subheader("Support Vector Machine (SVM)")
            st.markdown("SVM es útil en este problema porque busca un **margen óptimo entre clases, lo que ayuda a separar fallos minoritarios de la clase mayoritaria**. Usando un kernel no lineal (como RBF), puede capturar relaciones complejas entre los sensores que no se ven linealmente.")
            
            st.markdown("---")
            
            st.write("Parámetros seleccionados:")
            st.write(f"- Tamaño del test: {test_size}%")
            
            st.markdown("---")

            st.write("Parámetros por defecto:")
            st.write("- Modelo: Support Vector Machine (SVM)")
            st.write("- kernel: rbf")
            st.write("- probability: True")
            st.write("- random_state: 111")
            st.write(f"- class_weight: 'balanced'")
            
            st.markdown("---")

            if st.button("Entrenar modelo SVM"):
                col1, col2 = st.columns(2)
                svm_model = SVC(kernel='rbf', probability=True, random_state=111, class_weight='balanced')
                resultados_svm = evaluar_modelo_scikit(svm_model, X_train, X_test, y_train, y_test, "SVM", le)                
                mostrar_resultados_modelo(resultados_svm, le)


    # ----- TAB 2: MLP Básico -----
    with tab2:
        st.subheader("MLP Básico")
        st.write("Selección de parámetros específicos:")
        epochs = st.slider("Número de epochs", 10, 200, 50, key="Epochs basico")
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=3, key="Batch basico")
        lr = st.number_input("Learning rate", min_value=0.0001, max_value=0.05, value=0.001, step=0.0001, key="LR basico", format="%.4f")       
        st.markdown("---")
        
        st.write("Parámetros seleccionados:")
        st.write(f"- epochs: {epochs}")
        st.write(f"- batch_size: {batch_size}")
        st.write(f"- learning_rate: {lr:.4f}")
        st.write(f"- Tamaño del test: {test_size}%")

        st.markdown("---")
        if st.button("Entrenar MLP Básico"):
            col1, col2 = st.columns(2)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            input_dim = X_train.shape[1]
            output_dim = len(np.unique(y_train))
            model = MLP(input_dim=input_dim, output_dim=output_dim)


            resultados_mlp = entrenar_evaluar_pytorch(
                model=model,
                X_train=X_train.values,
                X_test=X_test.values,
                y_train=y_train.values,
                y_test=y_test.values,
                le=le,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device
            )

            mostrar_resultados_modelo(resultados_mlp, le)
                    
           

    # ----- TAB 3: MLP Avanzado -----
    with tab3:
        st.subheader("MLP Avanzado")
        st.write("Selección de parámetros específicos:")
        epochs = st.slider("Número de epochs", 10, 200, 50, key="Epochs avanzado")
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=3, key="Batch avanzado")
        lr = st.number_input("Learning rate", min_value=0.0001, max_value=0.05, value=0.001, step=0.0001, key="LR avanzado", format="%.4f")       
        usar_class_weights = st.checkbox("Usar class weights", value=False, key="Class weights avanzado")
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1, key="Dropout avanzado")
        st.markdown("---")
        
        st.write("Parámetros seleccionados:")
        st.write(f"- epochs: {epochs}")
        st.write(f"- batch_size: {batch_size}")
        st.write(f"- learning_rate: {lr:.4f}")
        st.write(f"- usar_class_weights: {usar_class_weights}")
        st.write(f"- Dropout: {dropout}")
        st.write(f"- Tamaño del test: {test_size}%")
        

        st.markdown("---")
        if st.button("Entrenar MLP Avanzado"):
            col1, col2 = st.columns(2)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            input_dim = X_train.shape[1]
            output_dim = len(np.unique(y_train))
            model = MLP_mejorado(input_dim=input_dim, output_dim=output_dim, dropout_rate=dropout)

            class_weights = None
            if usar_class_weights:
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

            resultados_mlp = entrenar_evaluar_pytorch(
                model=model,
                X_train=X_train.values,
                X_test=X_test.values,
                y_train=y_train.values,
                y_test=y_test.values,
                le=le,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                class_weights=class_weights
            )

            mostrar_resultados_modelo(resultados_mlp, le)
            
            
            
            
    # ----- TAB 8: LSTM -----
    with tab8:
        st.subheader("LSTM")
        st.write("Selección de parámetros específicos:")
        
        epochs = st.slider("Número de epochs", 10, 200, 50, key="Epochs_LSTM")
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=3, key="Batch_LSTM")
        lr = st.number_input("Learning rate", min_value=0.0001, max_value=0.05, value=0.001, step=0.0001, key="LR_LSTM", format="%.4f")       
        usar_class_weights = st.checkbox("Usar class weights", value=False, key="Class_weights_LSTM")
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1, key="Dropout_LSTM")
        hidden_dim = st.number_input("Hidden dim", min_value=16, max_value=256, value=64, step=16, key="Hidden_LSTM")
        num_layers = st.number_input("Número de capas LSTM", min_value=1, max_value=4, value=2, step=1, key="Layers_LSTM")
        ventana = st.slider("Tamaño de la ventana (timesteps)", 10, 200, 50, step=5, key="Ventana_LSTM")
        st.markdown("---")
        
        st.write("Parámetros seleccionados:")
        st.write(f"- epochs: {epochs}")
        st.write(f"- batch_size: {batch_size}")
        st.write(f"- learning_rate: {lr:.4f}")
        st.write(f"- usar_class_weights: {usar_class_weights}")
        st.write(f"- Dropout: {dropout}")
        st.write(f"- Tamaño del test: {test_size}%")
        st.write(f"- Hidden dim: {hidden_dim}")
        st.write(f"- Número de capas: {num_layers}")
        st.write(f"- Ventana (timesteps): {ventana}")
        
        

        st.markdown("---")
        if st.button("Entrenar LSTM"):
            col1, col2 = st.columns(2)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Crear secuencias para LSTM
            X_train_seq, y_train_seq = crear_secuencias(X_train.values, y_train.values, ventana)
            X_test_seq, y_test_seq = crear_secuencias(X_test.values, y_test.values, ventana)


            input_dim = X_train_seq.shape[2]
            output_dim = len(np.unique(y_train_seq))
                        
            model = LSTM_basico(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        output_dim=output_dim,
                        dropout=dropout
                    )
            
            class_weights = None
            if usar_class_weights:
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)

            resultados_lstm = entrenar_evaluar_pytorch(
                model=model,
                X_train=X_train_seq,
                X_test=X_test_seq,
                y_train=y_train_seq,
                y_test=y_test_seq,
                le=le,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                class_weights=class_weights
            )

            mostrar_resultados_modelo(resultados_lstm, le)
