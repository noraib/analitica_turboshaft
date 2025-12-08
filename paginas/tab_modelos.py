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

from plots import plot_confusion_matrix



from utils import *
from modelos.mlp import MLP


def run():
    st.header("🤖 Modelado de machine learning")
    
    #Cargar datos de la sesion
    df = st.session_state['df']
    
    #Configuracion interactiva
    st.sidebar.subheader("Parámetros de entrenamiento")
    
    #TEST SIZE
    test_size = st.sidebar.slider("Tamaño del test (%)", 10, 30, 10)

    
    X, y, le = preprocesar_datos(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size/100)
    
    tab1, tab2, tab3 = st.tabs([
        "Modelos Shallow Learning",
        "MLP Básico",
        "MLP Avanzado"
    ])
    


def run():
    st.header("🤖 Modelado de Machine Learning")

    # Cargar datos desde sesión
    df = st.session_state['df']

    # Parámetros interactivos
    st.sidebar.subheader("Parámetros de entrenamiento")
    test_size = st.sidebar.slider("Tamaño del test (%)", 10, 30, 20)
    

    # Preprocesamiento y split
    X, y, le = preprocesar_datos(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size/100)

    # Crear pestañas para los modelos
    tab1, tab2, tab3 = st.tabs([
        "Modelos Shallow Learning",
        "MLP Básico",
        "MLP Avanzado"
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

                with col1:
                    st.subheader("Métricas principales")
                    #Solo las que muestren numero concreto
                    metrics = {k: [v] for k, v in resultados_rf.items() if isinstance(v, (int, float))}
                    df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=["Valor"])
                    st.dataframe(df_metrics)
                with col2:
                    st.subheader("Matríz de Confusión")
                    fig = plot_confusion_matrix(y_true=resultados_rf['y_true'],
                                y_pred=resultados_rf['y_pred'],
                                class_names=le.classes_)
                    st.pyplot(fig)

                
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
                
                with col1:
                    st.subheader("Métricas principales")
                    metrics = {k: [v] for k, v in resultados_gb.items() if isinstance(v, (int, float))}
                    df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=["Valor"])
                    st.dataframe(df_metrics)

                with col2:
                    st.subheader("Matríz de Confusión")
                    fig = plot_confusion_matrix(y_true=resultados_gb['y_true'],
                                y_pred=resultados_gb['y_pred'],
                                class_names=le.classes_)
                    st.pyplot(fig)
             
             
                
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
            
                with col1:
                    st.subheader("Métricas principales")
                    metrics = {k: [v] for k, v in resultados_xgb.items() if isinstance(v, (int, float))}
                    df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=["Valor"])
                    st.dataframe(df_metrics)

                with col2:
                    st.subheader("Matríz de Confusión")
                    fig = plot_confusion_matrix(y_true=resultados_xgb['y_true'],
                                y_pred=resultados_xgb['y_pred'],
                                class_names=le.classes_)
                    st.pyplot(fig)
                    
                
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
                
                with col1:
                    st.subheader("Métricas principales")
                    st.dataframe(resultados_svm)

                with col2:
                    st.subheader("Matríz de Confusión")
                    fig = plot_confusion_matrix(y_true=resultados_svm['y_true'],
                                y_pred=resultados_svm['y_pred'],
                                class_names=le.classes_)
                    st.pyplot(fig)
                    

              



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

            with col1:
                st.subheader("Métricas principales")
                metrics = {k: [v] for k, v in resultados_mlp.items() if isinstance(v, (int, float))}
                df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=["Valor"])
                st.dataframe(df_metrics)

            with col2:
                st.subheader("Matríz de Confusión")
                fig = plot_confusion_matrix(y_true=resultados_mlp['y_true'],
                            y_pred=resultados_mlp['y_pred'],
                            class_names=le.classes_)
                st.pyplot(fig)
                    
           

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

            with col1:
                st.subheader("Métricas principales")
                metrics = {k: [v] for k, v in resultados_mlp.items() if isinstance(v, (int, float))}
                df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=["Valor"])
                st.dataframe(df_metrics)

            with col2:
                st.subheader("Matríz de Confusión")
                fig = plot_confusion_matrix(y_true=resultados_mlp['y_true'],
                            y_pred=resultados_mlp['y_pred'],
                            class_names=le.classes_)
                st.pyplot(fig)




# def run():
#     st.header("🤖 Modelado de machine learning")
#     st.write("En esta sección podrás entrenar y evaluar diferentes modelos de clasificación para detectar fallos en motores turboshaft.")

#     st.markdown("---")

#     target_col = "Fault_Label"

#     if target_col not in df.columns:
#         st.error("No se encontró la columna Fault_Label en el dataset.")
#         st.stop()

#     default_features = [c for c in df.columns if c != target_col]

#     st.subheader("Variables predictoras (X)")
#     feature_cols = st.multiselect(
#         "Selecciona las variables de entrada para el modelo:",
#         df.columns,
#         default=default_features
#     )

#     st.markdown("---")


#     col_left, col_right = st.columns([1, 1])

#     with col_left:
#         st.subheader("Modelos disponibles")
#         model_options = [
#            "Logistic Regression",
#             "Support Vector Machine (SVM)",
#             "Random Forest",
#             "Gradient Boosting",
#             "XGBoost",
#             "Red Neuronal (MLP - Keras)"
#         ]

#         selected_models = st.multiselect(
#             "Selecciona uno o varios modelos para entrenar:",
#             model_options,
#             default=["Random Forest"]
#         )

#     with col_right:
#         st.subheader("Parámetros del conjunto de entrenamiento")
#         test_size = st.slider(
#             "Tamaño del conjunto de test (%):",
#             min_value=10, max_value=50, value=20, step=5
#         )

#     st.markdown("---")

#     run_button = st.button("🚀 Entrenar Modelos")

#     if run_button:
#         st.subheader("Resultados del entrenamiento")

#         for model_name in selected_models:
#             st.write(f"## 🔽 Resultados para **{model_name}**")
                
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Accuracy", "—")
#             with col2:
#                 st.metric("F1-Score", "—")
#             with col3:
#                 st.metric("Muestras en test", f"{test_size}%")

#             st.write("### Matriz de Confusión:")
#             st.empty()

#             if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
#                 st.write("### Importancia de Variables:")
#                 st.empty()

#             st.write("### Predicciones de ejemplo:")
#             st.dataframe(df[feature_cols].head().assign(Predicción="—"))
        
#             st.markdown("---")


