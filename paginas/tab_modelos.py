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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


from utils import *
from plots import * 



def run():
    st.header("🤖 Modelado de machine learning")
    st.write("En esta sección podrás entrenar y evaluar diferentes modelos de clasificación para detectar fallos en motores turboshaft.")

    st.markdown("---")

    target_col = "Fault_Label"

    if target_col not in df.columns:
        st.error("No se encontró la columna Fault_Label en el dataset.")
        st.stop()

    default_features = [c for c in df.columns if c != target_col]

    st.subheader("Variables predictoras (X)")
    feature_cols = st.multiselect(
        "Selecciona las variables de entrada para el modelo:",
        df.columns,
        default=default_features
    )

    st.markdown("---")


    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Modelos disponibles")
        model_options = [
           "Logistic Regression",
            "Support Vector Machine (SVM)",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "Red Neuronal (MLP - Keras)"
        ]

        selected_models = st.multiselect(
            "Selecciona uno o varios modelos para entrenar:",
            model_options,
            default=["Random Forest"]
        )

    with col_right:
        st.subheader("Parámetros del conjunto de entrenamiento")
        test_size = st.slider(
            "Tamaño del conjunto de test (%):",
            min_value=10, max_value=50, value=20, step=5
        )

    st.markdown("---")

    run_button = st.button("🚀 Entrenar Modelos")

    if run_button:
        st.subheader("Resultados del entrenamiento")

        for model_name in selected_models:
            st.write(f"## 🔽 Resultados para **{model_name}**")
                
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", "—")
            with col2:
                st.metric("F1-Score", "—")
            with col3:
                st.metric("Muestras en test", f"{test_size}%")

            st.write("### Matriz de Confusión:")
            st.empty()

            if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                st.write("### Importancia de Variables:")
                st.empty()

            st.write("### Predicciones de ejemplo:")
            st.dataframe(df[feature_cols].head().assign(Predicción="—"))
        
            st.markdown("---")


