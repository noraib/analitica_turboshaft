import streamlit  as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


#----CONFIGURACION BASICA----#
st.set_page_config(
    page_title="Detección de Fallos en Motores Turboshaft",
    layout="wide"
)
st.title("Detección de Fallos en Motores Turboshaft")
st.markdown("---")



#Carga datos
file_name = "data/Helicopter_Turboshaft_Fault_Detection.csv"

@st.cache_data
def load_data():
    try:
    #Cargar dataset
        df = pd.read_csv(file_name)
        return df
    except FileNotFoundError:
        st.text(f"No se ha encontrado el archivo {file_name} en la carpeta del proyecto")
        return None

    except Exception as e:
        st.text(f"Error cargando el archivo: {e}")
        return None


load_data()
#Cargar datos solo una vez
if 'df' not in st.session_state:
    df = load_data()
    if df is not None:
        st.session_state['df'] = df
    else:
        st.sidebar.error("No se pudo cargar el dataset")



#Tabs
tab1, tab2, tab3 = st.tabs(["Contexto del Proyecto",
                            "Análisis Exploratorio (EDA)",
                            "Modelado de Machine Learning"])




#----TAB 1: INFORMACION BASICA----#
with tab1:
    st.header("Contexto del Proyecto")
    container1 = st.container(width=6000)
    container1.subheader("Evolución de la Industria 4.0")
    container1.markdown("La industria ha ido evolucionando a lo largo del tiempo, desde el siglo XVIII, cuando surgieron las primeras máquinas que reemplazaron el trabajo manual de los trabajadores, hasta hoy, cuando siguen surgiendo nuevas tecnologías para facilitar, mejorar y optimizar el rendimiento en la fabricación y el mantenimiento de sistemas industriales.")        
    container1.markdown("Tradicionalmente, estos avances se han enfocado en procesos repetitivos o de producción masiva, como la fabricación de tornillos, puertas o componentes mecánicos. Sin embargo, en las últimas décadas, la digitalización y la inteligencia artificial han abierto la puerta a aplicar los mismos principios de optimización en sectores más complejos y críticos, como la industria aeronáutica.")        
    container1.subheader("Importancia en la Industria Aeronáutica")
    container1.markdown("En este ámbito, los helicópteros son un caso especialmente interesante, pues su funcionamiento depende de sistemas muy sensibles, los cuales están sometidos a grandes esfuerzos mecánicos y donde cualquier fallo puede repercutir en consecuencias graves. Es por esto que es de vital importancia ser capaz de **detectar y predecir fallos en uno de los componentes más importantes del helicóptero, como son los motores** (corazón de helicóptero), permitiendo así mejorar la seguridad y la eficiencia en su construcción.")        
    st.markdown("---")
    container1.subheader("Carga dataset")
    container1.markdown("En esta primera fase cargaremos y analizaremos la información básica de los datos del dataset escogido de kaggle:"
    "https://www.kaggle.com/datasets/ziya07/helicopter-turboshaft-detection-dataset")

    if 'df' in st.session_state:
        st.subheader("📊 Información del Dataset")
        
        df = st.session_state['df']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total de Columnas", df.shape[1])
        with col3:
            st.metric("Columnas Numéricas", f"{df.select_dtypes(include=[np.number]).shape[1]}")
        
        with st.expander("👁️ Ver primeros registros"):
            st.dataframe(df.head())
        
        with st.expander("📋 Ver estructura del dataset"):
            buffer = []
            for col in df.columns:
                buffer.append({
                    "Columna": col,
                    "Tipo": str(df[col].dtype),
                    "No Nulos": df[col].notnull().sum(),
                    "Nulos": df[col].isnull().sum(),
                    "Únicos": df[col].nunique()
                })
            st.dataframe(pd.DataFrame(buffer))

with tab2:
    st.header("Análisis Exploratorio (EDA)")
    # Verificar si los datos están cargados
    if 'df' not in st.session_state:
        st.warning("Los datos no se han cargado. Ve a la pestaña 'Contexto' y verifica la carga.")
    else:
        df = st.session_state['df']
        st.success(f"Analizando dataset con {df.shape[0]} registros")
        
        # Opciones de análisis
        analysis_option = st.selectbox(
            "Selecciona tipo de análisis:",
            ["📊 Estadísticas Descriptivas", 
             "📈 Distribuciones", 
             "🔗 Matriz de Correlación",
             "🎯 Análisis por Clase"]
        )
        
        # 1. ESTADÍSTICAS DESCRIPTIVAS
        if analysis_option == "📊 Estadísticas Descriptivas":
            st.subheader("Estadísticas Descriptivas")
            
            # Mostrar estadísticas básicas
            st.dataframe(df.describe())
            
            # Mostrar información de tipos de fallo si existe
            if 'Fault_Label' in df.columns:
                st.subheader("Distribución de Fallos")
                fault_counts = df['Fault_Label'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Conteo por clase:")
                    st.dataframe(fault_counts)
                
                with col2:
                    # Gráfico de barras
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(fault_counts.index, fault_counts.values, color='skyblue')
                    ax.set_title('Distribución de Tipos de Fallo')
                    ax.set_xlabel('Tipo de Fallo')
                    ax.set_ylabel('Cantidad')
                    ax.tick_params(axis='x', rotation=45)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                               f'{int(height)}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
        
        # 2. DISTRIBUCIONES
        elif analysis_option == "📈 Distribuciones":
            st.subheader("Análisis de Distribuciones")
            
            # Seleccionar columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.multiselect(
                    "Selecciona columnas para visualizar:",
                    numeric_cols,
                    default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
                )


