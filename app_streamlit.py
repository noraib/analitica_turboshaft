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


#----CONFIGURACION BASICA----#
st.set_page_config(
    page_title="Detección de Fallos en Motores Turboshaft",
    layout="wide"
)
st.title("🚁 Detección de Fallos en Motores Turboshaft")
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
    st.header("🌐 Contexto del Proyecto")
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
    st.header("🔍 Análisis Exploratorio (EDA)")
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
             "⌚️ Tendencia temporal",
             "🔗 Matriz de Correlación",
             "📡 Radar por fallo",
             "🧭 PCA",
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

                st.subheader("Parallel Coordinates por Tipo de Fallo")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                le = LabelEncoder()
                df["Fault_Code"] = le.fit_transform(df["Fault_Label"])

                fig = px.parallel_coordinates(
                    df,
                    dimensions=numeric_cols,
                    color="Fault_Code",
                    labels={col: col for col in numeric_cols},
                    title="Parallel coordinates por tipo de tallo",
                    color_continuous_scale=px.colors.diverging.Tealrose,
                )

                st.plotly_chart(fig, use_container_width=True)
                        
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
            if selected_cols:
                for col in selected_cols:

                    data = df[col].dropna()
                    kde = gaussian_kde(data)

                    x_vals = np.linspace(data.min(), data.max(), 300)
                    y_vals = kde(x_vals)

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="royalblue", width=3),
                        name=col
                    ))

                    fig.update_layout(
                        title=f"Distribución de {col}",
                        xaxis_title=col,
                        yaxis_title="Densidad",
                        height=400,
                        template="simple_white",
                        margin=dict(l=50, r=50, t=60, b=50)
                    )

                    st.plotly_chart(fig, use_container_width=True)

        elif analysis_option == "🔗 Matriz de Correlación":
            st.subheader("🔗 Matriz de Correlación")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            st.write("Selecciona columnas para incluir en la matriz:")
            selected_corr_cols = st.multiselect(
                "Selecciona columnas para visualizar:",
                numeric_cols,
                default=[]
            )

            if len(selected_corr_cols) == 0:
                corr_df = df[numeric_cols].corr()
                st.info("No has seleccionado ninguna columna. Se mostrará la matriz completa.")
            else:
                corr_df = df[selected_corr_cols].corr()

            fig = px.imshow(
                corr_df,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                zmin=-1, #para que ese entre 1 y -1
                zmax=1,
                aspect="auto",
                title="Matriz de correlación"
            )

            fig.update_layout(
                height=800 if len(corr_df.columns) > 6 else 600,
                margin=dict(l=40, r=40, t=60, b=40),
                coloraxis_colorbar=dict(
                    title="Correlación",
                    ticks="outside"
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_option == "⌚️ Tendencia temporal":
            st.subheader("⌚️ Tendencias temporales (Time-Series)")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            col = st.selectbox("Selecciona una variable para analizar su evolución temporal:", numeric_cols)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                line=dict(color="royalblue", width=2),
                name=col
            ))

            fig.update_layout(
                title=f"Evolución Temporal de {col}",
                xaxis_title="Tiempo (índice de registro)",
                yaxis_title=col,
                height=400,
                template="simple_white"
            )

            st.plotly_chart(fig, use_container_width=True)

        elif analysis_option == "📡 Radar por fallo":
            st.subheader("📡 Radar chart por tipo de fallo")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            col = st.selectbox("Selecciona una variable numérica:", numeric_cols)

            avg = df.groupby("Fault_Label")[col].mean().reset_index()

            fig = px.line_polar(
                avg,
                r=col,
                theta="Fault_Label",
                line_close=True,
                title=f"Radar Chart de {col} por Tipo de Fallo",
                range_r=[avg[col].min(), avg[col].max()],
                template="simple_white"
            )

            fig.update_traces(fill="toself", line=dict(width=3))

            st.plotly_chart(fig, use_container_width=True)

        elif analysis_option == "🧭 PCA":
            st.subheader("🧭 PCA de los 3 principales componentes")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            clean_numeric = [c for c in numeric_cols if c not in ["Fault_Code"]]

            X = df[clean_numeric].dropna()
            y = df["Fault_Label"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=3)
            pcs = pca.fit_transform(X_scaled)
            loadings = pca.components_.T

            pca_df = pd.DataFrame({
                "PC1": pcs[:, 0],
                "PC2": pcs[:, 1],
                "PC3": pcs[:, 2],
                "Fault_Label": y
            })

            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=pca_df["PC1"],
                y=pca_df["PC2"],
                z=pca_df["PC3"],
                mode='markers',
                marker=dict(
                    size=4,
                    opacity=0.75,
                    color=pca_df["Fault_Label"].astype('category').cat.codes,
                    colorscale="Viridis"
                ),
                text=y,
                hovertemplate="<b>Clase:</b> %{text}",
                name="Muestras"
            ))

            arrow_scale = 3

            for i, feature in enumerate(clean_numeric):
                fig.add_trace(go.Scatter3d(
                    x=[0, loadings[i, 0] * arrow_scale],
                    y=[0, loadings[i, 1] * arrow_scale],
                    z=[0, loadings[i, 2] * arrow_scale],
                    mode='lines+text',
                    line=dict(color="black", width=6),
                    text=[None, feature],
                    textposition="top center",
                    name=feature,
                    showlegend=False
                ))

            var_exp = pca.explained_variance_ratio_ * 100
            fig.update_layout(
                title=(
                    f"PCA 3D Biplot<br>"
                    f"Varianza explicada: "
                    f"PC1={var_exp[0]:.1f}% • "
                    f"PC2={var_exp[1]:.1f}% • "
                    f"PC3={var_exp[2]:.1f}%"
                ),
                scene=dict(
                    xaxis_title=f"PC1 ({var_exp[0]:.1f}%)",
                    yaxis_title=f"PC2 ({var_exp[1]:.1f}%)",
                    zaxis_title=f"PC3 ({var_exp[2]:.1f}%)",
                ),
                height=700,
                template="simple_white",
                margin=dict(l=0, r=0, t=80, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)




