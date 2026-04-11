import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Data Analyzer Pro", layout="wide")

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("📊 Analizador de Datos Universitario")
st.markdown("""
Esta aplicación permite cargar conjuntos de datos externos o generar datos sintéticos 
para realizar análisis estadísticos rápidos.
""")

# --- SIDEBAR (BARRA LATERAL) ---
st.sidebar.header("Configuración de Datos")

# Opción para elegir la fuente de datos
opcion_datos = st.sidebar.radio(
    "Selecciona la fuente de datos:",
    ("Subir archivo CSV", "Generar datos sintéticos")
)

df = None # Variable para almacenar nuestro DataFrame

if opcion_datos == "Subir archivo CSV":
    uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type="csv")
    if uploaded_file is not None:
        try:
            # Intentamos leer el CSV
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

else:
    st.sidebar.subheader("Parámetros de Distribución")
    tipo_dist = st.sidebar.selectbox("Tipo de distribución:", ["Normal", "Uniforme", "Sesgada (Exponential)"])
    n_samples = st.sidebar.slider("Tamaño de la muestra (n):", 10, 1000, 100)
    
    # Generación de datos usando NumPy
    if tipo_dist == "Normal":
        data = np.random.normal(loc=0, scale=1, size=n_samples)
    elif tipo_dist == "Uniforme":
        data = np.random.uniform(low=0, high=10, size=n_samples)
    else:
        data = np.random.exponential(scale=1.0, size=n_samples)
    
    df = pd.DataFrame(data, columns=["Valores_Generados"])

# --- VISUALIZACIÓN Y ESTADÍSTICAS ---
if df is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("👀 Preview de los datos")
        st.write(df.head())

    with col2:
        st.subheader("📉 Estadísticas Básicas")
        # Seleccionamos solo columnas numéricas para las estadísticas
        stats = df.describe().T[['mean', 'std', 'min', 'max']]
        stats['n'] = len(df) # Añadimos el tamaño de la muestra manualmente
        st.write(stats)
else:
    st.info("Esperando carga o generación de datos...")