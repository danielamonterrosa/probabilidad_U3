import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, skew

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
    
    
# --- MÓDULO DE VISUALIZACIÓN ---
if df is not None:
    st.divider()
    st.header("📈 Visualización de Distribuciones")

    # Selección de variable numérica
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(cols_numericas) > 0:
        col_seleccionada = st.selectbox("Selecciona la variable a analizar:", cols_numericas)
        datos = df[col_seleccionada].dropna()

        # Opciones de visualización
        col_op1, col_op2 = st.columns(2)
        with col_op1:
            mostrar_kde = st.checkbox("Superponer KDE (Curva de densidad)", value=True)
        with col_op2:
            orientacion_box = st.radio("Orientación Boxplot:", ("Horizontal", "Vertical"), horizontal=True)

        # Creación de los gráficos
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Histograma
        sns.histplot(datos, kde=mostrar_kde, ax=ax[0], color="#4F8BF9")
        ax[0].set_title(f"Histograma de {col_seleccionada}")

        # 2. Boxplot
        if orientacion_box == "Horizontal":
            sns.boxplot(x=datos, ax=ax[1], color="#F94F4F")
        else:
            sns.boxplot(y=datos, ax=ax[1], color="#F94F4F")
        ax[1].set_title(f"Boxplot de {col_seleccionada}")

        st.pyplot(fig)

        # --- ANÁLISIS AUTOMÁTICO ---
        st.subheader("💡 Interpretación Automática")
        
        # Lógica de Outliers (IQR)
        Q1 = datos.quantile(0.25)
        Q3 = datos.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = datos[(datos < limite_inferior) | (datos > limite_superior)]

        # Lógica de Sesgo (Skewness)
        valor_sesgo = skew(datos)
        
        # Test de Normalidad (Shapiro-Wilk)
        # Solo si n > 3 y n < 5000 (limitación del test)
        stat, p_valor = shapiro(datos)

        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("Sesgo (Skewness)", f"{valor_sesgo:.2f}")
            if abs(valor_sesgo) < 0.5:
                st.write("Simétrica")
            else:
                st.write("Sesgada" + (" a la derecha" if valor_sesgo > 0 else " a la izquierda"))

        with c2:
            st.metric("Outliers detectados", len(outliers))
            st.write(f"Límites: [{limite_inferior:.2f}, {limite_superior:.2f}]")

        with c3:
            st.metric("Normalidad (p-value)", f"{p_valor:.4f}")
            if p_valor > 0.05:
                st.success("Parece Normal")
            else:
                st.warning("No es Normal")
    else:
        st.warning("El archivo no contiene columnas numéricas.")