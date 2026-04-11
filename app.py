import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, skew
from scipy.stats import norm

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
        
# --- MÓDULO DE PRUEBA DE HIPÓTESIS Z ---
if df is not None:
    st.divider()
    st.header("🧪 Prueba de Hipótesis Z (Media)")
    
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(cols_numericas) > 0:
        with st.expander("Configuración de la Prueba Z", expanded=True):
            col_z = st.selectbox("Variable para la prueba:", cols_numericas, key="z_var")
            
            # Limpieza de datos inmediata
            datos_z = df[col_z].dropna()
            n = len(datos_z)
            media_muestral = datos_z.mean()

            c1, c2, c3 = st.columns(3)
            with c1:
                mu_0 = st.number_input("Media Hipotética (H0)", value=0.0)
                sigma = st.number_input("Desviación Estándar Poblacional (σ)", value=1.0, min_value=0.01)
            with c2:
                tipo_prueba = st.selectbox("Tipo de prueba:", ["Bilateral (≠)", "Cola Izquierda (<)", "Cola Derecha (>)"])
                alfa = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05)
            with c3:
                st.metric("Media Muestral (x̄)", f"{media_muestral:.4f}")
                st.metric("Tamaño de muestra (n)", n)

        # VALIDACIONES
        if n < 30:
            st.warning("⚠️ El tamaño de la muestra es menor a 30. Los resultados de la prueba Z podrían no ser confiables (considere usar Prueba T).")
        
        # CÁLCULOS ESTADÍSTICOS
        # Estadístico Z = (x̄ - μ0) / (σ / √n)
        z_stat = (media_muestral - mu_0) / (sigma / np.sqrt(n))
        
        if tipo_prueba == "Bilateral (≠)":
            p_valor = 2 * (1 - norm.cdf(abs(z_stat)))
            z_critico_inf = norm.ppf(alfa/2)
            z_critico_sup = norm.ppf(1 - alfa/2)
        elif tipo_prueba == "Cola Izquierda (<)":
            p_valor = norm.cdf(z_stat)
            z_critico_inf = norm.ppf(alfa)
            z_critico_sup = None
        else: # Cola Derecha
            p_valor = 1 - norm.cdf(z_stat)
            z_critico_inf = None
            z_critico_sup = norm.ppf(1 - alfa)

        # DECISIÓN
        rechazar = p_valor < alfa

        # MOSTRAR RESULTADOS
        res1, res2 = st.columns(2)
        with res1:
            st.subheader("Resultados")
            st.write(f"**Estadístico Z:** {z_stat:.4f}")
            st.write(f"**P-Value:** {p_valor:.4f}")
            if rechazar:
                st.error("Decisión: Rechazar H0")
            else:
                st.success("Decisión: No rechazar H0")

        with res2:
            # Gráfico de la Normal y Región Crítica
            x = np.linspace(-4, 4, 1000)
            y = norm.pdf(x, 0, 1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y, color='black')
            
            # Sombreado de zonas de rechazo
            if tipo_prueba == "Bilateral (≠)":
                ax.fill_between(x, y, where=(x <= z_critico_inf) | (x >= z_critico_sup), color='red', alpha=0.5, label="Región de Rechazo")
            elif tipo_prueba == "Cola Izquierda (<)":
                ax.fill_between(x, y, where=(x <= z_critico_inf), color='red', alpha=0.5, label="Región de Rechazo")
            else:
                ax.fill_between(x, y, where=(x >= z_critico_sup), color='red', alpha=0.5, label="Región de Rechazo")
            
            # Línea del estadístico calculado
            ax.axvline(z_stat, color='blue', linestyle='--', label=f'Z calculado: {z_stat:.2f}')
            ax.legend(fontsize='small')
            ax.set_title("Distribución Normal Estándar")
            st.pyplot(fig)