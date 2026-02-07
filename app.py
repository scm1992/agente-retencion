import streamlit as st
import pandas as pd
import sqlite3
import time
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - Auto Discovery", layout="wide")

# 1. AUTODESCUBRIMIENTO DE MODELO (El paso mágico)
# Esto soluciona el error 404 encontrando qué nombre "ve" realmente tu servidor
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

@st.cache_resource
def get_valid_model_name():
    try:
        # Listamos todos los modelos disponibles para tu API Key
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Prioridad 1: Alguna versión de Flash 1.5
        for m in available_models:
            if "gemini-1.5-flash" in m:
                return m # Devolvemos el nombre exacto que el servidor reconoce (ej: models/gemini-1.5-flash-001)
        
        # Prioridad 2: Alguna versión Pro 1.5
        for m in available_models:
            if "gemini-1.5-pro" in m:
                return m
                
        # Prioridad 3: El viejo confiable Gemini Pro
        for m in available_models:
            if "gemini-pro" in m:
                return m
                
        return "gemini-1.5-flash" # Fallback por defecto
    except Exception as e:
        st.error(f"Error al listar modelos: {e}")
        return "gemini-1.5-flash"

# Obtenemos el nombre real
model_name_real = get_valid_model_name()

# 2. CARGA DE LANGCHAIN CON EL NOMBRE VALIDADO
@st.cache_resource
def load_langchain_resources():
    # Usamos el nombre que ACABAMOS de descubrir que sí existe
    llm = ChatGoogleGenerativeAI(
        model=model_name_real,
        google_api_key=api_key,
        temperature=0
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

# Inicialización
try:
    llm, embeddings = load_langchain_resources()
except Exception as e:
    st.error(f"Error cargando LangChain: {e}")
    st.stop()

# --- DB SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Streaming Premium', 15.99, 'Baja']], columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (Auto-Detect)")
# Mostramos al usuario qué modelo se ha seleccionado automáticamente
st.success(f"✅ Conexión establecida usando el modelo: **{model_name_real}**")

queja = st.text_area("Queja del cliente:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    try:
        # A. Clasificación
        with st.spinner(f"Clasificando con {model_name_real}..."):
            res_clase = llm.invoke(f"Clasifica el motivo: [Precio, Competencia, Calidad]. Texto: {queja}. Solo la categoría.")
            st.session_state.motivo = res_clase.content.strip()
            st.info(f"**Motivo:** {st.session_state.motivo}")
        
        time.sleep(1)

        # B. Text-to-SQL
        with st.spinner("Generando SQL..."):
            prompt_sql = """
            Genera SOLO el código SQL para SQLite (sin markdown) para obtener: nombre, nivel, producto, cuota.
            Del cliente id 450.
            Tablas: clientes, suscripciones.
            """
            res_sql = llm.invoke(prompt_sql)
            query = res_sql.content.strip().replace("```sql", "").replace("```", "").strip()
            st.code(query, language="sql")
            
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error en ejecución: {e}")
