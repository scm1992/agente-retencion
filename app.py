import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - Flash Only", layout="wide")

# 1. Configuración del Modelo
# FORZAMOS "gemini-1.5-flash". 
# Es el único que tiene una cuota gratuita generosa (15 peticiones/minuto) y no da error de límite 0.
model_name = "gemini-1.5-flash"

@st.cache_resource
def load_resources():
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        max_retries=2
    )
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

try:
    llm, embeddings = load_resources()
except Exception as e:
    st.error(f"Error al cargar recursos: {e}")
    st.stop()

# --- DB SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Datos de prueba
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Streaming Premium', 15.99, 'Baja']], columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (Modo Flash Gratuito)")
st.success(f"✅ Conectado al modelo económico: **{model_name}**")

queja = st.text_area("Queja del cliente:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    try:
        # A. CLASIFICACIÓN
        with st.spinner("Clasificando..."):
            res_clase = llm.invoke(f"Clasifica el motivo: [Precio, Competencia, Calidad]. Texto: {queja}. Solo la categoría.")
            st.session_state.motivo = res_clase.content.strip()
            st.info(f"**Motivo:** {st.session_state.motivo}")
        
        # PAUSA OBLIGATORIA (Rate Limiting)
        # Esperamos 4 segundos para asegurarnos de no saturar la cuota gratuita
        time.sleep(4) 

        # B. Text-to-SQL
        with st.spinner("Generando SQL..."):
            prompt_sql = """
            Genera SOLO el código SQL para SQLite (sin markdown) para obtener: nombre, nivel, producto, cuota.
            Del cliente id 450.
            Tablas: clientes, suscripciones.
            """
            res_sql = llm.invoke(prompt_sql)
            
            # Limpieza
            query = res_sql.content.strip().replace("```sql", "").replace("```", "").strip()
            st.code(query, language="sql")
            
            # Ejecución
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        # Si sale error 429, le decimos al usuario que espere
        if "429" in str(e):
            st.warning("⚠️ Tráfico alto. Espera 30 segundos antes de volver a intentar.")
        else:
            st.error(f"Error: {e}")
