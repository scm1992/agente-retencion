import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ESTRICTA ---
st.set_page_config(page_title="Retention Pro - V2 Core", layout="wide")

@st.cache_resource
def load_models():
    # 1. Recuperamos la API Key
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # 2. Configuración para langchain-google-genai >= 2.0.0
    # Usamos "gemini-1.5-flash" que es el estándar actual en la API v1.
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0
    )
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

# Inicialización segura
try:
    llm, embeddings = load_models()
except Exception as e:
    st.error(f"Error crítico de inicialización: {e}")
    st.stop()

# --- BASE DE DATOS SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Datos de Juan Pérez
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], 
                 columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Streaming Premium', 15.99, 'Baja']], 
                 columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (Core V2)")
st.info(f"Librería actualizada. Modelo activo: {llm.model}")

queja = st.text_area("Queja del cliente:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    try:
        # A. CLASIFICACIÓN
        with st.spinner("Clasificando..."):
            # Prompt simple y directo
            res_clase = llm.invoke(f"Clasifica el motivo: [Precio, Competencia, Calidad]. Texto: {queja}. Solo la categoría.")
            st.session_state.motivo = res_clase.content.strip()
            st.success(f"**Motivo:** {st.session_state.motivo}")
        
        time.sleep(1) 

        # B. TEXT-TO-SQL
        with st.spinner("Generando SQL..."):
            prompt_sql = """
            Genera SOLO el código SQL para SQLite (sin markdown) para obtener: nombre, nivel, producto, cuota.
            Del cliente id 450.
            Tablas: clientes, suscripciones.
            """
            res_sql = llm.invoke(prompt_sql)
            
            # Limpieza estándar
            query = res_sql.content.strip().replace("```sql", "").replace("```", "").strip()
            st.code(query, language="sql")
            
            # Ejecución
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error en ejecución: {e}")

# --- CHAT ---
st.divider()
if p := st.chat_input("Pregunta sobre el caso..."):
    with st.chat_message("user"): st.write(p)
    try:
        res = llm.invoke(p)
        with st.chat_message("assistant"): st.write(res.content)
    except Exception as e:
        st.error(f"Error chat: {e}")
