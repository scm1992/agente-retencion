import streamlit as st
import pandas as pd
import sqlite3
import time
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - Legacy Fix", layout="wide")

api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_resource
def load_resources():
    # 'gemini-pro' es el nombre más compatible en versiones antiguas de la librería
    try:
        model = genai.GenerativeModel('gemini-pro')
        model.generate_content("test")
    except Exception as e:
        # Si falla, intentamos la versión que a veces pide la v1beta
        try:
            model = genai.GenerativeModel('chat-bison-001') # Modelo legacy de respaldo
            model.generate_content("test")
        except:
            st.error(f"Error de conexión persistente: {e}")
            st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return model, embeddings

model, embeddings = load_resources()

# --- DB SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium', 'Alta']], 
                 columns=['id', 'nombre', 'antiguedad', 'nivel', 'fidelidad']).to_sql('clientes', conn, index=False)
    pd.DataFrame([
        [450, 'Fibra 1Gbps', 40.0, 'Activo', 'Internet'],
        [450, 'Streaming Premium', 15.99, 'Baja', 'Ocio'],
        [450, 'Línea Móvil 50GB', 12.0, 'Activo', 'Movilidad']
    ], columns=['id_cliente', 'producto', 'cuota', 'estado', 'categoria']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (Modo Pro)")
st.success(f"✅ Conectado exitosamente al modelo: {model.model_name}")

queja = st.text_area("Introduce la queja de Juan Pérez:", height=100)

if st.button("🚀 Ejecutar Análisis"):
    try:
        # A. CLASIFICACIÓN
        with st.spinner("Clasificando..."):
            res = model.generate_content(f"Clasifica: [Precio, Competencia]. Queja: {queja}. Solo una palabra.")
            st.write(f"**Motivo Detectado:** {res.text.strip()}")
        
        # B. SQL
        with st.spinner("Generando SQL..."):
            prompt_sql = """Genera un SQL para SQLite que devuelva nombre, nivel, producto y cuota del cliente 450. 
            Tablas: clientes (id, nombre, nivel), suscripciones (id_cliente, producto, cuota). 
            Responde SOLO el SQL."""
            res_sql = model.generate_content(prompt_sql)
            query = res_sql.text.strip().replace("```sql", "").replace("```", "").strip()
            st.code(query, language="sql")
            
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error en el análisis: {e}")
