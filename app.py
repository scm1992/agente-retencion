import streamlit as st
import pandas as pd
import sqlite3
import time
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - New Key Test", layout="wide")

# 1. Configurar la nueva API Key
api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_resource
def load_resources():
    # Intentamos con el modelo más compatible
    try:
        model = genai.GenerativeModel('gemini-pro')
        # Test rápido para validar la nueva Key
        model.generate_content("test")
    except Exception as e:
        st.error(f"La nueva API Key falló. Error: {e}")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return model, embeddings

model, embeddings = load_resources()

# --- DB SQLITE (Contexto Juan Pérez) ---
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

# --- RAG ---
@st.cache_resource
def load_rag():
    reglas = ["OFERTA: 50% dto 3 meses si es por precio.", "COMPETENCIA: StreamMax no tiene fútbol."]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

# --- ESTADO ---
if "datos_sql" not in st.session_state: st.session_state.datos_sql = ""
if "motivo" not in st.session_state: st.session_state.motivo = ""

# --- INTERFAZ ---
st.title("🛡️ Validación de Retención (Nueva Key)")
st.success("✅ Conexión con Gemini establecida correctamente.")

queja = st.text_area("Introduce la queja de Juan Pérez:", height=100)

if st.button("🚀 Ejecutar Análisis"):
    try:
        # A. CLASIFICACIÓN
        with st.spinner("Clasificando..."):
            res = model.generate_content(f"Clasifica: [Precio, Competencia]. Queja: {queja}. Solo una palabra.")
            st.session_state.motivo = res.text.strip()
            st.write(f"**Motivo:** {st.session_state.motivo}")
        
        # B. SQL
        with st.spinner("Generando SQL..."):
            prompt_sql = "Genera SQL para obtener nombre, nivel, producto, cuota y estado del cliente 450. Tablas: clientes, suscripciones. SQL solo."
            res_sql = model.generate_content(prompt_sql)
            query = res_sql.text.strip().replace("```sql", "").replace("```", "").strip()
            st.code(query, language="sql")
            
            df = pd.read_sql_query(query, conn)
            st.session_state.datos_sql = df.to_string()
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error durante el análisis: {e}")

# --- CHAT SIMPLE ---
st.divider()
pregunta = st.chat_input("Hazle una pregunta al asistente sobre el caso...")
if pregunta:
    with st.chat_message("user"): st.write(pregunta)
    res_chat = model.generate_content(f"Contexto: {st.session_state.datos_sql}. Pregunta: {pregunta}")
    with st.chat_message("assistant"): st.write(res_chat.text)

