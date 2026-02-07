import streamlit as st
import pandas as pd
import sqlite3
import time
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - Conexión Directa", layout="wide")

# Configuración directa de Google AI (Saltamos LangChain para evitar errores 404/429)
api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_resource
def load_resources():
    # Usamos el modelo más estable disponible
    model = genai.GenerativeModel('gemini-1.5-flash')
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

# --- RAG ---
@st.cache_resource
def load_rag():
    reglas = ["COMPETENCIA: StreamMax (12.99€) no tiene fútbol.", 
              "OFERTA: 50% dto por 3 meses.", 
              "REGLA: Bono 100GB gratis para Premium."]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

# --- ESTADO DE SESIÓN ---
if "datos_contexto" not in st.session_state: st.session_state.datos_contexto = ""
if "analisis_motivo" not in st.session_state: st.session_state.analisis_motivo = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (Conexión Directa)")

parrafo = st.text_area("Explicación del cliente:", height=100)

if st.button("🔍 Clasificar y Generar SQL"):
    if not parrafo:
        st.warning("Escribe la queja.")
    else:
        try:
            # A. Clasificación
            with st.spinner("Clasificando..."):
                response = model.generate_content(f"Clasifica: [Precio, Competencia, Calidad]. Queja: {parrafo}. Solo una palabra.")
                st.session_state.analisis_motivo = response.text.strip()
                st.success(f"Motivo: {st.session_state.analisis_motivo}")
            
            time.sleep(1) # Respiro

            # B. Text-to-SQL
            with st.spinner("Generando SQL..."):
                prompt_sql = f"""Genera SQL para obtener nombre, nivel, producto, cuota y estado del cliente 450.
                Tablas: clientes (id, nombre, nivel), suscripciones (id_cliente, producto, cuota, estado). 
                Escribe SOLO el código SQL plano."""
                response_sql = model.generate_content(prompt_sql)
                query = response_sql.text.strip().replace("```sql", "").replace("```", "").strip()
                st.code(query, language="sql")
                
                df = pd.read_sql_query(query, conn)
                st.session_state.datos_contexto = df.to_string()
                st.dataframe(df)
        except Exception as e:
            st.error(f"Error detectado: {e}")

# --- CHAT ---
st.divider()
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if p := st.chat_input("Pregunta al asistente..."):
    st.session_state.chat_history.append({"role": "user", "content": p})
    with st.chat_message("user"): st.markdown(p)
    
    docs = vectorstore.similarity_search(p, k=1)
    prompt_chat = f"Datos: {st.session_state.datos_contexto}. Motivo: {st.session_state.analisis_motivo}. Regla: {docs[0].page_content}. Pregunta: {p}"
    
    try:
        res_chat = model.generate_content(prompt_chat)
        st.session_state.chat_history.append({"role": "assistant", "content": res_chat.text})
        st.rerun()
    except Exception as e:
        st.error(f"Error en chat: {e}")
