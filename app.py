import streamlit as st
import pandas as pd
import sqlite3
import time
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Retention Pro - Final Fix", layout="wide")

# Configuración de API Key (Asegúrate de tenerla en Secrets de Streamlit)
api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_resource
def load_resources():
    # Estrategia de búsqueda de modelo para evitar el error 404
    llm_model = None
    posibles_nombres = [
        'models/gemini-1.5-flash', 
        'gemini-1.5-flash', 
        'models/gemini-pro'
    ]
    
    for nombre in posibles_nombres:
        try:
            m = genai.GenerativeModel(nombre)
            # Test de conectividad rápido
            m.generate_content("test")
            llm_model = m
            break
        except Exception:
            continue
            
    if llm_model is None:
        st.error("Error crítico: No se encuentra ningún modelo disponible. Revisa tu cuota en Google AI Studio.")
        st.stop()
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm_model, embeddings

model, embeddings = load_resources()

# --- BASE DE DATOS LOCAL (Para Text-to-SQL) ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Tabla Clientes
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium', 'Alta']], 
                 columns=['id', 'nombre', 'antiguedad', 'nivel', 'fidelidad']).to_sql('clientes', conn, index=False)
    # Tabla Suscripciones
    pd.DataFrame([
        [450, 'Fibra 1Gbps', 40.0, 'Activo', 'Internet'],
        [450, 'Streaming Premium', 15.99, 'Baja', 'Ocio'],
        [450, 'Línea Móvil 50GB', 12.0, 'Activo', 'Movilidad']
    ], columns=['id_cliente', 'producto', 'cuota', 'estado', 'categoria']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- RAG (Manual de Retención) ---
@st.cache_resource
def load_rag():
    reglas = [
        "COMPETENCIA: StreamMax cuesta 12.99€. No tienen fútbol. Nuestra fibra es más estable.",
        "OFERTA: Si el cliente se queja de precio en Ocio, ofrecer 50% de descuento durante 3 meses.",
        "REGLA_ORO: Clientes Premium con fidelidad Alta tienen derecho a un Bono de 100GB gratis."
    ]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

# --- ESTADO DE SESIÓN ---
if "datos_contexto" not in st.session_state: st.session_state.datos_contexto = ""
if "analisis_motivo" not in st.session_state: st.session_state.analisis_motivo = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- INTERFAZ DE USUARIO ---
st.title("🛡️ Sistema de Retención Inteligente")
st.info(f"Conectado a: {model.model_name}")

# PASO 1: ANÁLISIS
st.subheader("📥 1. Análisis de Queja")
parrafo = st.text_area("Transcripción de la llamada:", height=120, 
                       placeholder="Ej: Juan Pérez está molesto por el precio de Streaming Premium...")

if st.button("🚀 Clasificar y Extraer Datos (SQL)"):
    if not parrafo:
        st.warning("Escribe la queja del cliente.")
    else:
        try:
            # A. Clasificación de Motivo
            with st.spinner("Clasificando motivo..."):
                resp = model.generate_content(f"Clasifica esta queja: [Precio, Competencia, Calidad]. Queja: {parrafo}. Solo dime la categoría.")
                st.session_state.analisis_motivo = resp.text.strip()
                st.success(f"**Motivo detectado:** {st.session_state.analisis_motivo}")
            
            time.sleep(1) # Pequeña pausa para evitar ráfagas

            # B. Text-to-SQL (Evaluación Clave)
            with st.spinner("Generando SQL de extracción..."):
                prompt_sql = f"""
                Genera una consulta SQL para SQLite que extraiga nombre, nivel, producto, cuota y estado del cliente 450.
                Tablas: 
                - clientes (id, nombre, nivel)
                - suscripciones (id_cliente, producto, cuota, estado)
                Devuelve SOLO el código SQL sin bloques de markdown.
                """
                resp_sql = model.generate_content(prompt_sql)
                query = resp_sql.text.strip().replace("```sql", "").replace("```", "").strip()
                
                # Mostrar el SQL generado para evaluación
                st.code(query, language="sql")
                
                # Ejecución en la DB local
                df = pd.read_sql_query(query, conn)
                st.session_state.datos_contexto = df.to_string()
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Error en el proceso: {e}")

# PASO 2: CHAT CONTEXTUAL
st.divider()
st.subheader("💬 2. Chat de Profundización")

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if p := st.chat_input("Haz una pregunta sobre la competencia o el perfil del cliente..."):
    st.session_state.chat_history.append({"role": "user", "content": p})
    with st.chat_message("user"): st.markdown(p)
    
    with st.spinner("Consultando RAG y base de datos..."):
        # Recuperar información del manual (RAG)
        docs = vectorstore.similarity_search(p, k=1)
        info_rag = docs[0].page_content
        
        # Generar respuesta final
        prompt_chat = f"""
        Actúa como experto en retención.
        DATOS DEL CLIENTE: {st.session_state.datos_contexto}
        MOTIVO DETECTADO: {st.session_state.analisis_motivo}
        REGLA DE NEGOCIO: {info_rag}
        PREGUNTA: {p}
        
        Da una respuesta estratégica y breve.
        """
        try:
            res_chat = model.generate_content(prompt_chat)
            st.session_state.chat_history.append({"role": "assistant", "content": res_chat.text})
            with st.chat_message("assistant"): st.markdown(res_chat.text)
            st.rerun()
        except Exception as e:
            st.error(f"Error en el chat: {e}")
