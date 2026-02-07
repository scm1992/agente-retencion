import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro AI", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    # Gemini 2.0 para máxima capacidad de razonamiento SQL
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- DB SQLITE (Evaluación Text-to-SQL) ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium', 'Alta']], columns=['id', 'nombre', 'antiguedad', 'nivel', 'fidelidad']).to_sql('clientes', conn, index=False)
    pd.DataFrame([
        [450, 'Fibra 1Gbps', 40.0, 'Activo', 'Internet'],
        [450, 'Streaming Premium', 15.99, 'Baja', 'Ocio'],
        [450, 'Línea Móvil 50GB', 12.0, 'Activo', 'Movilidad']
    ], columns=['id_cliente', 'producto', 'cuota', 'estado', 'categoria']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- RAG (Datos Internos y Competencia) ---
@st.cache_resource
def load_rag():
    reglas = [
        "COMPETENCIA: StreamMax (12.99€) no tiene fútbol. Nuestra oferta incluye derechos de Liga.",
        "OFERTA_RETENCION: Si el motivo es Precio en Ocio, ofrecer 50% dto por 3 meses.",
        "REGLA_FIDELIDAD: Clientes Premium con fidelidad Alta pueden recibir un bono de 100GB gratis."
    ]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

# --- ESTADO DE SESIÓN ---
if "datos_contexto" not in st.session_state: st.session_state.datos_contexto = ""
if "analisis_motivo" not in st.session_state: st.session_state.analisis_motivo = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención Inteligente")

# 1. ENTRADA Y ANÁLISIS
st.subheader("📥 1. Análisis de Queja")
parrafo = st.text_area("Explicación del cliente:", height=150, placeholder="Ej: Juan Pérez quiere la baja de Streaming porque StreamMax es más barato...")

if st.button("🔍 Clasificar y Generar Resumen SQL"):
    if not parrafo:
        st.warning("Escribe la queja primero.")
    else:
        # A. Clasificación de Motivo
        with st.spinner("Clasificando motivo..."):
            prompt_clase = f"Clasifica esta queja en una de estas categorías: [Precio, Competencia, Calidad, Personal]. Queja: {parrafo}. Responde solo la palabra."
            try:
                res_clase = llm.invoke(prompt_clase)
                st.session_state.analisis_motivo = res_clase.content
                st.success(f"**Motivo detectado:** {st.session_state.analisis_motivo}")
            except: st.error("Error de cuota en clasificación.")

        # B. Text-to-SQL para Resumen
        with st.spinner("Generando resumen de datos mediante SQL..."):
            prompt_sql = f"""
            Genera un SQL para obtener nombre, nivel, producto, cuota y estado del cliente 450.
            Tablas: clientes (id, nombre, nivel), suscripciones (id_cliente, producto, cuota, estado).
            Solo el código SQL.
            """
            try:
                res_sql = llm.invoke(prompt_sql)
                query = res_sql.content.strip().replace("```sql", "").replace("```", "").strip()
                st.code(query, language="sql")
                df = pd.read_sql_query(query, conn)
                st.session_state.datos_contexto = df.to_string()
                st.dataframe(df)
            except: st.error("Error de cuota en Text-to-SQL.")

# 2. CHAT DE PROFUNDIZACIÓN
st.divider()
st.subheader("💬 2. Chat de Profundización")
st.caption("Consulta sobre competencia, reglas de negocio o detalles del cliente.")

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if p := st.chat_input("Ej: ¿Qué oferta de la competencia le afecta y qué podemos ofrecerle?"):
    st.session_state.chat_history.append({"role": "user", "content": p})
    with st.chat_message("user"): st.markdown(p)
    
    with st.spinner("Consultando RAG y contexto..."):
        # Búsqueda RAG
        docs = vectorstore.similarity_search(p, k=2)
        contexto_rag = "\n".join([d.page_content for d in docs])
        
        # Respuesta integral
        prompt_chat = f"""
        Eres un asistente de retención.
        DATOS CLIENTE (SQL): {st.session_state.datos_contexto}
        MOTIVO DETECTADO: {st.session_state.analisis_motivo}
        DATOS INTERNOS/COMPETENCIA (RAG): {contexto_rag}
        PREGUNTA: {p}
        
        Responde de forma concisa y profesional.
        """
        try:
            res_chat = llm.invoke(prompt_chat)
            st.session_state.chat_history.append({"role": "assistant", "content": res_chat.content})
            with st.chat_message("assistant"): st.markdown(res_chat.content)
        except: st.error("Cuota agotada para el chat.")
