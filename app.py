import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN DE MODELOS CON FALLBACK ---
st.set_page_config(page_title="Retention Pro - High Availability", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    # Intentamos cargar el 2.0, pero si la cuota es 0, usamos el 1.5 como respaldo
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
        # Pequeño test de conectividad
        llm.invoke("Ping")
    except Exception:
        # Si el 2.0 falla por cuota (RESOURCE_EXHAUSTED), usamos el 1.5 Flash
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- DB SQLITE (Estructura para Text-to-SQL) ---
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

# --- RAG (Manual de Retención y Competencia) ---
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
st.info(f"Modelo activo: {llm.model}")

# 1. ENTRADA Y ANÁLISIS
st.subheader("📥 1. Análisis de Queja")
parrafo = st.text_area("Transcripción de la llamada / Queja:", height=150, 
                       placeholder="Introduzca el texto para clasificar y extraer datos...")

if st.button("🔍 Clasificar y Generar Resumen SQL"):
    if not parrafo:
        st.warning("Por favor, introduzca una queja.")
    else:
        # A. Clasificación de Motivo
        with st.spinner("Clasificando motivo de baja..."):
            prompt_clase = f"Clasifica esta queja: [Precio, Competencia, Calidad, Personal]. Texto: {parrafo}. Responde solo con una de las palabras."
            try:
                res_clase = llm.invoke(prompt_clase)
                st.session_state.analisis_motivo = res_clase.content.strip()
                st.success(f"**Motivo detectado:** {st.session_state.analisis_motivo}")
            except Exception as e: 
                st.error(f"Error en clasificación: {e}")

        # Pausa de seguridad para evitar ráfagas (Rate Limiting)
        time.sleep(2) 

        # B. Text-to-SQL
        with st.spinner("Generando consulta SQL técnica..."):
            prompt_sql = f"""
            Genera un SQL para obtener nombre, nivel, producto, cuota y estado del cliente 450.
            Tablas: clientes (id, nombre, nivel), suscripciones (id_cliente, producto, cuota, estado).
            Responde SOLO el código SQL plano, sin bloques de código ni explicaciones.
            """
            try:
                res_sql = llm.invoke(prompt_sql)
                # Limpieza por si la IA añade markdown
                query = res_sql.content.strip().replace("```sql", "").replace("```", "").strip()
                st.code(query, language="sql")
                
                # Ejecución y guardado de contexto
                df = pd.read_sql_query(query, conn)
                st.session_state.datos_contexto = df.to_string()
                st.dataframe(df)
            except Exception as e: 
                st.error(f"Error en Text-to-SQL: {e}")

# 2. CHAT DE PROFUNDIZACIÓN
st.divider()
st.subheader("💬 2. Chat de Profundización (RAG)")

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if p := st.chat_input("Pregunte sobre ofertas de la competencia o datos del cliente..."):
    st.session_state.chat_history.append({"role": "user", "content": p})
    with st.chat_message("user"): st.markdown(p)
    
    with st.spinner("Buscando en manuales y datos..."):
        # Búsqueda semántica en RAG
        docs = vectorstore.similarity_search(p, k=2)
        contexto_rag = "\n".join([d.page_content for d in docs])
        
        # Generación de respuesta contextual
        prompt_chat = f"""
        Eres un experto en retención de clientes.
        CONTEXTO SQL: {st.session_state.datos_contexto}
        MOTIVO: {st.session_state.analisis_motivo}
        DATOS RAG: {contexto_rag}
        PREGUNTA: {p}
        
        Responde de forma estratégica y breve.
        """
        try:
            res_chat = llm.invoke(prompt_chat)
            st.session_state.chat_history.append({"role": "assistant", "content": res_chat.content})
            with st.chat_message("assistant"): st.markdown(res_chat.content)
            st.rerun()
        except Exception as e: 
            st.error(f"Error en chat: {e}")
