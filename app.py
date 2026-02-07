import streamlit as st
import pandas as pd
import sqlite3
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro AI", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    # Bajamos un poco la temperatura para mayor estabilidad
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- FUNCIÓN DE BLINDAJE (NUEVA) ---
def safe_invoke(prompt):
    """Llama a la IA con reintentos y esperas si la cuota se agota"""
    for intento in range(3):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.warning(f"⚠️ Cuota de Google agotada. Reintentando en {6 + intento*2}s...")
                time.sleep(6 + intento*2)
                continue
            raise e
    st.error("❌ No se pudo conectar con Google tras varios intentos. Espera 1 minuto.")
    return None

# --- DB EN MEMORIA ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium', 'Alta']], 
                 columns=['id', 'nombre', 'antiguedad', 'nivel_segmento', 'fidelidad']).to_sql('clientes', conn, index=False)
    pd.DataFrame([
        [450, 'Fibra 1Gbps', 40.0, 'Activo', 'Internet'],
        [450, 'Streaming Premium', 15.99, 'En Proceso Baja', 'Ocio'],
        [450, 'Línea Móvil 50GB', 12.0, 'Activo', 'Movilidad']
    ], columns=['id_cliente', 'producto', 'cuota', 'estado', 'categoria']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

@st.cache_resource
def load_rag():
    reglas = ["Oferta Ocio: 50% dto 3 meses.", "Ventaja: Tenemos fútbol, StreamMax no."]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

def tool_sql(pregunta, id_cliente=450):
    prompt_sql = f"Esquema: clientes, suscripciones. SQL para: '{pregunta}' del id {id_cliente}. Usa UPPER(estado)='ACTIVO'. Solo SQL."
    resp = safe_invoke(prompt_sql)
    if resp:
        query = resp.content.strip().replace("```sql", "").replace("```", "").strip()
        try:
            return pd.read_sql_query(query, conn).to_string(index=False)
        except:
            return "Datos no encontrados."
    return "Error de conexión."

# --- INTERFAZ ---
st.title("🛡️ Panel de Retención Inteligente")
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Entrada del Cliente")
    parrafo = st.text_area("Transcribe la queja aquí:", height=200)
    
    btn_motivo = st.button("🔍 Identificar Motivo")
    btn_resumen = st.button("📊 Generar Resumen Ejecutivo")
    
    if btn_motivo:
        if parrafo:
            with st.spinner("Clasificando..."):
                resp_motivo = safe_invoke(f"Categorías: Precio, Competencia, Calidad. Clasifica esta queja: '{parrafo}'. Responde solo 1 palabra.")
                if resp_motivo:
                    st.info(f"**Motivo:** {resp_motivo.content}")
        else:
            st.warning("Escribe la queja primero.")

with col2:
    st.subheader("2. Resumen y Defensa")
    if "resumen" not in st.session_state: st.session_state.resumen = ""
    if "messages" not in st.session_state: st.session_state.messages = []

    if btn_resumen:
        if not parrafo:
            st.warning("Escribe la queja primero.")
        else:
            with st.spinner("Analizando situación del cliente..."):
                sql_data = tool_sql("Datos del cliente y productos")
                time.sleep(4) # Pausa de seguridad
                regla = vectorstore.similarity_search(parrafo, k=1)[0].page_content
                time.sleep(4)
                resumen_prompt = f"Resume en 5 líneas (Cliente, Baja, Otros, Propuesta) usando: {sql_data} y regla: {regla}."
                resp_res = safe_invoke(resumen_prompt)
                if resp_res:
                    st.session_state.resumen = resp_res.content
    
    if st.session_state.resumen:
        st.markdown(st.session_state.resumen)
        st.divider()
        st.subheader("💬 Chat de Profundización")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Pregunta algo más..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                resp_chat = tool_sql(prompt) 
                st.markdown(resp_chat)
                st.session_state.messages.append({"role": "assistant", "content": resp_chat})
