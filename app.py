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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

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

# --- LÓGICA CON REINTENTOS (ANTI-QUOTA) ---
def tool_sql(pregunta, id_cliente=450):
    prompt = f"Esquema: clientes, suscripciones. SQL para: '{pregunta}' del id {id_cliente}. Usa UPPER(estado)='ACTIVO'. Solo SQL."
    for intento in range(3):
        try:
            query_resp = llm.invoke(prompt)
            query = query_resp.content.strip().replace("```sql", "").replace("```", "").strip()
            return pd.read_sql_query(query, conn).to_string(index=False)
        except Exception as e:
            if "429" in str(e):
                time.sleep(6) # Espera mayor para la cuota free
                continue
            return f"Error: {e}"
    return "Límite de API agotado."

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
            motivo = llm.invoke(f"Categorías: Precio, Competencia, Calidad. Clasifica: '{parrafo}'. 1 palabra.").content
            st.info(f"**Motivo:** {motivo}")

with col2:
    st.subheader("2. Resumen y Defensa")
    if "resumen" not in st.session_state: st.session_state.resumen = ""
    if "messages" not in st.session_state: st.session_state.messages = []

    if btn_resumen:
        if not parrafo:
            st.warning("Escribe la queja primero.")
        else:
            with st.spinner("Consultando datos y reglas..."):
                sql_data = tool_sql("Datos de id 450 y productos")
                time.sleep(3) # Pausa entre llamadas
                regla = vectorstore.similarity_search(parrafo, k=1)[0].page_content
                time.sleep(3) 
                resumen_prompt = f"Resume en 5 líneas (Cliente, Baja, Otros, Propuesta) usando: {sql_data} y regla: {regla}."
                st.session_state.resumen = llm.invoke(resumen_prompt).content
    
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
                # Simplificamos a una sola llamada para ahorrar cuota
                resp = tool_sql(prompt) 
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
