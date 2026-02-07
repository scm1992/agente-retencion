import streamlit as st
import pandas as pd
import sqlite3
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Retention Pro AI", layout="wide")

# --- INICIALIZACIÓN DE MODELOS (Cacheado para no gastar cuota extra) ---
@st.cache_resource
def load_models():
    # Asegúrate de tener la variable de entorno o búscala desde el secret de Streamlit
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- BASE DE DATOS Y RAG ---
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

# --- FUNCIONES DE LÓGICA ---
def tool_sql(pregunta, id_cliente=450):
    prompt = f"Esquema: clientes, suscripciones. SQL para: '{pregunta}' del id {id_cliente}. Usa UPPER(estado)='ACTIVO'. Solo SQL."
    query = llm.invoke(prompt).content.strip().replace("```sql", "").replace("```", "").strip()
    return pd.read_sql_query(query, conn).to_string(index=False)

# --- INTERFAZ STREAMLIT ---
st.title("🛡️ Panel de Retención Inteligente")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Entrada del Cliente")
    parrafo = st.text_area("Transcribe la queja aquí:", height=200, placeholder="Ej: Juan quiere irse a StreamMax porque es más barato...")
    
    btn_motivo, btn_resumen = st.columns(2)
    
    if btn_motivo.button("🔍 Identificar Motivo"):
        if parrafo:
            motivo = llm.invoke(f"Categorías: Precio, Competencia, Calidad. Clasifica: '{parrafo}'. 1 palabra.").content
            st.info(f"**Motivo Detectado:** {motivo}")
        else:
            st.warning("Escribe algo primero.")

with col2:
    st.subheader("2. Resumen y Defensa")
    
    if "resumen" not in st.session_state:
        st.session_state.resumen = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if btn_resumen.button("📊 Generar Resumen Ejecutivo"):
        with st.spinner("Analizando base de datos..."):
            sql_data = tool_sql("Datos de id 450 y productos")
            regla = vectorstore.similarity_search(parrafo, k=1)[0].page_content
            resumen_prompt = f"Resume en 5 líneas (Cliente, Baja, Otros, Propuesta) usando estos datos: {sql_data} y esta regla: {regla}."
            st.session_state.resumen = llm.invoke(resumen_prompt).content
    
    if st.session_state.resumen:
        st.markdown(st.session_state.resumen)
        st.divider()
        
        # --- PUNTO 3: CHAT HISTÓRICO ---
        st.subheader("💬 Chat de Profundización")
        
        # Mostrar historial (para el scroll)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("¿Qué más quieres saber?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Decisión SQL vs RAG
                decision = llm.invoke(f"¿'{prompt}' requiere datos DB o RAG? Responde 'DB' o 'RAG'.").content.strip()
                if "DB" in decision:
                    resp = tool_sql(prompt)
                else:
                    resp = vectorstore.similarity_search(prompt, k=1)[0].page_content
                
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})