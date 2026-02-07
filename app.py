import streamlit as st
import pandas as pd
import sqlite3
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Resilient AI Agent", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    # Volvemos al 2.0 que sí lo encontraba, pero con paciencia extra
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- BLINDAJE NIVEL LEYENDA (ESPERAS DE 1 MINUTO) ---
def safe_invoke(prompt):
    for intento in range(3):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                espera = 60 # UN MINUTO COMPLETO
                st.warning(f"⚠️ Cuota agotada. Google nos pide calma. Reintentando en {espera}s... (Intento {intento+1}/3)")
                time.sleep(espera)
                continue
            st.error(f"Error: {e}")
            return None
    return None

# --- DB Y RAG (IGUAL QUE ANTES) ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium', 'Alta']], columns=['id', 'nombre', 'antiguedad', 'nivel_segmento', 'fidelidad']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Fibra 1Gbps', 40.0, 'Activo', 'Internet'], [450, 'Streaming Premium', 15.99, 'En Proceso Baja', 'Ocio'], [450, 'Línea Móvil 50GB', 12.0, 'Activo', 'Movilidad']], columns=['id_cliente', 'producto', 'cuota', 'estado', 'categoria']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

@st.cache_resource
def load_rag():
    reglas = ["REGLA_OCIO: 50% dto 3 meses.", "VENTAJA: Tenemos fútbol.", "REGLA_FIDELIDAD: >12 meses = prioridad."]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

def tool_sql(pregunta):
    prompt_sql = f"Escribe SOLO el SQL para: {pregunta} del id 450. Tablas: clientes, suscripciones. No markdown."
    resp = safe_invoke(prompt_sql)
    if resp:
        query = resp.content.strip().replace("```sql", "").replace("```", "").strip()
        return pd.read_sql_query(query, conn).to_string(index=False)
    return "Error en DB"

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (Modo Ultra-Robusto)")
parrafo = st.text_area("Queja del cliente:", height=150)

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 1. Analizar Motivo"):
        with st.spinner("Analizando..."):
            res = safe_invoke(f"Clasifica en una palabra (Precio, Competencia, Calidad): {parrafo}")
            if res: st.success(f"Motivo: {res.content}")

with col2:
    if st.button("🚀 2. Generar Análisis Completo"):
        # Paso 1
        with st.spinner("Consultando SQL..."):
            datos = tool_sql("productos y cuotas")
        
        st.write("✅ Datos obtenidos. Esperando 60s para no saturar la API...")
        time.sleep(60) # ESPERA ENTRE PASOS
        
        # Paso 2
        with st.spinner("Consultando RAG..."):
            regla = vectorstore.similarity_search(parrafo, k=1)[0].page_content
        
        # Paso 3
        with st.spinner("Generando Informe..."):
            informe = safe_invoke(f"Haz un resumen de retención para el cliente id 450 usando estos datos: {datos} y esta regla: {regla}. Basado en esta queja: {parrafo}")
            if informe: 
                st.session_state.resumen_final = informe.content
                st.info(st.session_state.resumen_final)

# --- CHAT (BLINDADO TAMBIÉN) ---
st.divider()
st.subheader("💬 Chat de Profundización")
if "chat_history" not in st.session_state: st.session_state.chat_history = []

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]): st.write(m["content"])

if p := st.chat_input("Pregunta algo sobre el cliente..."):
    st.session_state.chat_history.append({"role": "user", "content": p})
    with st.chat_message("user"): st.write(p)
    
    with st.spinner("La IA está pensando (respetando cuotas)..."):
        # El chat también usa safe_invoke
        respuesta = safe_invoke(f"Responde esta duda sobre el cliente 450: {p}. Datos: {st.session_state.get('resumen_final', 'No hay datos aún')}")
        if respuesta:
            st.session_state.chat_history.append({"role": "assistant", "content": respuesta.content})
            with st.chat_message("assistant"): st.write(respuesta.content)
