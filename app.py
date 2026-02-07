import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="SQL & RAG Agent", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    # Usamos flash-lite o 1.5 directamente para intentar bypass de cuota
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- DB SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Fibra 1Gbps', 40.0, 'Activo'], [450, 'Streaming Premium', 15.99, 'Baja'], [450, 'Móvil 50GB', 12.0, 'Activo']], 
                 columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- RAG ---
@st.cache_resource
def load_rag():
    reglas = ["OFERTA: 50% dto 3 meses.", "VENTAJA: Fútbol incluido.", "BONO: 100GB gratis."]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

# --- VARIABLES DE SESIÓN (Para evitar timeouts) ---
if "datos_sql" not in st.session_state: st.session_state.datos_sql = None
if "resumen" not in st.session_state: st.session_state.resumen = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- INTERFAZ ---
st.title("🛡️ Agente de Retención (Arquitectura Real)")
parrafo = st.text_area("Queja del cliente:", height=100)

col1, col2 = st.columns(2)

with col1:
    if st.button("Step 1: Generar SQL y Consultar"):
        with st.spinner("IA generando SQL..."):
            # Prompt ultra-corto para no gastar tokens/cuota
            prompt_sql = "Table: suscripciones (id_cliente, producto, cuota, estado). Write SQL for client 450. SQL ONLY."
            try:
                # Solo un reintento corto para evitar timeout de la app
                res = llm.invoke(prompt_sql)
                query = res.content.strip().replace("```sql", "").replace("```", "").strip()
                st.code(query, language="sql")
                st.session_state.datos_sql = pd.read_sql_query(query, conn)
                st.success("¡Datos recuperados!")
            except Exception as e:
                st.error("Error de cuota. Espera 30 segundos manualmente y vuelve a pulsar.")

    if st.session_state.datos_sql is not None:
        st.dataframe(st.session_state.datos_sql)

with col2:
    if st.button("Step 2: Generar Informe Final"):
        if st.session_state.datos_sql is None:
            st.warning("Primero ejecuta el Step 1.")
        else:
            with st.spinner("Buscando en RAG y redactando..."):
                # RAG es local (FAISS)
                docs = vectorstore.similarity_search(parrafo, k=1)
                regla = docs[0].page_content
                
                # Llamada final
                prompt_f = f"Resume: {parrafo}. Data: {st.session_state.datos_sql.to_string()}. Rule: {regla}"
                try:
                    res_f = llm.invoke(prompt_f)
                    st.session_state.resumen = res_f.content
                except:
                    st.error("Error de cuota. Espera 30 segundos y reintenta este paso.")

if st.session_state.resumen:
    st.info(st.session_state.resumen)

# --- CHAT ---
st.divider()
st.subheader("💬 Chat")
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]): st.write(m["content"])

if p := st.chat_input("Duda sobre el cliente..."):
    st.session_state.chat_history.append({"role": "user", "content": p})
    # Solo intentamos el chat si hay contexto
    ctx = st.session_state.resumen if st.session_state.resumen else "No hay resumen."
    try:
        r = llm.invoke(f"Contexto: {ctx}. Pregunta: {p}")
        st.session_state.chat_history.append({"role": "assistant", "content": r.content})
        st.rerun()
    except:
        st.error("Cuota agotada para el chat.")
