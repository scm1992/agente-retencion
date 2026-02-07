import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN DE MODELOS ---
st.set_page_config(page_title="Evaluación Text-to-SQL", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    # Usamos 1.5-flash por su estabilidad en Text-to-SQL
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- DB SQLITE REAL ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([
        [450, 'Fibra 1Gbps', 40.0, 'Activo'],
        [450, 'Streaming Premium', 15.99, 'Baja'],
        [450, 'Móvil 50GB', 12.0, 'Activo']
    ], columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- RAG ---
@st.cache_resource
def load_rag():
    reglas = ["OFERTA: 50% dto 3 meses.", "VENTAJA: Fútbol incluido.", "BONO: 100GB gratis."]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

# --- ESTADO DE SESIÓN ---
if "datos_sql" not in st.session_state: st.session_state.datos_sql = None
if "resumen" not in st.session_state: st.session_state.resumen = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- INTERFAZ ---
st.title("🛡️ Laboratorio de Evaluación: Text-to-SQL + RAG")
st.markdown("---")

pregunta_natural = st.text_input("Consulta en lenguaje natural:", "Dime qué productos tiene contratados el cliente 450 y cuánto paga")
queja_cliente = st.text_area("Contexto de la queja (RAG):", "Juan Pérez quiere darse de baja de la televisión porque es cara.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fase 1: Text-to-SQL")
    if st.button("🔍 Ejecutar Evaluación SQL"):
        with st.spinner("IA traduciendo a SQL..."):
            # PROMPT DE EVALUACIÓN TÉCNICA
            prompt_sql = f"""
            Task: Convert natural language to SQLite.
            Tables: 
            - clientes (id, nombre, antiguedad, nivel)
            - suscripciones (id_cliente, producto, cuota, estado)
            Question: {pregunta_natural}
            Output: SQL ONLY.
            """
            try:
                res = llm.invoke(prompt_sql)
                query = res.content.strip().replace("```sql", "").replace("```", "").strip()
                st.code(query, language="sql") # Aquí evalúas la sintaxis
                
                # Ejecución real en DB
                st.session_state.datos_sql = pd.read_sql_query(query, conn)
                st.success("Consulta ejecutada con éxito.")
            except Exception as e:
                st.error(f"Fallo de cuota o sintaxis: {e}")

    if st.session_state.datos_sql is not None:
        st.dataframe(st.session_state.datos_sql)

with col2:
    st.subheader("Fase 2: RAG + Informe")
    if st.button("📄 Generar Propuesta Final"):
        if st.session_state.datos_sql is None:
            st.warning("Falta el contexto del Step 1.")
        else:
            with st.spinner("Consultando reglas y redactando..."):
                # RAG LOCAL
                docs = vectorstore.similarity_search(queja_cliente, k=1)
                regla = docs[0].page_content
                
                # INFORME
                prompt_f = f"Basado en estos datos: {st.session_state.datos_sql.to_string()} y esta regla: {regla}. Resume la propuesta para: {queja_cliente}"
                try:
                    res_f = llm.invoke(prompt_f)
                    st.session_state.resumen = res_f.content
                    st.info(st.session_state.resumen)
                except Exception as e:
                    st.error("Error de cuota en el informe final.")

# --- CHAT ---
st.divider()
if st.session_state.resumen:
    st.subheader("💬 Chat de Profundización")
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]): st.write(m["content"])

    if p := st.chat_input("Pregunta sobre la estrategia..."):
        st.session_state.chat_history.append({"role": "user", "content": p})
        try:
            r = llm.invoke(f"Contexto: {st.session_state.resumen}. Pregunta: {p}")
            st.session_state.chat_history.append({"role": "assistant", "content": r.content})
            st.rerun()
        except:
            st.error("Cuota de chat agotada.")
