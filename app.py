import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - Library Fix", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    
    # FORZAMOS LA VERSIÓN ESTABLE Y EL NOMBRE DEL MODELO
    # Usamos gemini-1.5-flash que es el estándar actual
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key, 
        temperature=0,
        # Esto es clave para evitar el error 404 en algunas regiones:
        client_options={"api_version": "v1"} 
    )
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- DB SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Fibra 1Gbps', 40.0, 'Activo']], columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (Library Mode)")
st.info(f"Modelo configurado en LangChain: {llm.model}")

queja = st.text_area("Queja del cliente:", "Juan Pérez quiere la baja porque es caro.")

if st.button("🚀 Analizar Caso"):
    try:
        # A. CLASIFICACIÓN
        with st.spinner("IA Clasificando..."):
            # Usamos invoke que es el método estándar de LangChain
            res_clase = llm.invoke(f"Clasifica en una palabra [Precio, Competencia]: {queja}")
            st.success(f"**Motivo:** {res_clase.content.strip()}")
        
        time.sleep(2) # Pausa para no saturar la cuota

        # B. TEXT-TO-SQL
        with st.spinner("IA Generando SQL..."):
            prompt_sql = """Genera un SQL para SQLite para obtener nombre, nivel, producto y cuota del cliente 450. 
            Tablas: clientes (id, nombre, nivel), suscripciones (id_cliente, producto, cuota). 
            Solo el código SQL."""
            res_sql = llm.invoke(prompt_sql)
            query = res_sql.content.strip().replace("```sql", "").replace("```", "").strip()
            st.code(query, language="sql")
            
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error detectado en la librería: {e}")

# --- CHAT ---
st.divider()
if p := st.chat_input("Chatea con los datos..."):
    with st.chat_message("user"): st.write(p)
    try:
        res_chat = llm.invoke(p)
        with st.chat_message("assistant"): st.write(res_chat.content)
    except Exception as e:
        st.error(f"Error en chat: {e}")
