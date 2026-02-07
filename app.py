import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Retention Pro - Library Final", layout="wide")

@st.cache_resource
def load_models():
    # Obtenemos la Key de los secretos
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # Configuramos el LLM de forma limpia
    # Quitamos client_options para evitar el error de Pydantic
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Cargamos embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

# Inicialización de recursos
try:
    llm, embeddings = load_models()
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

# --- BASE DE DATOS SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Tabla Clientes
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], 
                 columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    # Tabla Suscripciones
    pd.DataFrame([[450, 'Streaming Premium', 15.99, 'Baja']], 
                 columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (LangChain Stable)")
st.success("✅ Librerías cargadas correctamente.")

queja = st.text_area("Queja del cliente:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🔍 Analizar Caso"):
    try:
        # A. Clasificación
        with st.spinner("Clasificando con Gemini..."):
            # Usamos el formato de mensaje que LangChain prefiere ahora
            res_clase = llm.invoke(f"Clasifica el motivo de esta queja en una sola palabra [Precio, Competencia, Calidad]: {queja}")
            st.session_state.motivo_detectado = res_clase.content.strip()
            st.info(f"**Motivo detectado:** {st.session_state.motivo_detectado}")

        time.sleep(1) # Respiro para la API

        # B. Text-to-SQL
        with st.spinner("Generando SQL de consulta..."):
            prompt_sql = """
            Genera un código SQL para SQLite que devuelva nombre, nivel, producto y cuota del cliente con ID 450.
            Tablas: clientes (id, nombre, nivel), suscripciones (id_cliente, producto, cuota).
            IMPORTANTE: Devuelve SOLO el texto del SQL, nada más.
            """
            res_sql = llm.invoke(prompt_sql)
            query = res_sql.content.strip().replace("```sql", "").replace("```", "").strip()
            st.code(query, language="sql")
            
            # Ejecución
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Hubo un problema con la IA: {e}")

# --- CHAT ---
st.divider()
if prompt := st.chat_input("Pregunta sobre el cliente..."):
    with st.chat_message("user"): st.write(prompt)
    try:
        respuesta = llm.invoke(prompt)
        with st.chat_message("assistant"): st.write(respuesta.content)
    except Exception as e:
        st.error(f"Error en el chat: {e}")
