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

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    # Usamos temperatura 0 para que la IA sea más precisa y no "alucine"
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- BLINDAJE ULTRA-RESILIENTE ---
def safe_invoke(prompt):
    """Llama a la IA con reintentos largos para asegurar la cuota gratuita"""
    for intento in range(3):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                espera = 15 + (intento * 5) # Espera 15s, luego 20s...
                st.warning(f"⚠️ API Saturada. Esperando {espera} segundos para reintentar...")
                time.sleep(espera)
                continue
            st.error(f"Error inesperado: {e}")
            return None
    return None

# --- BASE DE DATOS (SQL REAL) ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Tabla de Clientes
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium', 'Alta']], 
                 columns=['id', 'nombre', 'antiguedad', 'nivel_segmento', 'fidelidad']).to_sql('clientes', conn, index=False)
    # Tabla de Suscripciones
    pd.DataFrame([
        [450, 'Fibra 1Gbps', 40.0, 'Activo', 'Internet'],
        [450, 'Streaming Premium', 15.99, 'En Proceso Baja', 'Ocio'],
        [450, 'Línea Móvil 50GB', 12.0, 'Activo', 'Movilidad']
    ], columns=['id_cliente', 'producto', 'cuota', 'estado', 'categoria']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- MOTOR RAG (MANUAL REAL) ---
@st.cache_resource
def load_rag():
    reglas = [
        "REGLA_OCIO: Si el cliente quiere bajar Streaming, ofrecer 50% dto por 3 meses (Oferta Retención Ocio).",
        "VENTAJA_COMPETITIVA: StreamMax no incluye deportes en vivo ni fútbol, nuestra plataforma sí.",
        "REGLA_FIDELIDAD: Clientes con >12 meses y nivel Premium tienen prioridad para descuentos en paquetes de Fibra."
    ]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

def tool_sql(pregunta, id_cliente=450):
    prompt_sql = f"Dada la tabla 'clientes' (id, nombre, antiguedad, nivel_segmento, fidelidad) y 'suscripciones' (id_cliente, producto, cuota, estado, categoria). Escribe SOLO la sentencia SQL para: {pregunta} del id {id_cliente}. No uses Markdown."
    resp = safe_invoke(prompt_sql)
    if resp:
        query = resp.content.strip().replace("```sql", "").replace("```", "").strip()
        try:
            df = pd.read_sql_query(query, conn)
            return df.to_string(index=False)
        except:
            return "No se encontraron datos específicos en la DB."
    return "Error al conectar con la base de datos."

# --- INTERFAZ DE USUARIO ---
st.title("🛡️ Panel de Retención Inteligente (Resilient Mode)")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📥 Entrada del Cliente")
    parrafo = st.text_area("Copia aquí la queja o transcripción:", height=200, placeholder="Ej: Juan Pérez quiere darse de baja porque StreamMax es más barato...")
    
    if st.button("🔍 1. Analizar Motivo"):
        if parrafo:
            with st.spinner("Analizando intención..."):
                resp = safe_invoke(f"Clasifica esta queja en una sola palabra (Precio, Competencia o Calidad): {parrafo}")
                if resp: st.success(f"**Motivo detectado:** {resp.content}")
        else:
            st.warning("Introduce un texto primero.")

with col2:
    st.subheader("📊 2. Resumen Ejecutivo de Retención")
    if "resumen" not in st.session_state: st.session_state.resumen = ""

    if st.button("🚀 Generar Análisis Completo"):
        if not parrafo:
            st.warning("Escribe la queja antes de analizar.")
        else:
            # PASO 1: SQL
            with st.spinner("Consultando perfiles y productos (Paso 1/3)..."):
                sql_data = tool_sql("Lista todos los productos y cuotas")
                time.sleep(12) # Pausa estratégica
            
            # PASO 2: RAG
            with st.spinner("Buscando mejores ofertas en manuales (Paso 2/3)..."):
                docs = vectorstore.similarity_search(parrafo, k=2)
                reglas_encontradas = "\n".join([d.page_content for d in docs])
                time.sleep(12) # Pausa estratégica
            
            # PASO 3: RESUMEN
            with st.spinner("Redactando propuesta final (Paso 3/3)..."):
                prompt_final = f"""
                Actúa como experto en retención. Usa estos datos:
                DATOS CLIENTE: {sql_data}
                REGLAS APLICABLES: {reglas_encontradas}
                QUEJA: {parrafo}
                
                Escribe un resumen con este formato:
                - CLIENTE: (Nombre y antigüedad)
                - ESTADO ACTUAL: (Qué productos tiene y cuánto paga)
                - RIESGO: (Por qué se quiere ir)
                - PROPUESTA: (Usa las reglas para ofrecer una solución específica)
                """
                resp_final = safe_invoke(prompt_final)
                if resp_final:
                    st.session_state.resumen = resp_final.content

    if st.session_state.resumen:
        st.info("### Análisis Generado")
        st.markdown(st.session_state.resumen)

