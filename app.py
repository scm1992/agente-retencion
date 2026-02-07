import streamlit as st
import pandas as pd
import sqlite3
import google.generativeai as genai
import platform
import time

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Debug Mode - Retention", layout="wide")

def run_diagnostics():
    st.subheader("Diagnostic Log")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Python Version:** {platform.python_version()}")
        st.write(f"**GenAI Version:** {genai.__version__}")
    with col2:
        try:
            # Intentamos listar los modelos que tu API KEY realmente puede ver
            api_key = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=api_key)
            models = [m.name for m in genai.list_models()]
            st.write("**Modelos disponibles en tu Key:**")
            st.write(models)
            return models
        except Exception as e:
            st.error(f"Error en diagnóstico de Key: {e}")
            return []

# Ejecutamos el print de diagnóstico en la UI
available_models = run_diagnostics()

@st.cache_resource
def load_resources():
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    
    # ESTRATEGIA: Probamos el nombre sin el prefijo 'models/' 
    # por si la librería lo está duplicando internamente.
    model_to_use = "gemini-1.5-flash" 
    
    # Si en el diagnóstico vimos nombres, intentamos el primero que sea flash
    for m in available_models:
        if "1.5-flash" in m:
            model_to_use = m # Esto suele ser 'models/gemini-1.5-flash'
            break
            
    return genai.GenerativeModel(model_to_use)

model = load_resources()

# --- DB SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Streaming Premium', 15.99, 'Baja']], columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- INTERFAZ ---
st.divider()
st.title("🛡️ Sistema de Retención (Library Debug Mode)")

queja = st.text_area("Queja de Juan Pérez:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    st.write("### Proceso interno:")
    
    # PASO 1: Clasificación
    try:
        st.write("1. Enviando prompt de clasificación...")
        response = model.generate_content(f"Clasifica: [Precio, Competencia]. Queja: {queja}")
        st.success(f"**Motivo:** {response.text}")
    except Exception as e:
        st.error(f"Fallo en Paso 1: {e}")
        st.stop()

    time.sleep(1)

    # PASO 2: SQL
    try:
        st.write("2. Generando SQL...")
        prompt_sql = "Genera SQL para SQLite: SELECT nombre, nivel FROM clientes WHERE id=450. Solo el código."
        response_sql = model.generate_content(prompt_sql)
        query = response_sql.text.strip().replace("```sql", "").replace("```", "").strip()
        st.code(query, language="sql")
        
        df = pd.read_sql_query(query, conn)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Fallo en Paso 2: {e}")

# --- CHAT ---
st.divider()
if p := st.chat_input("Pregunta algo..."):
    with st.chat_message("user"): st.write(p)
    try:
        res = model.generate_content(p)
        with st.chat_message("assistant"): st.write(res.text)
    except Exception as e:
        st.error(f"Error: {e}")
