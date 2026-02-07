import streamlit as st
import pandas as pd
import sqlite3
import time
import google.generativeai as genai

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - Quota Fix", layout="wide")

# Usamos el alias universal que suele tener cuota abierta: 'gemini-pro-latest'
MODEL_NAME = "models/gemini-pro-latest"

@st.cache_resource
def load_resources():
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)

model = load_resources()

# --- 2. BASE DE DATOS SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], 
                 columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Streaming Premium', 15.99, 'Baja']], 
                 columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- 3. INTERFAZ ---
st.title("🛡️ Sistema de Retención (Modelo de Cuota Estable)")
st.info(f"Probando modelo con cuota universal: {MODEL_NAME}")

queja = st.text_area("Queja de Juan Pérez:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    # SISTEMA DE PRINTS (Log de ejecución)
    log_area = st.empty()
    
    try:
        log_area.write("⏳ Paso 1: Clasificando motivo...")
        res = model.generate_content(f"Clasifica: [Precio, Competencia]. Queja: {queja}")
        motivo = res.text.strip()
        st.success(f"**Motivo Detectado:** {motivo}")
        
        log_area.write("⏳ Paso 2: Esperando 5 segundos para no saturar cuota...")
        time.sleep(5) 

        log_area.write("⏳ Paso 3: Generando SQL...")
        prompt_sql = "Genera SQL para SQLite: SELECT nombre, nivel, producto, cuota FROM clientes JOIN suscripciones ON clientes.id = suscripciones.id_cliente WHERE clientes.id=450. Solo el código."
        res_sql = model.generate_content(prompt_sql)
        query = res_sql.text.strip().replace("```sql", "").replace("```", "").strip()
        
        st.code(query, language="sql")
        
        log_area.write("⏳ Paso 4: Consultando base de datos...")
        df = pd.read_sql_query(query, conn)
        st.dataframe(df)
        
        log_area.write("✅ Proceso completado con éxito.")
            
    except Exception as e:
        log_area.empty()
        if "429" in str(e):
            st.error(f"Tu cuenta aún tiene cuota 0 para {MODEL_NAME}. Google tarda hasta 24h en activar cuotas en proyectos nuevos. Prueba a crear otra API Key en un proyecto distinto en AI Studio.")
        else:
            st.error(f"Error en ejecución: {e}")

# --- 4. CHAT ---
st.divider()
if p := st.chat_input("Pregunta algo..."):
    with st.chat_message("user"): st.write(p)
    try:
        res_chat = model.generate_content(p)
        with st.chat_message("assistant"): st.write(res_chat.text)
    except Exception as e:
        st.error(f"Error en chat: {e}")
