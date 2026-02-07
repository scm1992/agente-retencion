import streamlit as st
import pandas as pd
import sqlite3
import time
import google.generativeai as genai

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - Old Key Test", layout="wide")

# Usamos la Key antigua que vas a poner en los Secrets
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

@st.cache_resource
def load_resources():
    # Intentamos el modelo 2.0-flash que apareció en tu lista
    # Si la Key es antigua y tiene cuota, este volará.
    return genai.GenerativeModel("models/gemini-2.0-flash")

model = load_resources()

# --- 2. BASE DE DATOS SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Datos de Juan Pérez
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], 
                 columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Streaming Premium', 15.99, 'Baja']], 
                 columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- 3. INTERFAZ ---
st.title("🛡️ Sistema de Retención (Test con Key Antigua)")
st.info("Intentando ejecutar con models/gemini-2.0-flash...")

queja = st.text_area("Queja de Juan Pérez:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    log = st.empty()
    try:
        log.write("⏳ Paso 1: Clasificando...")
        # CLASIFICACIÓN
        res = model.generate_content(f"Clasifica en una palabra [Precio, Competencia]: {queja}")
        st.success(f"**Motivo Detectado:** {res.text.strip()}")
        
        time.sleep(2) # Respiro para la cuota

        log.write("⏳ Paso 2: Generando SQL...")
        # SQL
        prompt_sql = "Genera SQL para SQLite: SELECT nombre, nivel, producto, cuota FROM clientes JOIN suscripciones ON clientes.id = suscripciones.id_cliente WHERE clientes.id=450. Solo el código."
        res_sql = model.generate_content(prompt_sql)
        query = res_sql.text.strip().replace("```sql", "").replace("```", "").strip()
        
        st.code(query, language="sql")
        
        log.write("⏳ Paso 3: Consultando DB...")
        df = pd.read_sql_query(query, conn)
        st.dataframe(df)
        log.write("✅ ¡Éxito!")
            
    except Exception as e:
        log.empty()
        if "429" in str(e):
            st.error("Esta Key también tiene cuota 0 o está saturada. El error 429 confirma que el modelo existe pero la cuenta no tiene permiso de uso.")
        elif "404" in str(e):
            st.warning("El modelo 2.0-flash no responde. Intentando con gemini-pro-latest...")
            # Fallback rápido si el 404 persiste
            try:
                alt_model = genai.GenerativeModel("models/gemini-pro-latest")
                res = alt_model.generate_content(f"Clasifica: {queja}")
                st.success(f"**Motivo (vía Pro):** {res.text.strip()}")
            except Exception as e2:
                st.error(f"Fallo total: {e2}")
        else:
            st.error(f"Error: {e}")

# --- 4. CHAT ---
st.divider()
if p := st.chat_input("Pregunta algo..."):
    with st.chat_message("user"): st.write(p)
    try:
        res_chat = model.generate_content(p)
        with st.chat_message("assistant"): st.write(res_chat.text)
    except:
        st.error("No se pudo obtener respuesta del chat.")
