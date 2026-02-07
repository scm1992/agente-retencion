import streamlit as st
import pandas as pd
import sqlite3
import time
import google.generativeai as genai

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - 2.0 Flash", layout="wide")

# Usamos el nombre exacto que salió en tu diagnóstico
MODEL_NAME = "models/gemini-2.0-flash"

@st.cache_resource
def load_resources():
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    # Cargamos el modelo que SI tienes disponible
    model = genai.GenerativeModel(MODEL_NAME)
    return model

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
st.title("🛡️ Sistema de Retención (Serie 2.0 Flash)")
st.success(f"✅ Utilizando modelo validado: {MODEL_NAME}")

queja = st.text_area("Queja de Juan Pérez:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    try:
        # A. CLASIFICACIÓN
        with st.spinner("IA Clasificando motivo..."):
            # Print de control para ver qué enviamos
            prompt_clase = f"Clasifica el motivo en una palabra [Precio, Competencia, Calidad]: {queja}"
            response = model.generate_content(prompt_clase)
            st.info(f"**Motivo Detectado:** {response.text.strip()}")
        
        time.sleep(2) # Pausa para evitar el error 429 de cuota

        # B. TEXT-TO-SQL
        with st.spinner("Generando SQL..."):
            prompt_sql = """Genera SOLO el código SQL para SQLite (sin bloques markdown) para obtener: 
            nombre, nivel, producto y cuota del cliente id 450. 
            Tablas: clientes, suscripciones."""
            response_sql = model.generate_content(prompt_sql)
            
            # Limpieza del SQL
            query = response_sql.text.strip().replace("```sql", "").replace("```", "").strip()
            st.code(query, language="sql")
            
            # Ejecución
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error en la ejecución con {MODEL_NAME}: {e}")

# --- 4. CHAT ---
st.divider()
if p := st.chat_input("Pregunta sobre el cliente..."):
    with st.chat_message("user"): st.write(p)
    try:
        res_chat = model.generate_content(p)
        with st.chat_message("assistant"): st.write(res_chat.text)
    except Exception as e:
        st.error(f"Error en chat: {e}")
