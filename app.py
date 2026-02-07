import streamlit as st
import pandas as pd
import sqlite3
import requests
import json
import time

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - V1 Direct", layout="wide")

API_KEY = st.secrets["GOOGLE_API_KEY"]
# FORZAMOS MANUALMENTE LA VERSIÓN v1 (ESTABLE) EN LA URL
API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"

def call_gemini_v1(prompt):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
            "maxOutputTokens": 800
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        error_msg = response.text
        # Si Google nos dice que el modelo no existe en v1, probamos el alias universal
        if "404" in str(response.status_code):
            alt_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"
            response = requests.post(alt_url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
        
        raise Exception(f"Error {response.status_code}: {error_msg}")

# --- DB SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Streaming Premium', 15.99, 'Baja']], columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (Fuerza Bruta v1)")
st.warning("⚠️ Saltando librerías corruptas. Conexión directa por HTTP v1.")

queja = st.text_area("Queja de Juan Pérez:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    try:
        # A. CLASIFICACIÓN
        with st.spinner("IA Clasificando (v1)..."):
            prompt_clase = f"Clasifica el motivo de esta queja en una sola palabra [Precio, Competencia, Calidad]: {queja}"
            motivo = call_gemini_v1(prompt_clase)
            st.info(f"**Motivo Detectado:** {motivo.strip()}")
        
        time.sleep(1)

        # B. TEXT-TO-SQL
        with st.spinner("Generando SQL (v1)..."):
            prompt_sql = """Genera SOLO el código SQL para SQLite (sin markdown) para obtener: 
            nombre, nivel, producto y cuota del cliente id 450. 
            Tablas: clientes (id, nombre, nivel), suscripciones (id_cliente, producto, cuota)."""
            
            sql_raw = call_gemini_v1(prompt_sql)
            query = sql_raw.strip().replace("```sql", "").replace("```", "").strip()
            
            st.code(query, language="sql")
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Fallo en la conexión directa: {e}")

# --- CHAT ---
st.divider()
if p := st.chat_input("Pregunta algo..."):
    with st.chat_message("user"): st.write(p)
    try:
        respuesta = call_gemini_v1(p)
        with st.chat_message("assistant"): st.write(respuesta)
    except Exception as e:
        st.error(f"Error: {e}")
