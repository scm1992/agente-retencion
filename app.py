import streamlit as st
import pandas as pd
import sqlite3
import time
import google.generativeai as genai

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - Native Fix", layout="wide")

# Configuración de la API Key
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

@st.cache_resource
def load_model():
    # Usamos el SDK oficial de Google que gestiona las versiones automáticamente
    return genai.GenerativeModel('gemini-1.5-flash')

model = load_model()

# --- DB SQLITE ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    pd.DataFrame([[450, 'Streaming Premium', 15.99, 'Baja']], columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Retención (Conexión Nativa)")
st.success("✅ SDK de Google configurado.")

queja = st.text_area("Queja del cliente:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    try:
        # A. CLASIFICACIÓN
        with st.spinner("Clasificando..."):
            prompt_clase = f"Clasifica el motivo en una palabra [Precio, Competencia, Calidad]: {queja}"
            response = model.generate_content(prompt_clase)
            motivo = response.text.strip()
            st.info(f"**Motivo Detectado:** {motivo}")
        
        time.sleep(2) # Pausa de seguridad

        # B. TEXT-TO-SQL
        with st.spinner("Generando SQL..."):
            prompt_sql = """Genera SOLO el código SQL para SQLite para obtener: nombre, nivel, producto, cuota.
            Del cliente id 450. Tablas: clientes, suscripciones. No uses markdown."""
            response_sql = model.generate_content(prompt_sql)
            query = response_sql.text.strip().replace("```sql", "").replace("```", "").strip()
            
            st.code(query, language="sql")
            
            # Ejecución
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        if "429" in str(e):
            st.warning("⚠️ Límite de cuota alcanzado. Espera 60 segundos.")
        else:
            st.error(f"Error: {e}")

# --- CHAT ---
st.divider()
if p := st.chat_input("Pregunta algo sobre el caso..."):
    with st.chat_message("user"): st.write(p)
    res_chat = model.generate_content(p)
    with st.chat_message("assistant"): st.write(res_chat.text)
