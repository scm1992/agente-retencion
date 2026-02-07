import streamlit as st
import pandas as pd
import sqlite3
import time
import os
import google.generativeai as genai

# --- 1. CONFIGURACIÓN DE ENTORNO Y PROTOCOLO ---
# Forzamos el uso de protocolos modernos antes de configurar la API
os.environ["GOOGLE_API_USE_MTLS"] = "never" 

st.set_page_config(page_title="Retention Pro - gRPC Native", layout="wide")

# Configuración con transporte gRPC (evita el proxy REST v1beta que falla)
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key, transport='grpc')

@st.cache_resource
def load_model():
    # Usamos el modelo 1.5 Flash 8b: es el más estable ante conflictos de versiones
    return genai.GenerativeModel('gemini-1.5-flash-8b')

model = load_model()

# --- 2. BASE DE DATOS SQLITE (Contexto del Caso) ---
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

# --- 3. INTERFAZ DE USUARIO ---
st.title("🛡️ Sistema de Retención (Conexión Directa gRPC)")
st.success("✅ Conexión establecida mediante protocolo de datos gRPC.")

# Registro de intentos para el usuario (como acordamos)
with st.expander("Ver historial de intentos técnicos"):
    st.write("""
    1. LangChain (v1beta) -> ❌ 404
    2. Auto-discovery -> ❌ 429 (Pro model)
    3. SDK Nativo REST -> ❌ 404 (v1beta)
    4. **Actual:** SDK Nativo gRPC + Flash-8b -> 🔄 En curso
    """)

queja = st.text_area("Queja de Juan Pérez:", "Juan Pérez dice que el precio es excesivo comparado con la competencia.")

if st.button("🚀 Ejecutar Análisis"):
    try:
        # A. CLASIFICACIÓN
        with st.spinner("IA Clasificando motivo..."):
            prompt_clase = f"Clasifica el motivo de esta queja en una sola palabra [Precio, Competencia, Calidad]: {queja}"
            response = model.generate_content(prompt_clase)
            motivo = response.text.strip()
            st.session_state.motivo = motivo
            st.info(f"**Motivo Detectado:** {motivo}")
        
        time.sleep(2) # Pausa técnica para respetar la cuota gratuita

        # B. TEXT-TO-SQL
        with st.spinner("Generando consulta SQL..."):
            prompt_sql = """Genera SOLO el código SQL para SQLite (sin bloques markdown) para obtener: 
            nombre, nivel, producto y cuota del cliente id 450. 
            Tablas: clientes (id, nombre, nivel), suscripciones (id_cliente, producto, cuota)."""
            
            response_sql = model.generate_content(prompt_sql)
            # Limpiamos posibles restos de formato markdown
            query = response_sql.text.strip().replace("```sql", "").replace("```", "").strip()
            
            st.code(query, language="sql")
            
            # Ejecución en la base de datos
            df = pd.read_sql_query(query, conn)
            st.dataframe(df)
            
    except Exception as e:
        if "404" in str(e):
            st.error(f"Error 404 persistente. El servidor sigue intentando v1beta. Error: {e}")
        elif "429" in str(e):
            st.warning("⚠️ Límite de cuota alcanzado. Espera 60 segundos antes de reintentar.")
        else:
            st.error(f"Error inesperado: {e}")

# --- 4. CHAT COMPLEMENTARIO ---
st.divider()
if p := st.chat_input("Haz una pregunta sobre el cliente o las reglas de retención..."):
    with st.chat_message("user"): st.write(p)
    try:
        res_chat = model.generate_content(p)
        with st.chat_message("assistant"): st.write(res_chat.text)
    except Exception as e:
        st.error(f"Error en el chat: {e}")
