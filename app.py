import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Retention Pro - Full Architecture", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
    # Cambiamos a gemini-1.5-flash que tiene una cuota mucho más relajada y estable
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- BLINDAJE ---
def safe_invoke(prompt, step_name=""):
    for intento in range(3):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                espera = 90 + (intento * 30)
                st.warning(f"⚠️ [{step_name}] Cuota agotada. Esperando {espera}s... (Intento {intento+1}/3)")
                time.sleep(espera)
                continue
            st.error(f"Error en {step_name}: {e}")
            return None
    return None

# --- DB SQL REAL ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium', 'Alta']], columns=['id', 'nombre', 'antiguedad', 'nivel_segmento', 'fidelidad']).to_sql('clientes', conn, index=False)
    pd.DataFrame([
        [450, 'Fibra 1Gbps', 40.0, 'Activo'],
        [450, 'Streaming Premium', 15.99, 'Baja'],
        [450, 'Móvil 50GB', 12.0, 'Activo']
    ], columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- RAG REAL ---
@st.cache_resource
def load_rag():
    reglas = [
        "OFERTA_OCIO: 50% dto 3 meses en Streaming si el cliente se queja del precio.",
        "VENTAJA_FUTBOL: Recordar que StreamMax no tiene derechos de fútbol y nosotros sí.",
        "BONO_DATOS: Ofrecer bono 100GB gratis si el cliente es Premium (como el ID 450)."
    ]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

# --- VARIABLES DE ESTADO ---
if "resumen_final" not in st.session_state: st.session_state.resumen_final = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "datos_cliente_str" not in st.session_state: st.session_state.datos_cliente_str = ""

# --- INTERFAZ ---
st.title("🛡️ Sistema Integral de Retención")
st.markdown("Esta versión ejecuta SQL real, consulta manuales RAG y permite chat de seguimiento.")

parrafo = st.text_area("Introduzca la queja del cliente:", height=150)

# BOTONES DE ACCIÓN
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🚀 Ejecutar Análisis Completo"):
        if not parrafo:
            st.warning("Por favor, escriba la queja.")
        else:
            # FASE 1: SQL
            with st.status("Fase 1: Consultando SQL...", expanded=True) as s:
                prompt_sql = "Escribe SOLO el SQL para: 'listar productos y cuotas del cliente 450'. Tablas: clientes, suscripciones. No uses markdown."
                resp_sql = safe_invoke(prompt_sql, "Extracción de Datos")
                if resp_sql:
                    query = resp_sql.content.strip().replace("```sql", "").replace("```", "").strip()
                    df_datos = pd.read_sql_query(query, conn)
                    st.session_state.datos_cliente_str = df_datos.to_string()
                    st.dataframe(df_datos) # Verificación visual
                    s.update(label="Datos SQL extraídos ✅", state="complete")
                else: st.stop()

            st.info("🕒 Enfriando API (90s) para cumplir cuota gratuita...")
            time.sleep(90)

            # FASE 2: RAG
            with st.status("Fase 2: Consultando RAG...", expanded=True) as s:
                docs = vectorstore.similarity_search(parrafo, k=2)
                contexto_rag = "\n".join([d.page_content for d in docs])
                st.code(contexto_rag) # Verificación visual de reglas
                s.update(label="Reglas RAG recuperadas ✅", state="complete")

            st.info("🕒 Enfriando API (90s) para informe final...")
            time.sleep(90)

            # FASE 3: INFORME
            with st.status("Fase 3: Generando Informe...", expanded=True) as s:
                prompt_f = f"Genera informe de retención. DATOS: {st.session_state.datos_cliente_str}. REGLAS: {contexto_rag}. QUEJA: {parrafo}"
                res = safe_invoke(prompt_f, "Generación Informe")
                if res:
                    st.session_state.resumen_final = res.content
                    s.update(label="Informe finalizado ✅", state="complete")

# EXPOSICIÓN DEL RESULTADO
if st.session_state.resumen_final:
    st.success("### 📄 Informe de Retención Generado")
    st.markdown(st.session_state.resumen_final)

# --- SECCIÓN DEL CHAT ---
st.divider()
st.subheader("💬 Chat de Profundización")
st.caption("Pregunta detalles adicionales sobre el cliente o la estrategia.")

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt_chat := st.chat_input("Ej: ¿Cuál es el impacto en margen de esta oferta?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt_chat})
    with st.chat_message("user"): st.markdown(prompt_chat)
    
    with st.spinner("IA analizando datos históricos..."):
        # Contexto enriquecido para el chat
        contexto_completo = f"Datos Cliente: {st.session_state.datos_cliente_str}. Informe previo: {st.session_state.resumen_final}"
        res_chat = safe_invoke(f"Contexto: {contexto_completo}. Pregunta: {prompt_chat}", "Chat")
        if res_chat:
            st.session_state.chat_history.append({"role": "assistant", "content": res_chat.content})
            with st.chat_message("assistant"): st.markdown(res_chat.content)

