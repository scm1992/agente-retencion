import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Arquitectura Real Resiliente", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- BLINDAJE CON LOGS ---
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
    reglas = ["OFERTA_OCIO: 50% dto 3 meses en Streaming.", "VENTAJA: Fútbol incluido.", "FIDELIDAD: Regalo bono 100GB."]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

# --- INTERFAZ ---
st.title("🛡️ Agente de Retención: Arquitectura SQL + RAG")
parrafo = st.text_area("Queja del cliente:", height=150)

if st.button("🚀 Ejecutar Proceso Completo (SQL -> RAG -> IA)"):
    if not parrafo:
        st.warning("Escribe la queja.")
    else:
        # CONTENEDORES PARA VER EL PROGRESO
        status_sql = st.empty()
        status_rag = st.empty()
        status_final = st.empty()

        # PASO 1: SQL
        with st.status("Fase 1: Consultando Base de Datos SQL...", expanded=True) as s:
            st.write("Generando sentencia SQL...")
            prompt_sql = "Escribe SOLO el SQL para: 'listar productos y cuotas del cliente 450'. Tablas: clientes, suscripciones. No markdown."
            resp_sql = safe_invoke(prompt_sql, "SQL")
            
            if resp_sql:
                query = resp_sql.content.strip().replace("```sql", "").replace("```", "").strip()
                st.write(f"Ejecutando: `{query}`")
                df_datos = pd.read_sql_query(query, conn)
                st.dataframe(df_datos) # MOSTRAR LOS DATOS EXTRAÍDOS
                contexto_sql = df_datos.to_string()
                s.update(label="Fase 1: SQL Completado ✅", state="complete")
            else:
                s.update(label="Fase 1: Error ❌", state="error")
                st.stop()

        st.info("🕒 Enfriando API por 90 segundos antes del siguiente paso...")
        time.sleep(90)

        # PASO 2: RAG (FAISS es local, no gasta cuota de IA, pero lo ponemos aquí)
        with st.status("Fase 2: Consultando Manuales (RAG)...", expanded=True) as s:
            docs = vectorstore.similarity_search(parrafo, k=2)
            contexto_rag = "\n".join([d.page_content for d in docs])
            st.write("Reglas de negocio encontradas:")
            st.code(contexto_rag) # MOSTRAR LAS REGLAS EXTRAÍDAS
            s.update(label="Fase 2: RAG Completado ✅", state="complete")

        st.info("🕒 Enfriando API por 90 segundos antes del Informe Final...")
        time.sleep(90)

        # PASO 3: INFORME FINAL
        with st.status("Fase 3: Redactando Informe con IA...", expanded=True) as s:
            prompt_final = f"Crea un informe de retención. DATOS SQL: {contexto_sql}. REGLAS RAG: {contexto_rag}. QUEJA: {parrafo}"
            resumen = safe_invoke(prompt_final, "Informe Final")
            if resumen:
                st.success("### 📄 INFORME FINAL GENERADO")
                st.markdown(resumen.content)
                s.update(label="Fase 3: Informe Completado ✅", state="complete")
            else:
                s.update(label="Fase 3: Error ❌", state="error")
