import streamlit as st
import pandas as pd
import sqlite3
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Text-to-SQL & RAG Agent", layout="wide")

@st.cache_resource
def load_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    # Intentamos gemini-1.5-flash-latest que suele ser más estable en LangChain
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)
    except:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models()

# --- DB SQLITE (EN MEMORIA) ---
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Tabla Clientes
    pd.DataFrame([[450, 'Juan Pérez', 24, 'Premium']], 
                 columns=['id', 'nombre', 'antiguedad', 'nivel']).to_sql('clientes', conn, index=False)
    # Tabla Suscripciones
    pd.DataFrame([
        [450, 'Fibra 1Gbps', 40.0, 'Activo'],
        [450, 'Streaming Premium', 15.99, 'Baja'],
        [450, 'Móvil 50GB', 12.0, 'Activo']
    ], columns=['id_cliente', 'producto', 'cuota', 'estado']).to_sql('suscripciones', conn, index=False)
    return conn

conn = init_db()

# --- RAG ---
@st.cache_resource
def load_rag():
    reglas = ["OFERTA: 50% dto 3 meses.", "VENTAJA: Fútbol incluido.", "BONO: 100GB gratis."]
    return FAISS.from_texts(reglas, embeddings)

vectorstore = load_rag()

# --- LÓGICA DE IA CON REINTENTOS ---
def safe_invoke(prompt, label="IA"):
    for i in range(3):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            st.warning(f"⚠️ {label}: Error de cuota. Reintentando en 60s... (Intento {i+1}/3)")
            time.sleep(60)
    return None

# --- INTERFAZ ---
st.title("🛡️ Agente de Retención (Text-to-SQL + RAG)")
parrafo = st.text_area("Queja del cliente:", height=100)

if "resumen" not in st.session_state: st.session_state.resumen = ""
if "datos_sql" not in st.session_state: st.session_state.datos_sql = ""

if st.button("🚀 Ejecutar Análisis Completo"):
    # PASO 1: TEXT-TO-SQL
    with st.status("Generando y Ejecutando SQL...", expanded=True) as s:
        # Prompt específico para solo SQL
        prompt_sql = f"""
        Base de datos con tablas 'clientes' (id, nombre, antiguedad, nivel) 
        y 'suscripciones' (id_cliente, producto, cuota, estado).
        Escribe SOLO la sentencia SQL para obtener los productos y cuotas del cliente 450.
        No uses markdown, no expliques nada. Solo el código SQL.
        """
        resp_sql = safe_invoke(prompt_sql, "Fase SQL")
        if resp_sql:
            query = resp_sql.content.strip().replace("```sql", "").replace("```", "").strip()
            st.code(query, language="sql")
            try:
                df = pd.read_sql_query(query, conn)
                st.session_state.datos_sql = df.to_string()
                st.dataframe(df)
                s.update(label="SQL ejecutado con éxito", state="complete")
            except Exception as e:
                st.error(f"Error al ejecutar SQL: {e}")
                st.stop()
    
    st.info("Esperando 60s para refrescar cuota...")
    time.sleep(60)

    # PASO 2: RAG
    with st.status("Consultando RAG...", expanded=True) as s:
        docs = vectorstore.similarity_search(parrafo, k=2)
        contexto_rag = "\n".join([d.page_content for d in docs])
        st.write("Reglas encontradas:")
        st.code(contexto_rag)
        s.update(label="RAG completado", state="complete")

    st.info("Esperando 60s para informe final...")
    time.sleep(60)

    # PASO 3: INFORME
    with st.spinner("Redactando informe final..."):
        prompt_f = f"Resume este caso: {parrafo}. Datos cliente: {st.session_state.datos_sql}. Reglas: {contexto_rag}"
        informe = safe_invoke(prompt_f, "Fase Informe")
        if informe:
            st.session_state.resumen = informe.content
            st.success("### Resultado Final")
            st.markdown(st.session_state.resumen)

# --- CHAT ---
st.divider()
st.subheader("💬 Chat de Profundización")
if "chat_history" not in st.session_state: st.session_state.chat_history = []
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]): st.write(m["content"])

if p := st.chat_input("Pregunta algo..."):
    st.session_state.chat_history.append({"role": "user", "content": p})
    with st.chat_message("user"): st.write(p)
    # El chat usa el contexto acumulado
    r_chat = safe_invoke(f"Contexto: {st.session_state.resumen}. Pregunta: {p}", "Chat")
    if r_chat:
        st.session_state.chat_history.append({"role": "assistant", "content": r_chat.content})
        with st.chat_message("assistant"): st.write(r_chat.content)
