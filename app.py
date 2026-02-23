import streamlit as st
import streamlit_authenticator as stauth
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# ✅ Modern LangChain RAG imports (2026)
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import tempfile
import os
import re
import yaml
import bcrypt
from yaml.loader import SafeLoader

# Configure page
st.set_page_config(
    page_title="📚 Groq Study Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Load users from YAML ───────────────────────────────────────────────────
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.yaml")

def load_users():
    with open(USERS_FILE) as f:
        return yaml.load(f, Loader=SafeLoader)

def save_users(config):
    with open(USERS_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0E2148; }

    /* ── Auth form wrapper ── */
    .auth-box {
        background-color: #1C2F5E;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        max-width: 420px;
        margin: 3rem auto;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .auth-title {
        text-align: center;
        color: #E3D095;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .auth-subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* ── Main app cards ── */
    .main { padding: 2rem; max-width: 900px; margin: 0 auto; }
    .card {
        background-color: #7965C1;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .card-question { font-weight: bold; font-size: 1.1rem; color: #E3D095; }
    .card-answer { margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; }
    .header { text-align: center; margin-bottom: 2rem; }
    .header h1 { color: #E3D095; margin-bottom: 0.5rem; }
    .mode-selector {
        background-color: #483AA0; padding: 15px;
        border-radius: 10px; margin: 20px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .upload-section {
        background-color: #483AA0; padding: 20px;
        border-radius: 10px; margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* Hide default Streamlit auth labels colour */
    label { color: #E3D095 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Auth Setup ──────────────────────────────────────────────────────────────
config = load_users()

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# ─── Session state for page switching + signup field persistence ─────────────
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"   # "login" | "signup"

# Persist signup field values so they survive validation reruns
for _key in ["signup_name", "signup_username", "signup_email", "signup_error", "signup_success"]:
    if _key not in st.session_state:
        st.session_state[_key] = ""

# ════════════════════════════════════════════════════════════════════════════
# SIGNUP PAGE
# ════════════════════════════════════════════════════════════════════════════
def show_signup():
    st.markdown("""
    <div style="text-align:center; margin-top:2rem;">
        <h1 style="color:#E3D095;">📚 Groq Study Assistant</h1>
        <p style="color:#a0aec0;">Create your free account</p>
    </div>
    """, unsafe_allow_html=True)

    # Show persistent feedback messages
    if st.session_state.signup_error:
        st.error(st.session_state.signup_error)
    if st.session_state.signup_success:
        st.success(st.session_state.signup_success)

    # ── Form WITHOUT clear_on_submit so values are NOT wiped on error ────────
    with st.form("signup_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("👤 Full Name", value=st.session_state.signup_name)
        with col2:
            new_username = st.text_input("🔑 Username", value=st.session_state.signup_username)

        new_email    = st.text_input("📧 Email",            value=st.session_state.signup_email)
        new_password = st.text_input("🔒 Password",         type="password")
        confirm_pw   = st.text_input("🔒 Confirm Password", type="password")
        submit = st.form_submit_button("✅ Create Account", use_container_width=True)

        if submit:
            # ── Persist non-sensitive fields immediately ──────────────────
            st.session_state.signup_name     = new_name.strip()
            st.session_state.signup_username = new_username.strip().lower()
            st.session_state.signup_email    = new_email.strip()
            st.session_state.signup_error    = ""
            st.session_state.signup_success  = ""

            name     = st.session_state.signup_name
            username = st.session_state.signup_username
            email    = st.session_state.signup_email

            # ── Validation ───────────────────────────────────────────────
            if not all([name, username, email, new_password, confirm_pw]):
                st.session_state.signup_error = "⚠️ Please fill in all fields."
            elif new_password != confirm_pw:
                st.session_state.signup_error = "❌ Passwords do not match."
            elif len(new_password) < 6:
                st.session_state.signup_error = "⚠️ Password must be at least 6 characters."
            elif username in config["credentials"]["usernames"]:
                st.session_state.signup_error = "❌ Username already exists. Please choose another."
            elif "@" not in email:
                st.session_state.signup_error = "⚠️ Please enter a valid email."
            else:
                # ── Hash & save ──────────────────────────────────────────
                hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt(12)).decode()
                config["credentials"]["usernames"][username] = {
                    "email": email,
                    "name": name,
                    "password": hashed_pw,
                }
                save_users(config)
                # Clear all signup state on success
                for _k in ["signup_name", "signup_username", "signup_email", "signup_error"]:
                    st.session_state[_k] = ""
                st.session_state.signup_success = f"🎉 Account created for **{name}**! You can now log in."
                st.session_state.auth_page = "login"
            st.rerun()

    st.markdown("---")
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.markdown('<p style="color:#a0aec0; margin-top:0.6rem;">Already have an account?</p>', unsafe_allow_html=True)
    with col_r:
        if st.button("🔐 Log In", use_container_width=True):
            st.session_state.signup_error   = ""
            st.session_state.signup_success = ""
            st.session_state.auth_page = "login"
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE  (rendered by streamlit-authenticator)
# ════════════════════════════════════════════════════════════════════════════
def show_login():
    st.markdown("""
    <div style="text-align:center; margin-top:2rem;">
        <h1 style="color:#E3D095;">📚 Groq Study Assistant</h1>
        <p style="color:#a0aec0;">Sign in to continue</p>
    </div>
    """, unsafe_allow_html=True)

    # Show success message carried over from signup
    if st.session_state.get("signup_success"):
        st.success(st.session_state.signup_success)
        st.session_state.signup_success = ""

    authenticator.login(location="main")

    if st.session_state.get("authentication_status") is False:
        st.error("❌ Incorrect username or password.")
    elif st.session_state.get("authentication_status") is None:
        st.info("ℹ️ Enter your credentials above to log in.")

    st.markdown("---")
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.markdown('<p style="color:#a0aec0; margin-top:0.6rem;">Don\'t have an account?</p>', unsafe_allow_html=True)
    with col_r:
        if st.button("📝 Sign Up", use_container_width=True):
            st.session_state.auth_page = "signup"
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# ROUTE: show login / signup until authenticated
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.get("authentication_status"):
    if st.session_state.auth_page == "signup":
        show_signup()
    else:
        show_login()
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP  (only reached when logged in)
# ════════════════════════════════════════════════════════════════════════════

# ── Top-right logout button ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👋 Welcome, **{st.session_state.get('name', 'User')}**!")
    st.markdown(f"🔑 `{st.session_state.get('username', '')}`")
    st.markdown("---")
    authenticator.logout("🚪 Logout", location="sidebar")

# Header
st.markdown('<div class="header"><h1>📚 Groq Study Assistant</h1></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #E3D095; margin-bottom: 2rem;">Ask questions, summarize topics, or generate flashcards from your study material!</p>', unsafe_allow_html=True)

# Configure Groq API key
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚫 Groq API Key not found in secrets. Please configure it to use this app.")
    st.stop()

# File uploader
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload a text based PDF (Lecture Notes, Textbook, etc.)", type=["pdf"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_file_path = tmp.name

    st.success("✅ PDF uploaded successfully!")

    with st.spinner("🔍 Reading and processing document..."):
        progress_bar = st.progress(0)

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        pages = pages[:50]

        progress_bar.progress(40)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)
        if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
            st.error("🚫 The uploaded PDF appears to contain no readable text. It may be scanned or image-based. Please upload a text-based PDF.")
            os.unlink(temp_file_path)
            st.stop()


        progress_bar.progress(60)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        progress_bar.progress(80)

        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.7
        )

        # ✅ Modern Prompt
        prompt = ChatPromptTemplate.from_template("""
You are a helpful study assistant. Answer the question using ONLY the context below.

<context>
{context}
</context>

Question: {input}

Give a clear, concise, student-friendly answer.
""")

        # ✅ Modern RAG Chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = (
    RunnableParallel(
        context=(lambda x: x["input"]) | retriever,
        input=RunnablePassthrough()
    )
    | document_chain
)

        progress_bar.progress(100)

    st.success("📄 Document is ready for queries!")

    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    mode = st.selectbox(
        "Choose a Mode:",
        ["📖 Ask a Question", "📝 Summarize Notes", "🧠 Generate Flashcards"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Ask Question ----------
    if mode == "📖 Ask a Question":
        question = st.text_input("❓ Type your question about the content:")
        if question and st.button("Ask", use_container_width=True):
            with st.spinner("🧠 Thinking..."):
                result = qa_chain.invoke({"input": question})
                answer = result

                st.markdown(f"""
                <div class="card">
                    <div class="card-question">Question: {question}</div>
                    <div class="card-answer">{answer}</div>
                </div>
                """, unsafe_allow_html=True)

    # ---------- Summary ----------
    elif mode == "📝 Summarize Notes":
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("📝 Creating summary..."):
                summary_prompt = "Summarize this document into short bullet points for revision."
                result = qa_chain.invoke({"input": summary_prompt})
                summary = result

                st.markdown("### 📝 Summary")
                st.markdown(summary)

    # ---------- Flashcards ----------
    elif mode == "🧠 Generate Flashcards":
        num_cards = st.slider("Number of flashcards:", 3, 10, 5)
        if st.button("Generate Flashcards", use_container_width=True):
            with st.spinner("🧠 Creating flashcards..."):
                flashcard_prompt = f"""
Create {num_cards} flashcards from this document in Q&A format.
Format:
Q: question
A: answer
"""
                result = qa_chain.invoke({"input": flashcard_prompt})
                flashcard_text = result

                pattern = r'Q:(.*?)A:(.*?)(?=Q:|$)'
                matches = re.findall(pattern, flashcard_text, re.DOTALL)

                st.markdown("### 🧠 Flashcards")
                for i, (q, a) in enumerate(matches):
                    with st.expander(f"Question {i+1}: {q.strip()}"):
                        st.markdown(a.strip())

    os.unlink(temp_file_path)

else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color:#483AA0; border-radius: 10px; margin-top: 2rem;">
        <h2>Welcome to Groq Study Assistant!</h2>
        <p>Upload a PDF document to get started.</p>
    </div>
    """, unsafe_allow_html=True)
