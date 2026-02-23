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
from langchain_core.output_parsers import StrOutputParser

import tempfile
import os
import re
import yaml
import bcrypt
from yaml.loader import SafeLoader
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="📚 Groq Study Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Load / Save users ──────────────────────────────────────────────────────
# On Streamlit Cloud the filesystem is read-only, so we keep the credential
# store in st.secrets["credentials"] (a TOML table).  Locally we fall back
# to users.yaml so new sign-ups are persisted between runs.

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.yaml")
_IS_CLOUD   = not os.path.exists(USERS_FILE)   # True when deployed

def load_users() -> dict:
    """Return the auth config dict from secrets (cloud) or YAML (local)."""
    if _IS_CLOUD:
        # Build the structure streamlit-authenticator expects
        creds = dict(st.secrets.get("credentials", {}))
        # st.secrets gives AttrDict; convert usernames sub-table to plain dict
        usernames = {
            k: dict(v) for k, v in creds.get("usernames", {}).items()
        }
        return {
            "credentials": {"usernames": usernames},
            "cookie": {
                "expiry_days": int(st.secrets["cookie"]["expiry_days"]),
                "key":         st.secrets["cookie"]["key"],
                "name":        st.secrets["cookie"]["name"],
            },
        }
    with open(USERS_FILE) as f:
        return yaml.load(f, Loader=SafeLoader)

def save_users(config: dict) -> None:
    """Persist users.  On cloud we update st.secrets in-memory only
       (new users exist for the current session; permanent storage requires
       a database — see README for upgrading to SQLite/Supabase)."""
    if _IS_CLOUD:
        # Merge back into secrets so the running authenticator sees the change
        st.secrets["credentials"] = config["credentials"]
        return
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

    /* ── Document ready badge ── */
    .doc-ready-banner {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, #1a4731, #1e5c3a);
        border: 1px solid #2ecc71;
        border-radius: 10px;
        padding: 14px 20px;
        margin: 1rem 0;
    }
    .doc-ready-banner .doc-icon { font-size: 2rem; }
    .doc-ready-banner .doc-info { flex: 1; }
    .doc-ready-banner .doc-title {
        color: #2ecc71;
        font-weight: 700;
        font-size: 1rem;
        margin: 0;
    }
    .doc-ready-banner .doc-meta {
        color: #a0aec0;
        font-size: 0.82rem;
        margin: 2px 0 0 0;
    }
    .doc-ready-banner .doc-badge {
        background-color: #2ecc71;
        color: #0E2148;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        white-space: nowrap;
    }

    /* ── Welcome feature cards ── */
    .welcome-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 1.5rem 0;
    }
    .feature-card {
        background-color: #1C2F5E;
        border: 1px solid #2d3f70;
        border-radius: 12px;
        padding: 1.4rem 1rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover { transform: translateY(-4px); border-color: #7965C1; }
    .feature-card .feat-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .feature-card .feat-title { color: #E3D095; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem; }
    .feature-card .feat-desc  { color: #a0aec0; font-size: 0.8rem; line-height: 1.4; }

    /* ── Footer ── */
    .footer {
        margin-top: 3rem;
        padding: 1.2rem 0;
        border-top: 1px solid #2d3f70;
        text-align: center;
        color: #4a5568;
        font-size: 0.8rem;
    }
    .footer a { color: #7965C1; text-decoration: none; }
    .footer a:hover { color: #E3D095; }

    /* ── Chat history ── */
    .chat-bubble {
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        line-height: 1.6;
        position: relative;
    }
    .chat-bubble.user {
        background-color: #2d3f70;
        border-left: 4px solid #7965C1;
    }
    .chat-bubble.assistant {
        background-color: #1C2F5E;
        border-left: 4px solid #E3D095;
    }
    .chat-bubble .bubble-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .chat-bubble.user .bubble-label    { color: #7965C1; }
    .chat-bubble.assistant .bubble-label { color: #E3D095; }
    .chat-bubble .bubble-time {
        font-size: 0.7rem;
        color: #4a5568;
        position: absolute;
        top: 10px;
        right: 14px;
    }
    .chat-bubble .bubble-text { color: #e2e8f0; font-size: 0.92rem; }

    /* ── Sidebar stats chips ── */
    .stat-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #1C2F5E;
        border: 1px solid #2d3f70;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: #a0aec0;
        margin: 3px 2px;
    }
    .stat-chip span { color: #E3D095; font-weight: 700; }

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

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # list of {role, text, time}
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = None             # {name, pages, chunks, size_kb}
if "last_toasted_file" not in st.session_state:
    st.session_state.last_toasted_file = None    # track which file already got the toast
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None          # cached FAISS vectorstore
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None       # name of already-processed file

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👋 {st.session_state.get('name', 'User')}")
    st.markdown(f"<small style='color:#a0aec0;'>@{st.session_state.get('username', '')}</small>", unsafe_allow_html=True)
    st.markdown("---")

    # Session stats chips
    st.markdown("**📊 Session Stats**")
    doc_name = st.session_state.doc_meta["name"] if st.session_state.doc_meta else "None"
    doc_pages = st.session_state.doc_meta["pages"] if st.session_state.doc_meta else 0
    st.markdown(f"""
    <div class="stat-chip">📄 Doc: <span>{doc_name[:18] + "…" if len(doc_name) > 18 else doc_name}</span></div><br>
    <div class="stat-chip">📑 Pages: <span>{doc_pages}</span></div>
    <div class="stat-chip">❓ Questions: <span>{st.session_state.questions_asked}</span></div>
    <div class="stat-chip">💬 History: <span>{len(st.session_state.chat_history)}</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Clear history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.questions_asked = 0
        st.toast("Chat history cleared!", icon="🗑️")
        st.rerun()

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

    # ── Process ONLY if this is a newly uploaded file ─────────────────────
    if st.session_state.processed_file != uploaded_file.name:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_file_path = tmp.name

        file_size_kb = round(uploaded_file.size / 1024, 1)
        st.success(f"✅ **{uploaded_file.name}** uploaded successfully! ({file_size_kb} KB)")

        with st.spinner("🔍 Reading and processing document..."):
            progress_bar = st.progress(0, text="📖 Loading PDF pages…")

            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()
            pages = pages[:50]
            total_pages = len(pages)
            progress_bar.progress(40, text="✂️ Splitting into chunks…")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = text_splitter.split_documents(pages)
            if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
                st.error("🚫 The uploaded PDF appears to contain no readable text. It may be scanned or image-based. Please upload a text-based PDF.")
                os.unlink(temp_file_path)
                st.stop()

            progress_bar.progress(60, text="🧠 Building vector index…")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)

            progress_bar.progress(100, text="✅ Done!")

        # ── Delete temp file immediately — we no longer need it ──────────
        os.unlink(temp_file_path)

        # ── Cache everything in session state ────────────────────────────
        st.session_state.vectorstore     = vectorstore
        st.session_state.processed_file  = uploaded_file.name
        st.session_state.doc_meta        = {
            "name":     uploaded_file.name,
            "pages":    total_pages,
            "chunks":   len(docs),
            "size_kb":  file_size_kb,
        }
        # Clear chat history when a new document is loaded
        st.session_state.chat_history    = []
        st.session_state.questions_asked = 0

        # Toast fires only once for this file
        st.session_state.last_toasted_file = uploaded_file.name
        st.toast(f"📄 {uploaded_file.name} is ready!", icon="✅")

    # ── Rebuild chains from cached vectorstore (instant — no re-processing) ──
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    llm       = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful study assistant. Answer the question using ONLY the context below.

<context>
{context}
</context>

Question: {input}

Give a clear, concise, student-friendly answer.
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = (
        RunnableParallel(
            context=(lambda x: x["input"]) | retriever,
            input=RunnablePassthrough()
        )
        | document_chain
    )
    streaming_chain = (
        RunnableParallel(
            context=(lambda x: x["input"]) | retriever,
            input=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # ── Pull metadata from session ────────────────────────────────────────
    meta = st.session_state.doc_meta

    # ── Document ready banner ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="doc-ready-banner">
        <div class="doc-icon">📄</div>
        <div class="doc-info">
            <p class="doc-title">{meta['name']}</p>
            <p class="doc-meta">{meta['pages']} page(s) &nbsp;·&nbsp; {meta['chunks']} chunks indexed &nbsp;·&nbsp; {meta['size_kb']} KB</p>
        </div>
        <div class="doc-badge">✅ Ready</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    mode = st.selectbox(
        "Choose a Mode:",
        ["📖 Ask a Question", "📝 Summarize Notes", "🧠 Generate Flashcards"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Ask Question ──────────────────────────────────────────────────────
    if mode == "📖 Ask a Question":

        # Render existing chat history
        if st.session_state.chat_history:
            st.markdown("#### 💬 Chat History")
            for msg in st.session_state.chat_history:
                role_class = "user" if msg["role"] == "user" else "assistant"
                label = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
                st.markdown(f"""
                <div class="chat-bubble {role_class}">
                    <div class="bubble-label">{label}</div>
                    <div class="bubble-time">{msg['time']}</div>
                    <div class="bubble-text">{msg['text']}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

        question = st.text_input("❓ Type your question about the content:", key="question_input")
        if question and st.button("Ask ➤", use_container_width=True):
            now = datetime.now().strftime("%H:%M")

            # Append user message to history
            st.session_state.chat_history.append({"role": "user", "text": question, "time": now})
            st.session_state.questions_asked += 1

            # ── Streaming response ─────────────────────────────────────
            st.markdown("#### 🤖 Answer")
            answer_box = st.empty()
            full_answer = ""
            with st.spinner(""):
                for chunk in streaming_chain.stream({"input": question}):
                    full_answer += chunk
                    answer_box.markdown(full_answer + "▌")  # blinking cursor effect
            answer_box.markdown(full_answer)  # final answer without cursor

            # Copy-to-clipboard button using st.code trick
            with st.expander("📋 Copy raw answer"):
                st.code(full_answer, language=None)

            # Append assistant message to history
            st.session_state.chat_history.append({"role": "assistant", "text": full_answer, "time": now})
            st.rerun()

    # ── Summarize Notes ───────────────────────────────────────────────────
    elif mode == "📝 Summarize Notes":
        st.markdown("Generate a structured bullet-point summary of your document.")
        if st.button("📝 Generate Summary", use_container_width=True):
            st.markdown("#### 📝 Summary")
            summary_box = st.empty()
            full_summary = ""
            summary_prompt = "Summarize this document into clear, concise bullet points grouped by topic for revision."
            with st.spinner(""):
                for chunk in streaming_chain.stream({"input": summary_prompt}):
                    full_summary += chunk
                    summary_box.markdown(full_summary + "▌")
            summary_box.markdown(full_summary)
            with st.expander("� Copy raw summary"):
                st.code(full_summary, language=None)
            st.toast("Summary generated!", icon="📝")

    # ── Flashcards ────────────────────────────────────────────────────────
    elif mode == "🧠 Generate Flashcards":
        num_cards = st.slider("Number of flashcards:", 3, 10, 5)
        if st.button("🧠 Generate Flashcards", use_container_width=True):
            with st.spinner("🧠 Creating flashcards…"):
                flashcard_prompt = f"""Create exactly {num_cards} flashcards from this document in Q&A format.
Format STRICTLY as:
Q: question
A: answer
"""
                result = qa_chain.invoke({"input": flashcard_prompt})
                flashcard_text = result

            pattern = r'Q:(.*?)A:(.*?)(?=Q:|$)'
            matches = re.findall(pattern, flashcard_text, re.DOTALL)

            if matches:
                st.markdown(f"#### 🧠 {len(matches)} Flashcard(s) Generated")
                for i, (q, a) in enumerate(matches):
                    with st.expander(f"**Card {i+1}:** {q.strip()}"):
                        st.markdown(f"**Answer:** {a.strip()}")
                st.toast(f"{len(matches)} flashcards ready!", icon="🧠")
            else:
                st.warning("⚠️ Could not parse flashcards. Try again.")

else:
    # ── Welcome screen with feature highlight cards ───────────────────────
    st.markdown("""
    <div style="text-align:center; margin-top:1.5rem; margin-bottom:0.5rem;">
        <h2 style="color:#E3D095;">👋 Welcome to Groq Study Assistant!</h2>
        <p style="color:#a0aec0;">Upload a PDF below to unlock all features.</p>
    </div>

    <div class="welcome-grid">
        <div class="feature-card">
            <div class="feat-icon">❓</div>
            <div class="feat-title">Ask a Question</div>
            <div class="feat-desc">Ask anything about your document and get instant, accurate answers powered by Groq.</div>
        </div>
        <div class="feature-card">
            <div class="feat-icon">📝</div>
            <div class="feat-title">Summarize Notes</div>
            <div class="feat-desc">Generate clean bullet-point summaries of your lecture notes or textbook chapters.</div>
        </div>
        <div class="feature-card">
            <div class="feat-icon">🧠</div>
            <div class="feat-title">Generate Flashcards</div>
            <div class="feat-desc">Create 3–10 Q&A flashcards from your material — perfect for quick revision.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    📚 <strong>Groq Study Assistant</strong> &nbsp;·&nbsp; v1.0 &nbsp;·&nbsp;
    Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> &amp;
    <a href="https://groq.com" target="_blank">Groq</a> &amp;
    <a href="https://python.langchain.com" target="_blank">LangChain</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/aakashthapa12/Groq-Study-Assitant-RAG-PROJECT" target="_blank">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)
