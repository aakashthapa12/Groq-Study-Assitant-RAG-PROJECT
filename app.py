"""
Groq Study Assistant
====================
A Streamlit app where a student uploads a PDF and can then:
  1. Ask questions about it
  2. Get a bullet-point summary
  3. Generate 3-10 flashcards

How the RAG (Retrieval-Augmented Generation) pipeline works:
  1. Load     - read the text of each PDF page with pypdf.
  2. Split    - cut the text into chunks of up to 500 characters, with 100 characters of overlap.
  3. Embed    - turn every chunk into a vector (a list of 384 numbers that captures its meaning)
                using the all-MiniLM-L6-v2 model, which runs locally for free.
  4. Index    - store those vectors in a FAISS index so we can search the chunks by meaning.
  5. Retrieve - turn the question into a vector too, and ask FAISS for the 4 most similar chunks.
  6. Generate - send only those chunks plus the question to an LLM on Groq, and show its answer.
"""

import io
import os
import re

import faiss
import groq
import numpy as np
import streamlit as st
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# =============================================================================
# Settings - every important number lives here, so it is easy to find and explain
# =============================================================================

# Groq shut down llama-3.3-70b-versatile on 16 Aug 2026, which broke the old app.
# openai/gpt-oss-120b is the replacement Groq recommends. If Groq retires a model
# again, this is the only line that needs to change.
LLM_MODEL = "openai/gpt-oss-120b"

# Small, fast embedding model (384 numbers per chunk). It runs on our own server, so it is free.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_PAGES = 50       # only the first 50 pages are processed, to keep memory use and wait time low
CHUNK_SIZE = 500     # maximum characters in one chunk
CHUNK_OVERLAP = 100  # characters shared by neighbouring chunks, so a sentence cut in half is not lost
TOP_K = 4            # how many chunks we retrieve to answer one question

# Summaries and flashcards need the whole document, not just 4 chunks. But Groq limits how
# many tokens we can send per minute (8,000 for this model on our account). 8,000 characters
# is roughly 2,000 tokens, which leaves room for the instructions, the model's reasoning
# and its reply.
WHOLE_DOC_CHAR_BUDGET = 8000

NOT_FOUND_REPLY = "I couldn't find that in your document."

# One prompt is shared by all three modes. Only {task} changes between them.
# The rules are what keep answers grounded in the PDF: the model is told to use only
# the excerpts we give it and not to add outside knowledge.
STUDY_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful study assistant for students.

Rules:
- Use ONLY the document excerpts below. Do not add facts from your own knowledge.
- Write in clear, simple, student-friendly language.

Document excerpts:
{context}

Task: {task}"""
)


# =============================================================================
# Styling - colours for the page are set in .streamlit/config.toml (the theme).
# This CSS only adds the rounded purple panels and the answer card.
# =============================================================================

CUSTOM_CSS = """
<style>
    /* !important is needed because Streamlit's own heading style would otherwise win */
    .app-title    { text-align: center; color: #E3D095 !important; }
    .app-subtitle { text-align: center; color: #E3D095; margin-bottom: 1.5rem; }

    /* Streamlit gives every container created with key="xyz" the CSS class "st-key-xyz",
       which is how we style the panels around the uploader and the mode picker. */
    .st-key-upload_section, .st-key-mode_selector {
        background-color: #483AA0;
        padding: 20px;
        border-radius: 10px;
    }
    .st-key-answer_card {
        background-color: #7965C1;
        padding: 20px;
        border-radius: 10px;
    }
    /* Long formulas scroll sideways instead of being cut off */
    .katex-display { overflow-x: auto; overflow-y: hidden; }
    .welcome-box {
        text-align: center;
        padding: 2rem;
        background-color: #483AA0;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
"""


# =============================================================================
# The RAG pipeline
# =============================================================================

def get_setting(name):
    """Read a secret such as GROQ_API_KEY.

    Looks in Streamlit secrets first (Streamlit Cloud, or .streamlit/secrets.toml locally),
    then falls back to an environment variable. Returns None if it is set in neither place.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # st.secrets raises an error when there is no secrets.toml file at all.
        pass
    return os.environ.get(name)


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the embedding model once and reuse it for every user and every click.

    Streamlit re-runs this whole script after every click. Without caching, we would
    reload the model (and rebuild the index) every time a button is pressed.

    normalize_embeddings=True scales every vector to a length of 1. That way, the
    inner product of two vectors is their cosine similarity (see build_faiss_index).
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def read_pdf_pages(pdf_bytes):
    """Pipeline step 1 (Load): return one Document per page (up to MAX_PAGES) and the total page count.

    A Document is LangChain's container for some text plus metadata. We save the page
    number in the metadata so we can later show which page an answer came from.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))  # read straight from memory, no temporary file needed
    pages = []
    for page_index, page in enumerate(reader.pages[:MAX_PAGES]):
        text = page.extract_text() or ""  # a scanned page is only a picture, so it has no text
        # people count pages from 1, Python counts from 0
        pages.append(Document(page_content=text, metadata={"page": page_index + 1}))
    return pages, len(reader.pages)


def build_faiss_index(chunks):
    """Pipeline steps 3 + 4 (Embed + Index).

    Every chunk becomes a vector of 384 numbers. IndexFlatIP compares a question's vector
    with every chunk's vector using the inner (dot) product. Because all vectors have a
    length of 1, the inner product is the cosine similarity: close to 1 means "same meaning",
    close to 0 means "unrelated". "Flat" means FAISS checks every vector, which is exact
    and fast enough for the few thousand chunks a 50-page PDF can produce.
    """
    texts = [chunk.page_content for chunk in chunks]
    vectors = np.array(load_embedding_model().embed_documents(texts), dtype="float32")  # FAISS needs float32
    index = faiss.IndexFlatIP(vectors.shape[1])  # vectors.shape[1] is 384, the length of each vector
    index.add(vectors)
    return index


@st.cache_resource(show_spinner=False, max_entries=5)
def process_pdf(pdf_bytes):
    """Pipeline steps 1-4: load -> split -> embed -> index.

    The result is cached using the file's bytes as the key, so each PDF is processed
    only once instead of on every click. max_entries=5 keeps at most 5 PDFs in memory,
    so a small free server does not run out of RAM.

    Returns (index, chunks, total_pages).
    index is None when the PDF has no readable text (for example, a scanned PDF).
    """
    pages, total_pages = read_pdf_pages(pdf_bytes)

    # Step 2 (Split): RecursiveCharacterTextSplitter tries to cut at paragraph breaks first,
    # then line breaks, then spaces, so chunks usually end at a natural boundary.
    # Each chunk keeps the page number of the page it came from.
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [chunk for chunk in splitter.split_documents(pages) if chunk.page_content.strip()]

    # No text at all (e.g. a scanned PDF), so there is nothing to index.
    if not chunks:
        return None, [], total_pages

    return build_faiss_index(chunks), chunks, total_pages


def retrieve(index, chunks, question):
    """Pipeline step 5 (Retrieve): return the TOP_K chunks most similar in meaning to the question."""
    question_vector = np.array([load_embedding_model().embed_query(question)], dtype="float32")
    k = min(TOP_K, len(chunks))  # a very short PDF may have fewer than TOP_K chunks
    _similarities, positions = index.search(question_vector, k)
    # FAISS returns the positions of the best-matching vectors, best first.
    # Chunk number i was added to the index as vector number i, so we look the chunks up by position.
    return [chunks[position] for position in positions[0]]


def create_llm(api_key):
    """Create the Groq chat model."""
    return ChatGroq(
        api_key=api_key,
        model=LLM_MODEL,
        temperature=0.2,         # low temperature = sticks to the excerpts, less creative guessing
        reasoning_effort="low",  # gpt-oss "thinks" before answering; "low" keeps replies fast and cheap
    )


def format_chunks(chunks):
    """Join chunks into one block of text, labelling each one with its page number."""
    return "\n\n".join(f"[Page {chunk.metadata['page']}]\n{chunk.page_content}" for chunk in chunks)


def spread_out_chunks(chunks, char_budget):
    """Pick chunks evenly from the start, middle and end of the document.

    Used for summaries and flashcards. These need the whole document, not similarity search:
    searching for "summarize" would return the 4 chunks that best match that word, which
    could all come from one page. So we use every chunk if they fit in the budget, otherwise
    an evenly spaced selection.
    """
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    if total_chars <= char_budget:
        return chunks  # the whole document fits, so use all of it

    # Work out how many chunks fit in the budget. Then split the list of chunks into that
    # many equal sections and take the chunk in the middle of each section, so the
    # beginning, middle and end of the document are all included.
    average_chunk_size = total_chars / len(chunks)
    how_many = max(1, int(char_budget / average_chunk_size))
    section_size = len(chunks) / how_many
    return [chunks[int((i + 0.5) * section_size)] for i in range(how_many)]


def fix_math_for_streamlit(text):
    """Make formulas in the model's reply display properly.

    Models usually write maths in LaTeX as \\( x^2 \\) (inline) or \\[ x^2 \\] (on its own line),
    but Streamlit only renders maths written as $x^2$ (inline) or as a block with $$ on its
    own line before and after the formula. Without this, students would see raw code like
    "\\text{head}_i" instead of a formula.
    """
    # Block formulas: \[ ... \]  ->  $$ on its own line, the formula, $$ on its own line
    text = re.sub(r"\\\[(.+?)\\\]", lambda match: f"\n$$\n{match.group(1).strip()}\n$$\n", text, flags=re.DOTALL)
    # Inline formulas: \( ... \)  ->  $ ... $
    text = re.sub(r"\\\((.+?)\\\)", lambda match: f"${match.group(1).strip()}$", text, flags=re.DOTALL)
    return text


def ask_groq(llm, chunks, task):
    """Pipeline step 6 (Generate): fill the prompt, send it to Groq and return the reply text.

    Returns None (after showing a friendly message) if Groq returns an error, so one failed
    request shows a message instead of crashing the whole app.
    """
    # LangChain "pipe" syntax: the filled-in prompt goes to the model, and the
    # model's reply goes to StrOutputParser, which turns it into a plain string.
    chain = STUDY_PROMPT | llm | StrOutputParser()
    try:
        reply = chain.invoke({"context": format_chunks(chunks), "task": task})
        return fix_math_for_streamlit(reply)
    except groq.RateLimitError:
        st.error("⏳ Groq's rate limit was reached. Please wait a minute and try again.")
    except groq.AuthenticationError:
        st.error("🔑 Groq rejected the API key. Check GROQ_API_KEY in your Streamlit secrets.")
    except groq.APIError as error:  # any other Groq problem: network error, retired model, etc.
        st.error(f"❌ Groq returned an error: {error}")
    return None


def parse_flashcards(text):
    """Turn the model's "Q: ... / A: ..." lines into a list of (question, answer) pairs."""
    cards = []
    question = None
    for line in text.splitlines():
        # Models sometimes add bold (**Q:**), bullets (- Q:) or numbers (1. Q:). Remove them.
        line = line.replace("**", "").strip().lstrip("-*•0123456789.) ")
        if line[:2].upper() == "Q:":
            question = line[2:].strip()
        elif line[:2].upper() == "A:" and question:
            cards.append((question, line[2:].strip()))
            question = None
    return cards


# =============================================================================
# The three modes
# =============================================================================

def ask_question_mode(index, chunks, llm):
    """Classic RAG: retrieve the most relevant chunks, then answer from them only."""
    # A form lets the student press Enter to submit, and the app only reruns when they do.
    with st.form("question_form"):
        question = st.text_input("❓ Type your question about the content:")
        submitted = st.form_submit_button("Ask", width="stretch")

    if not submitted or not question.strip():
        return

    with st.spinner("🧠 Thinking..."):
        # Step 5 (Retrieve): the TOP_K chunks closest in meaning to the question.
        relevant_chunks = retrieve(index, chunks, question)
        # Step 6 (Generate): answer using only those chunks.
        task = (
            f"Answer this question: {question}\n"
            f'If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND_REPLY}"'
        )
        answer = ask_groq(llm, relevant_chunks, task)

    if answer is None:
        return

    with st.container(key="answer_card"):
        st.markdown(f"**Question:** {question}")
        st.markdown(answer)

    # Show where the answer came from, so the student can check it against the PDF.
    with st.expander("📄 Sources used for this answer"):
        for chunk in relevant_chunks:
            st.markdown(f"**Page {chunk.metadata['page']}**")
            st.caption(chunk.page_content)


def summary_mode(chunks, llm):
    """Summarize chunks taken from across the whole document."""
    if not st.button("Generate Summary", width="stretch"):
        return

    with st.spinner("📝 Creating summary..."):
        selected_chunks = spread_out_chunks(chunks, WHOLE_DOC_CHAR_BUDGET)
        summary = ask_groq(
            llm,
            selected_chunks,
            "Summarize these notes into short bullet points for revision. "
            "Cover the main topics in the order they appear.",
        )

    if summary is None:
        return

    st.markdown("### 📝 Summary")
    st.markdown(summary)
    st.caption(f"Based on {len(selected_chunks)} of {len(chunks)} chunks, taken from across the whole document.")


def flashcard_mode(chunks, llm):
    """Generate 3-10 question/answer flashcards from across the whole document."""
    num_cards = st.slider("Number of flashcards:", min_value=3, max_value=10, value=5)
    if not st.button("Generate Flashcards", width="stretch"):
        return

    with st.spinner("🧠 Creating flashcards..."):
        selected_chunks = spread_out_chunks(chunks, WHOLE_DOC_CHAR_BUDGET)
        task = (
            f"Create exactly {num_cards} flashcards that test the key ideas in these notes.\n"
            "Write each flashcard as two lines, exactly like this, with nothing else:\n"
            "Q: <question>\n"
            "A: <short answer>"
        )
        reply = ask_groq(llm, selected_chunks, task)

    if reply is None:
        return

    # Keep at most num_cards, in case the model writes more than we asked for.
    cards = parse_flashcards(reply)[:num_cards]

    st.markdown("### 🧠 Flashcards")
    if not cards:
        # The model ignored the Q:/A: format. Show its reply as plain text instead of nothing.
        st.markdown(reply)
        return

    for number, (question, answer) in enumerate(cards, start=1):
        with st.expander(f"Question {number}: {question}"):
            st.markdown(answer)


# =============================================================================
# Main page
# =============================================================================

def main():
    st.set_page_config(page_title="Groq Study Assistant", page_icon="📚", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("<h1 class='app-title'>📚 Groq Study Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='app-subtitle'>Ask questions, summarize topics, or generate flashcards "
        "from your study material!</p>",
        unsafe_allow_html=True,
    )

    api_key = get_setting("GROQ_API_KEY")
    if not api_key:
        st.error(
            "🚫 Groq API key not found. Add GROQ_API_KEY to .streamlit/secrets.toml "
            "(when running locally) or to the app's Secrets on Streamlit Cloud."
        )
        st.stop()

    with st.container(key="upload_section"):
        uploaded_file = st.file_uploader(
            "📤 Upload a text-based PDF (lecture notes, textbook, etc.)", type=["pdf"]
        )

    if uploaded_file is None:
        st.markdown(
            "<div class='welcome-box'><h2>Welcome to Groq Study Assistant!</h2>"
            "<p>Upload a PDF document to get started.</p></div>",
            unsafe_allow_html=True,
        )
        return

    with st.spinner("🔍 Reading and indexing your document..."):
        try:
            index, chunks, total_pages = process_pdf(uploaded_file.getvalue())
        except Exception as error:  # e.g. a damaged or password-protected PDF
            st.error(f"🚫 Could not read this PDF: {error}")
            return

    if index is None:
        st.error(
            "🚫 The uploaded PDF has no readable text. It may be scanned or image-based. "
            "Please upload a text-based PDF."
        )
        return

    st.success(f"✅ Document ready: {len(chunks)} chunks from {min(total_pages, MAX_PAGES)} page(s).")
    if total_pages > MAX_PAGES:
        st.info(f"ℹ️ This PDF has {total_pages} pages. Only the first {MAX_PAGES} pages are used.")

    llm = create_llm(api_key)

    with st.container(key="mode_selector"):
        mode = st.selectbox(
            "Choose a mode:",
            ["📖 Ask a Question", "📝 Summarize Notes", "🧠 Generate Flashcards"],
        )

    if mode == "📖 Ask a Question":
        ask_question_mode(index, chunks, llm)
    elif mode == "📝 Summarize Notes":
        summary_mode(chunks, llm)
    else:
        flashcard_mode(chunks, llm)


# `streamlit run app.py` runs this file as "__main__", so the app starts.
# Importing the file (for example from a test) does not start the app.
if __name__ == "__main__":
    main()
