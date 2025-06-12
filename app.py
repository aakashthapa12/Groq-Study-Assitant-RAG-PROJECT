import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import tempfile
import os
import re

# Configure page
st.set_page_config(
    page_title="📚 Gemini Study Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
        max-width: 900px;
        margin: 0 auto;
    }
    .stApp {
        background-color: #0E2148;
    }
    .card {
        background-color: #7965C1;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .card-question {
        font-weight: bold;
        font-size: 1.1rem;
        color: #E3D095;
    }
    .card-answer {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #eee;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .header h1 {
        color: #E3D095;
        margin-bottom: 0.5rem;
    }
    .mode-selector {
        background-color: #483AA0;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .upload-section {
        background-color: #483AA0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>📚 Gemini Study Assistant</h1></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #E3D095; margin-bottom: 2rem;">Ask questions, summarize topics, or generate flashcards from your study material!</p>', unsafe_allow_html=True)

# Configure Gemini API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# File uploader with improved UI
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload a PDF (Lecture Notes, Textbook, etc.)", type=["pdf"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_file_path = tmp.name

    st.success("✅ PDF uploaded successfully!")
    
    # Process document with a nice progress indicator
    with st.spinner("🔍 Reading and processing document..."):
        progress_bar = st.progress(0)
        
        # Load document
        progress_bar.progress(20)
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        
        # Chunk text
        progress_bar.progress(40)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)
        
        # Embedding model
        progress_bar.progress(60)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )
        
        # Create vector store
        progress_bar.progress(80)
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
        
        # LLM model
        progress_bar.progress(90)
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )
        
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        progress_bar.progress(100)
    
    st.success("📄 Document is ready for queries!")
    
    # User options with improved UI
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    mode = st.selectbox(
        "Choose a Mode:",
        ["📖 Ask a Question", "📝 Summarize Notes", "🧠 Generate Flashcards"]
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if mode == "📖 Ask a Question":
        question = st.text_input("❓ Type your question about the content:")
        if question:
            if st.button("Ask", key="ask_button", use_container_width=True):
                with st.spinner("🧠 Thinking..."):
                    result = qa_chain.invoke({"query": question})
                    
                    # Display answer in a card
                    st.markdown("""
                    <div class="card">
                        <div class="card-question">Question: {}</div>
                        <div class="card-answer">{}</div>
                    </div>
                    """.format(question, result['result']), unsafe_allow_html=True)
    
    elif mode == "📝 Summarize Notes":
        if st.button("Generate Summary", key="summary_button", use_container_width=True):
            with st.spinner("📝 Creating summary..."):
                summary_prompt = "Summarize this document into short bullet points for revision. Format each bullet point with a '•' symbol."
                result = qa_chain.invoke({"query": summary_prompt})
                
                # Display summary in a nice format
                st.markdown("### 📝 Summary")
                st.markdown(result['result'])
    
    elif mode == "🧠 Generate Flashcards":
        num_cards = st.slider("Number of flashcards to generate:", min_value=3, max_value=10, value=5)
        
        if st.button("Generate Flashcards", key="flashcard_button", use_container_width=True):
            with st.spinner("🧠 Creating flashcards..."):
                flashcard_prompt = f"""Create {num_cards} flashcards from this document in Q&A format for revision.
                Format each flashcard as 'Q: [question]' on one line, followed by 'A: [answer]' on the next line.
                Make sure to separate each flashcard with two newlines.
                Make questions concise but specific, and answers comprehensive but not too long."""
                
                result = qa_chain.invoke({"query": flashcard_prompt})
                
                # Parse the flashcards
                flashcard_text = result['result']
                # Use regex to extract Q&A pairs
                pattern = r'Q:(.*?)A:(.*?)(?=Q:|$)'
                matches = re.findall(pattern, flashcard_text, re.DOTALL)
                
                if matches:
                    st.markdown("### 🧠 Flashcards")
                    st.markdown("Click on a card to reveal the answer")
                    
                    # Create interactive flashcards
                    for i, (question, answer) in enumerate(matches):
                        question = question.strip()
                        answer = answer.strip()
                        
                        # Create an expander that looks like a card
                        with st.expander(f"**Question {i+1}:** {question}"):
                            st.markdown(f"**Answer:** {answer}")
                else:
                    # Fallback if regex doesn't work
                    st.markdown("### 🧠 Flashcards")
                    st.markdown(result['result'])
    
    # Clean up
    os.unlink(temp_file_path)

else:
    # Display welcome message when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color:#483AA0; border-radius: 10px; margin-top: 2rem;">
        <img src="https://em-content.zobj.net/source/microsoft-teams/363/open-book_1f4d6.png" width="100">
        <h2>Welcome to Gemini Study Assistant!</h2>
        <p>Upload a PDF document to get started with your study session.</p>
        <p>You can ask questions about the content, generate summaries, or create flashcards to help with your studies.</p>
    </div>
    """, unsafe_allow_html=True)