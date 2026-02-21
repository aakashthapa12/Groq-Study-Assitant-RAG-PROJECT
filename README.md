# 📚 Groq Study Assistant - RAG Project

A powerful AI-powered study assistant that uses **RAG (Retrieval Augmented Generation)** to help you study smarter. Upload your PDFs and get instant answers, summaries, and flashcards powered by Groq's fast LLM inference.

## ✨ Features

- 📄 **PDF Upload**: Upload lecture notes, textbooks, or any study material
- ❓ **Ask Questions**: Get instant answers from your documents
- 📝 **Summarize Notes**: Generate bullet-point summaries for quick revision
- 🧠 **Generate Flashcards**: Create Q&A flashcards (3-10 cards) for studying

## 🚀 Tech Stack

- **Streamlit** - Web UI
- **LangChain** - RAG framework
- **FAISS** - Vector database for document embeddings
- **HuggingFace** - Text embeddings (all-MiniLM-L6-v2)
- **Groq** - Ultra-fast LLM inference (llama-3.3-70b-versatile)
- **PyPDF** - PDF processing

## 📦 Installation

### Prerequisites
- Python 3.8+
- Groq API Key ([Get it here](https://console.groq.com))

### Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:aakashthapa12/Groq-Study-Assitant-RAG-PROJECT.git
   cd Groq-Study-Assitant-RAG-PROJECT
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   
   Create `.streamlit/secrets.toml`:
   ```bash
   mkdir -p .streamlit
   ```
   
   Add your Groq API key to `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will open in your browser at `http://localhost:8501`

## 🔀 Branching Strategy

This project follows modern Git practices:

- **`main`** - Production-ready, stable code
- **`aakash-dev`** - Active development branch

### Workflow:
```bash
# Switch to dev branch
git checkout aakash-dev

# Make your changes
git add .
git commit -m "Your commit message"

# Push to dev branch
git push origin aakash-dev

# When stable, merge to main
git checkout main
git merge aakash-dev
git push origin main
```

## 📝 Usage

1. Upload a text-based PDF (not scanned images)
2. Wait for the document to be processed
3. Choose a mode:
   - **Ask a Question** - Type any question about the PDF content
   - **Summarize Notes** - Get bullet-point summaries
   - **Generate Flashcards** - Create study flashcards

## 🔒 Security

- **API keys are protected**: `.streamlit/secrets.toml` is gitignored
- Never commit your API keys to version control

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Aakash Thapa**
- GitHub: [@aakashthapa12](https://github.com/aakashthapa12)

## 🙏 Acknowledgments

- Built with [Groq](https://groq.com) for ultra-fast inference
- Powered by [LangChain](https://langchain.com) and [Streamlit](https://streamlit.io)
