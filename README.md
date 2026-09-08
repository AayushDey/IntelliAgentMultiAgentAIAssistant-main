# 🧠 IntelliAgent — Multi-Agent AI Assistant

An intelligent, multi-agent AI assistant built with **Streamlit**, **LangChain**, and **Groq**. IntelliAgent features a supervisor-directed architecture that automatically analyzes query intent and routes questions to specialized AI agents—or allows manual agent override—with modern dark glassmorphic UI, local PDF RAG indexing, and YouTube transcript Q&A.

---

## ✨ Features & Specialized Agents

| Agent | Capability | Key Backing Tool |
| :--- | :--- | :--- |
| 💻 **Code Assistant** | Code generation, line-by-line breakdown, debugging, syntax repair, and code reviews across any programming language. | `code_assistant_tool` |
| 🎤 **Interview Prep** | Role-specific technical & behavioral interview questions, STAR method model answers, and coaching. | `interview_prep_tool` |
| 🏥 **Health Advisor** | Plain-language physiological explanations, evidence-based wellness guidance, and medical disclaimers. | `health_advisor_tool` |
| 🎓 **Education Agent** | Structured conceptual breakdowns, principles, practical examples, and study techniques. | `topic_explanation` |
| 🔬 **Research Agent** | Academic paper search, abstract summarization, and key findings synthesis via Semantic Scholar. | `semantic_scholar_research` |
| 📰 **News Agent** | Real-time news search and executive briefings using Tavily and DuckDuckGo. | `tavily_search` / `duckduckgo_search` |
| 📄 **Resume Agent** | Generation of ATS-compliant LaTeX resume templates and actionable tailoring advice. | `generate_resume` |
| 🎥 **Video Analysis** | Direct question answering on YouTube videos using subtitles and transcripts (no API key needed). | `youtube_qa` (`youtube-transcript-api`) |
| 📑 **PDF Q&A with RAG** | Document indexing and conversational retrieval using ChromaDB and local sentence transformers. | `PDFRAGAgent` (`all-MiniLM-L6-v2`) |

---

## 🎨 User Interface Highlights
- **High-Contrast Dark Theme**: Custom CSS palette (Cyber Obsidian `#0A0F1D` and Electric Indigo/Cyan `#38BDF8`) with WCAG AAA readability.
- **Native Chat Bubbles**: Seamless `st.chat_message` rendering with distinct specialist badges, avatar icons, and Pygments syntax highlighting.
- **Agent Mode Selector**: Let the auto-detect supervisor choose the specialist or force-route queries directly via the sidebar.
- **Follow-Up Suggestions**: Dynamic, context-aware prompt chips below assistant responses to accelerate exploration.
- **Session Stats & Quick Reset**: Track messages and active specialist count with one-click conversation clearing.

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM Orchestration**: LangChain & LangChain Community
- **Inference Engine**: [Groq Cloud](https://console.groq.com/) (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`)
- **Embeddings & Vector Store**: ChromaDB + Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2` running locally)
- **Web & Academic Search**: Tavily AI, DuckDuckGo Search, Semantic Scholar API
- **Video Processing**: `youtube-transcript-api`

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/AayushDey/IntelliAgentMultiAgentAIAssistant-main.git
cd IntelliAgentMultiAgentAIAssistant-main
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```
Inside `.env`:
```env
# Required: Free Groq API Key (https://console.groq.com/)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Tavily Search Key for real-time web search (https://tavily.com/)
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔒 Security & Privacy
- Sensitive files such as `.env` and local secrets are excluded from source control via `.gitignore`.
- Local document embeddings run offline on your machine without external API token requirements.
