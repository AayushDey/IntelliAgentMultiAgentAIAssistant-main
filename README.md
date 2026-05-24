# 🚀 Multi-Agent AI Assistant

## 📌 Problem Statement
Building an AI-powered **multi-agent assistant** that can:

- 📖 Explain educational concepts with supporting YouTube content
- 🔬 Perform research summarization (Semantic Scholar + video references)
- 📝 Assist with resume generation and provide improvement resources
- 📰 Fetch and summarize latest news with context videos
- 🎥 Analyze and answer questions from YouTube transcripts
- 📄 Provide Q&A over uploaded PDFs using **RAG (Retrieval-Augmented Generation)**

👉 This project showcases how **LangChain, HuggingFace, Tavily, and Streamlit** can be combined into a unified intelligent assistant.

---

## 🧩 Agent Interactions

### 🎓 Education Agent
- Takes user queries
- Returns explanations + relevant YouTube videos  

### 🔬 Research Agent
- Summarizes papers from Semantic Scholar
- Provides supporting YouTube content  

### 📝 Resume Agent
- Generates a LaTeX-based resume
- Suggests resume-building tutorials  

### 📰 News Agent
- Fetches top stories using Tavily or DuckDuckGo
- Summarizes + provides related YouTube videos  

### 🎥 Video Agent
- Extracts YouTube transcripts
- Allows Q&A on video content
- Suggests related resources  

### 📄 PDF Agent
- Supports PDF upload
- Performs Q&A with embeddings via **ChromaDB + HuggingFace**

---

## ⚙️ Technologies Used
- 🎨 **Streamlit** → Frontend UI  
- 🔗 **LangChain / LangGraph** → Multi-agent orchestration  
- 🤗 **HuggingFace Transformers** → Embeddings & NLP  
- 🗄 **ChromaDB** → Vector storage for PDFs  
- 🌐 **Tavily / DuckDuckGo** → News & web search  
- 🎥 **YouTube APIs (pytube, transcript)** → Video analysis  
- 📑 **Semantic Scholar API** → Research paper summaries  
- 🖋 **LaTeX** → Resume generation  

---

## ✅ Multi-agent architecture
👉 This project follows the **Supervisor (tool-calling)** system:  
- Each specialist is exposed as a tool  
- A central LLM supervisor decides which tool/agent to invoke  
- Execution flow: **Reason → Tool → Reason → Tool** (ReAct-style handoffs)  

### 🔹 ReAct Ability
✅ **Yes!**  
Your supervisor uses reasoning to decide the next tool → this is **ReAct (Reason + Act)** applied through **LangGraph tool-calling**.  

---

## ⚡ Setup & Run Instructions


```bash
1️⃣ Clone the repository
git clone https://github.com/your-username/multi-agent-assistant.git
cd multi-agent-assistant

2️⃣ Create a virtual environment
python -m venv langvenv
langvenv\Scripts\activate   # Windows
source langvenv/bin/activate   # Linux/Mac

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Setup environment variables

Add API keys in a .env file

For Streamlit Cloud, configure inside .streamlit/secrets.toml

5️⃣ Run the Streamlit app
streamlit run app.py

6️⃣ Open in Browser

🎉 Open your browser at: http://localhost:8501

