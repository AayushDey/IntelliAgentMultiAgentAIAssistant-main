import streamlit as st
import re
import tempfile
import requests
from typing import List, Dict
import os
import time
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
try:
    from langchain_tavily import TavilySearch as TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import urlparse, parse_qs

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IntelliAgent — Multi-Agent AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT & API KEYS
# ═══════════════════════════════════════════════════════════════
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip().strip('"').strip("'")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


def get_tavily_tool():
    if TAVILY_API_KEY and not TAVILY_API_KEY.startswith("your_"):
        try:
            return TavilySearchResults(max_results=5)
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════
# HIGH-CONTRAST THEME CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global Typography ── */
html, body, [data-testid="stAppViewContainer"], .stApp,
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label,
.stApp input, .stApp button, .stApp textarea, .stApp div, .stApp span {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre, kbd, samp {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── High-Contrast Dark Background ── */
.stApp {
    background-color: #0A0F1D;
    background-image: radial-gradient(circle at 50% 0%, rgba(30, 41, 59, 0.45) 0%, #0A0F1D 70%);
    background-attachment: fixed;
    color: #F8FAFC;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0B1120 !important;
    border-right: 1px solid #1E293B !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #94A3B8 !important;
}

/* ── Hero Title ── */
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94A3B8;
    font-size: 1.02rem;
    font-weight: 400;
    margin-top: 6px;
}

/* ── Status Pills ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.status-online {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.status-offline {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

/* ── Chat Messages Styling ── */
div[data-testid="stChatMessage"] {
    background-color: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin: 10px 0 !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background-color: #162032 !important;
    border: 1px solid #2B3A55 !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background-color: #0E1626 !important;
    border: 1px solid #1E2C45 !important;
    border-left: 3px solid #6366F1 !important;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li {
    color: #F1F5F9 !important;
    font-size: 0.96rem !important;
    line-height: 1.65 !important;
}
div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4 {
    color: #FFFFFF !important;
}

/* ── Agent Badges ── */
.agent-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.badge-education { background: rgba(99,102,241,0.2); color: #A5B4FC; border: 1px solid rgba(99,102,241,0.4); }
.badge-code { background: rgba(16,185,129,0.2); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.4); }
.badge-research { background: rgba(236,72,153,0.2); color: #F472B6; border: 1px solid rgba(236,72,153,0.4); }
.badge-interview { background: rgba(249,115,22,0.2); color: #FDBA74; border: 1px solid rgba(249,115,22,0.4); }
.badge-health { background: rgba(20,184,166,0.2); color: #5EEAD4; border: 1px solid rgba(20,184,166,0.4); }
.badge-news { background: rgba(14,165,233,0.2); color: #7DD3FC; border: 1px solid rgba(14,165,233,0.4); }
.badge-resume { background: rgba(234,179,8,0.2); color: #FDE047; border: 1px solid rgba(234,179,8,0.4); }
.badge-video { background: rgba(239,68,68,0.2); color: #FCA5A5; border: 1px solid rgba(239,68,68,0.4); }
.badge-pdf { background: rgba(168,85,247,0.2); color: #D8B4FE; border: 1px solid rgba(168,85,247,0.4); }
.badge-general { background: rgba(100,116,139,0.2); color: #CBD5E1; border: 1px solid rgba(100,116,139,0.4); }

/* ── Primary Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(37,99,235,0.5) !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Text Inputs & Textareas ── */
.stTextInput input, .stTextArea textarea {
    background-color: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #F8FAFC !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.25s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.2) !important;
    outline: none !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: #64748B !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0F172A;
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
    border: 1px solid #1E293B;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #F1F5F9 !important;
    background: rgba(255,255,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.12) !important;
    color: #38BDF8 !important;
    border-bottom: 2px solid #38BDF8 !important;
    font-weight: 600 !important;
}

/* ── File Uploader ── */
.stFileUploader {
    background: #0F172A !important;
    border: 2px dashed #334155 !important;
    border-radius: 14px !important;
    transition: all 0.3s ease !important;
}
.stFileUploader:hover {
    border-color: #38BDF8 !important;
    background: rgba(56,189,248,0.04) !important;
}

/* ── Headings & Markdown Text ── */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
.stMarkdown p, .stMarkdown li {
    color: #E2E8F0 !important;
}
.stMarkdown a {
    color: #38BDF8 !important;
    text-decoration: none;
    font-weight: 500;
}
.stMarkdown a:hover {
    text-decoration: underline;
}

/* ── Code Formatting ── */
:not(pre) > code {
    background: #1E293B !important;
    color: #38BDF8 !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    padding: 2px 7px !important;
    font-size: 0.88em !important;
}
pre {
    background: #090D16 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
}

/* ── Alerts & Expanders ── */
.stAlert, div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.streamlit-expanderHeader {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
    background: #111827 !important;
    border-radius: 10px !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero-title { font-size: 1.8rem; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
def initialize_session_state():
    defaults = {
        'chat_history': [],
        'current_agent': 'general',
        'processing': False,
        'model_name': 'Not Connected',
        'force_agent': 'auto',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_session_state()


# ═══════════════════════════════════════════════════════════════
# MODEL INITIALIZATION (Groq)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def initialize_model(groq_key: str):
    """Initialize Groq LLM model. Returns (model, model_display_name)."""
    if groq_key:
        models_to_try = [
            ("openai/gpt-oss-120b", "Groq GPT-OSS 120B"),
            ("llama-3.3-70b-versatile", "Groq Llama 3.3 70B"),
            ("llama-3.1-8b-instant", "Groq Llama 3.1 8B"),
        ]
        for model_id, display_name in models_to_try:
            try:
                model = ChatGroq(model=model_id, groq_api_key=groq_key)
                return model, display_name
            except Exception:
                continue
    return None, "Not Connected"

hf_model, _model_name = initialize_model(GROQ_API_KEY)
st.session_state.model_name = _model_name


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def extract_video_id(url: str):
    """Extract video ID from YouTube URL."""
    try:
        parsed = urlparse(url)
        if parsed.hostname in ('www.youtube.com', 'youtube.com'):
            return parse_qs(parsed.query).get('v', [None])[0]
        elif parsed.hostname == 'youtu.be':
            return parsed.path.lstrip('/')
    except Exception:
        pass
    match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
    return match.group(1) if match else None

def is_valid_youtube_url(url: str):
    if not url:
        return False
    patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=',
        r'https?://youtu\.be/',
        r'https?://(?:www\.)?youtube\.com/embed/'
    ]
    return any(re.match(p, url) for p in patterns)

def get_video_thumbnail(video_id: str):
    return f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"


# ═══════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════
@tool
def topic_explanation(query: str) -> str:
    """Return a comprehensive conceptual explanation of the topic."""
    if not hf_model:
        return "AI model is not available. Please check your GROQ_API_KEY."
    try:
        prompt = f"""Provide a detailed, well-structured explanation of '{query}' covering:
1. Core concept and clear definition
2. Key components, principles, or mechanisms
3. Real-world applications and examples
4. Common pitfalls or misconceptions

Use clean markdown formatting. Be beginner-friendly yet technically thorough."""
        return hf_model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


@tool
def generate_resume(job_description: str, candidate_info: str = "") -> str:
    """Generate a professional LaTeX resume optimized for ATS systems."""
    if not hf_model:
        return "AI model is not available. Please check your GROQ_API_KEY."
    try:
        prompt = f"""Create a modern, ATS-friendly LaTeX resume using clean formatting.

Requirements:
- Use standard LaTeX packages (article, geometry, enumitem, hyperref)
- Include proper sections: Contact Information, Professional Summary, Core Competencies, Experience, Education, Technical Projects
- Maximize keyword alignment with the job description
- Use bullet points starting with strong action verbs

Job Description: {job_description}
Candidate Info: {candidate_info if candidate_info else "Entry-level to mid-level candidate"}

Return only valid LaTeX code inside a code block."""
        return hf_model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        return f"Error generating resume: {str(e)}"


try:
    ss = SemanticScholarAPIWrapper(top_k_results=5, load_max_docs=5)
except Exception:
    ss = None

@tool
def semantic_scholar_research(query: str) -> List[Dict]:
    """Fetch and summarize top research papers from Semantic Scholar."""
    if not ss:
        return [{"error": "Semantic Scholar search is currently unavailable."}]
    try:
        raw = ss.run(query)
        papers = raw.split("\n\n")
        result = []
        for p in papers[:3]:
            if "abstract:" in p.lower():
                if hf_model:
                    summary = hf_model.invoke([
                        HumanMessage(content=f"Summarize this research paper in 3 concise bullet points focusing on methodology and key findings:\n{p}")
                    ]).content
                    result.append({"raw_info": p, "summary": summary})
                else:
                    result.append({"raw_info": p, "summary": "AI model unavailable"})
        return result
    except Exception as e:
        return [{"error": f"Error fetching research papers: {str(e)}"}]


# YouTube Transcript Q&A (No API Key Required)
qa_prompt = PromptTemplate(
    template="""You are a helpful video assistant. Answer the user's question using ONLY the provided transcript context.
If the information is not in the transcript, say "I cannot find information about that in the video transcript."

Transcript:
{context}

Question:
{question}

Answer:""",
    input_variables=['context', 'question']
)

def get_transcript(video_id: str):
    """Fetch transcript safely without requiring a YouTube API key."""
    if not video_id:
        return None, "Invalid video ID provided."
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
    except NoTranscriptFound:
        try:
            transcript_list = YouTubeTranscriptApi().list_transcripts(video_id)
            transcript = transcript_list.find_generated_transcript(['en', 'hi'])
            transcript_list = transcript.fetch()
        except Exception:
            return None, "No transcript available for this video."
    except (TranscriptsDisabled, VideoUnavailable) as e:
        return None, f"Video transcript unavailable: {str(e)}"
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"

    try:
        texts = [s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "") for s in transcript_list]
        transcript = " ".join(texts)
        if len(transcript.strip()) < 10:
            return None, "Retrieved transcript is empty."
        return transcript, None
    except Exception as e:
        return None, f"Error processing transcript: {str(e)}"

@tool
def youtube_qa(video_url: str, question: str) -> str:
    """Given a YouTube URL and question, return answer based on subtitles/transcript."""
    if not hf_model:
        return "AI model is not available."
    try:
        if not is_valid_youtube_url(video_url):
            return "Invalid YouTube URL format. Please provide a valid YouTube link."
        video_id = extract_video_id(video_url)
        if not video_id:
            return "Could not extract video ID from the provided URL."
        transcript, error = get_transcript(video_id)
        if error:
            return f"Transcript Error: {error}"
        if len(transcript) > 8000:
            transcript = transcript[:8000] + "... (truncated)"
        rag_runnable = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | qa_prompt
            | hf_model
        )
        answer = rag_runnable.invoke({"context": transcript, "question": question})
        return answer.content if hasattr(answer, 'content') else str(answer)
    except Exception as e:
        return f"Error processing video: {str(e)}"


@tool
def code_assistant_tool(query: str) -> str:
    """Generate, explain, debug, or review code based on the user's request."""
    if not hf_model:
        return "AI model is not available."
    try:
        prompt = f"""You are an expert senior software engineer. Help the user with their coding request.

User Request: {query}

Instructions:
- Write clean, production-ready, well-commented code with error handling.
- If debugging, pinpoint the bug, explain why it happens, and provide the corrected code.
- If explaining, provide a structured breakdown of logic and time/space complexities.
- Always use markdown code blocks with the appropriate language identifier.
- Include example usage or test cases."""
        return hf_model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        return f"Error with code assistant: {str(e)}"


@tool
def interview_prep_tool(query: str) -> str:
    """Generate role-specific interview questions with model answers and tips."""
    if not hf_model:
        return "AI model is not available."
    try:
        prompt = f"""You are an elite executive career coach and technical hiring manager.

User Request: {query}

Provide a comprehensive interview prep package:
1. **Top 3-5 Likely Questions** (mix of technical, architectural, and behavioral).
2. **Model Answers (STAR Method)**: Situation, Task, Action, Result for behavioral questions, or step-by-step problem breakdown for technical questions.
3. **Common Pitfalls**: What bad candidates say vs. what top 1% candidates say.
4. **Questions to Ask the Interviewer**: Strategic questions that impress hiring committees.

Format clearly with markdown."""
        return hf_model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        return f"Error with interview prep: {str(e)}"


@tool
def health_advisor_tool(query: str) -> str:
    """Provide educational health and wellness information with medical disclaimers."""
    if not hf_model:
        return "AI model is not available."
    try:
        prompt = f"""You are an empathetic, evidence-based health and wellness educator.

User Question: {query}

Instructions:
- Explain the physiology or medical concepts in accessible, clear language.
- Provide evidence-based wellness insights, lifestyle suggestions, or nutrition advice.
- Highlight when a person should urgently consult a qualified doctor or emergency services.
- Keep the tone compassionate, objective, and supportive.

DISCLAIMER: Always include a prominent reminder that this is for educational purposes only and not medical advice."""
        return hf_model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        return f"Error with health advisor: {str(e)}"


# Search tools initialization
try:
    duck_tool = DuckDuckGoSearchRun()
except Exception:
    duck_tool = None

tavily_tool = get_tavily_tool()


# ═══════════════════════════════════════════════════════════════
# PDF RAG AGENT (Local Embeddings — No API Key Required)
# ═══════════════════════════════════════════════════════════════
class PDFRAGAgent:
    def __init__(self):
        self.retriever = None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_embedding_model():
        """Load sentence-transformers embeddings locally (no API key required)."""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_pdf(self, pdf_file):
        """Load and index a single PDF."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=40)
            split_docs = text_splitter.split_documents(documents)
            embedding_model = self.get_embedding_model()
            db = Chroma.from_documents(
                split_docs,
                embedding=embedding_model,
                collection_name="student_pdf"
            )
            self.retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            return True
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def answer_question(self, query: str, model):
        """Answer questions based on the indexed PDF."""
        if not self.retriever:
            return "⚠️ Please upload a PDF first."
        try:
            docs = self.retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])
            if not context.strip():
                return "❌ No relevant content found in the PDF for your question."
            prompt = ChatPromptTemplate.from_template(
                "You are an expert document assistant.\n"
                "Answer the question based only on the provided document context.\n"
                "If the information is not clearly present in the context, say so.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
            chain = prompt | model
            response = chain.invoke({"context": context, "question": query})
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"❌ Error processing question: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# AGENTS
# ═══════════════════════════════════════════════════════════════
AGENT_META = {
    "education":       {"icon": "🎓", "label": "Education",       "badge": "badge-education"},
    "code":            {"icon": "💻", "label": "Code Assistant",  "badge": "badge-code"},
    "research":        {"icon": "🔬", "label": "Research",        "badge": "badge-research"},
    "interview":       {"icon": "🎤", "label": "Interview Prep",  "badge": "badge-interview"},
    "health":          {"icon": "🏥", "label": "Health Advisor",  "badge": "badge-health"},
    "news":            {"icon": "📰", "label": "News",            "badge": "badge-news"},
    "resume":          {"icon": "📄", "label": "Resume",          "badge": "badge-resume"},
    "video_analysis":  {"icon": "🎥", "label": "Video Analysis",  "badge": "badge-video"},
    "pdf":             {"icon": "📑", "label": "PDF Q&A",         "badge": "badge-pdf"},
    "general":         {"icon": "🤖", "label": "General",         "badge": "badge-general"},
    "error":           {"icon": "❌", "label": "Error",           "badge": "badge-general"},
}


class EducationAgent:
    def __init__(self, model):
        self.model = model

    def process(self, query: str) -> str:
        try:
            explanation = topic_explanation.invoke({"query": query})
            return f"""## 🎓 Educational Breakdown: {query}

{explanation}

### 💡 Study & Application Tips
- Focus on the core mechanism before diving into edge cases.
- Practice by explaining this concept to someone else in simple terms.
- Try asking for code examples or real-world case studies if you want to go deeper!"""
        except Exception as e:
            return f"❌ Educational query error: {str(e)}"


class ResearchAgent:
    def __init__(self, model):
        self.model = model

    def process(self, query: str) -> str:
        try:
            papers = semantic_scholar_research.invoke({"query": query})
            response = f"## 🔬 Academic Research: {query}\n\n"
            if papers and isinstance(papers, list) and "error" not in papers[0]:
                response += "### 📄 Key Papers & Abstracts\n\n"
                for i, paper in enumerate(papers[:3], 1):
                    if isinstance(paper, dict):
                        summary = paper.get("summary", "")
                        raw = paper.get("raw_info", "")
                        if summary:
                            response += f"**{i}. Key Findings:**\n{summary}\n\n"
                        elif raw:
                            response += f"**{i}. Paper Details:**\n{raw[:300]}...\n\n"
            else:
                # Fallback to model synthesis
                prompt = f"Provide a research-level synthesis of academic consensus on: '{query}'. Include key theories, milestones, and open questions."
                response += self.model.invoke([HumanMessage(content=prompt)]).content

            return response
        except Exception as e:
            return f"❌ Research query error: {str(e)}"


class ResumeAgent:
    def __init__(self, model):
        self.model = model

    def process(self, query: str, job_desc: str = "", candidate_info: str = "") -> str:
        try:
            resume_latex = generate_resume.invoke({
                "job_description": job_desc or query,
                "candidate_info": candidate_info
            })
            return f"""## 📄 ATS-Optimized Resume

{resume_latex}

### 📋 Next Steps
1. Copy the LaTeX code above.
2. Open [Overleaf](https://overleaf.com) and create a blank project.
3. Paste and compile into a crisp, ATS-compliant PDF!
4. Customize the achievements with quantifiable metrics (e.g., *'boosted latency by 35%'*)."""
        except Exception as e:
            return f"❌ Resume generation error: {str(e)}"


class NewsAgent:
    def __init__(self, model):
        self.model = model

    def process(self, query: str) -> str:
        t_tool = get_tavily_tool()
        d_tool = duck_tool
        news_results = None

        if t_tool:
            try:
                news_results = t_tool.invoke({"query": f"{query} latest news"})
            except Exception:
                pass

        if not news_results and d_tool:
            try:
                news_results = d_tool.invoke({"query": f"{query} latest news"})
            except Exception:
                pass

        if not news_results:
            prompt = f"Give a briefing on recent developments and current landscape regarding: '{query}'."
            return self.model.invoke([HumanMessage(content=prompt)]).content

        try:
            summary_prompt = f"""Summarize these latest news updates about '{query}':
{news_results}

Provide:
1. **Headline Summary**: Main story.
2. **Key Developments**: Bullet points of facts.
3. **Broader Impact**: What this means going forward."""
            summary = self.model.invoke([HumanMessage(content=summary_prompt)]).content
            return f"## 📰 News Briefing: {query}\n\n{summary}"
        except Exception as e:
            return f"❌ News query error: {str(e)}"


class VideoAnalysisAgent:
    def __init__(self, model):
        self.model = model

    def process(self, query: str, video_url: str = None) -> str:
        try:
            if video_url:
                analysis = youtube_qa.invoke({"video_url": video_url, "question": query})
                return f"""## 🎥 Video Transcript Q&A

**Video**: [{video_url}]({video_url})  
**Question**: {query}

### 📋 Answer from Transcript
{analysis}"""
            else:
                return "🎥 **Video Analysis Agent**: Please provide a YouTube video link along with your question (e.g., *'Summarize this video: https://youtube.com/watch?v=xyz'*)."
        except Exception as e:
            return f"❌ Video analysis error: {str(e)}"


class CodeAgent:
    def __init__(self, model):
        self.model = model

    def process(self, query: str) -> str:
        try:
            return code_assistant_tool.invoke({"query": query})
        except Exception as e:
            return f"❌ Code assistant error: {str(e)}"


class InterviewAgent:
    def __init__(self, model):
        self.model = model

    def process(self, query: str) -> str:
        try:
            return interview_prep_tool.invoke({"query": query})
        except Exception as e:
            return f"❌ Interview prep error: {str(e)}"


class HealthAgent:
    def __init__(self, model):
        self.model = model

    def process(self, query: str) -> str:
        try:
            return health_advisor_tool.invoke({"query": query})
        except Exception as e:
            return f"❌ Health advisor error: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# QUERY ANALYZER & SUPERVISOR
# ═══════════════════════════════════════════════════════════════
class QueryAnalyzer:
    def __init__(self, model):
        self.model = model

    def analyze_intent(self, query: str) -> Dict:
        try:
            video_url = None
            video_match = re.search(
                r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([^&\s]+)', query
            )
            if video_match:
                video_url = video_match.group(0)

            sub_queries = re.split(r'[;|\n]|and also|also', query)
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
            complexity = "complex" if len(sub_queries) > 1 else "simple"

            query_lower = query.lower()

            if any(w in query_lower for w in ['code', 'program', 'function', 'debug', 'script',
                    'algorithm', 'python', 'javascript', 'java', 'html', 'css', 'sql', 'api',
                    'bug', 'error in code', 'write a', 'implement', 'syntax', 'compile',
                    'class', 'method', 'variable', 'array', 'loop', 'regex']):
                intent = "code"
            elif any(w in query_lower for w in ['interview', 'hire', 'job interview', 'mock interview',
                    'interview question', 'behavioral question', 'prepare for interview',
                    'interview tips', 'hr round', 'technical round']):
                intent = "interview"
            elif any(w in query_lower for w in ['health', 'medical', 'symptom', 'disease', 'medicine',
                    'doctor', 'wellness', 'nutrition', 'diet', 'exercise', 'mental health',
                    'anxiety', 'depression', 'vitamin', 'blood pressure', 'diabetes',
                    'fever', 'headache', 'pain', 'infection', 'vaccine']):
                intent = "health"
            elif any(w in query_lower for w in ['paper', 'research', 'study', 'academic', 'scholar']):
                intent = "research"
            elif any(w in query_lower for w in ['news', 'latest', 'recent', 'today', 'current', 'headlines']):
                intent = "news"
            elif any(w in query_lower for w in ['resume', 'cv', 'job application', 'career', 'cover letter']):
                intent = "resume"
            elif video_url or 'analyze video' in query_lower or 'summarize video' in query_lower:
                intent = "video_analysis"
            else:
                intent = "education"

            return {
                "intent": intent,
                "sub_queries": sub_queries if len(sub_queries) > 1 else [query],
                "video_url": video_url,
                "complexity": complexity
            }
        except Exception:
            return {
                "intent": "education",
                "sub_queries": [query],
                "video_url": None,
                "complexity": "simple"
            }


class MultiAgentSupervisor:
    def __init__(self, model):
        self.model = model
        self.analyzer = QueryAnalyzer(model)
        self.agents = {
            "education": EducationAgent(model),
            "code": CodeAgent(model),
            "research": ResearchAgent(model),
            "interview": InterviewAgent(model),
            "health": HealthAgent(model),
            "news": NewsAgent(model),
            "resume": ResumeAgent(model),
            "video_analysis": VideoAnalysisAgent(model),
            "pdf": PDFRAGAgent(),
        }

    def process_query(self, query: str) -> str:
        force = st.session_state.get("force_agent", "auto")
        if force != "auto" and force in self.agents:
            st.session_state.current_agent = force
            agent = self.agents[force]
            if force == "video_analysis":
                video_match = re.search(
                    r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([^&\s]+)', query)
                video_url = video_match.group(0) if video_match else None
                return agent.process(query, video_url)
            return agent.process(query)

        analysis = self.analyzer.analyze_intent(query)
        st.session_state.current_agent = analysis["intent"]

        if len(analysis["sub_queries"]) > 1:
            responses = []
            for sub_query in analysis["sub_queries"]:
                single_analysis = {
                    "intent": analysis["intent"],
                    "sub_queries": [sub_query],
                    "video_url": analysis.get("video_url"),
                }
                responses.append(self._process_single_query(single_analysis))
            return "\n\n---\n\n".join(responses)
        else:
            return self._process_single_query(analysis)

    def _process_single_query(self, analysis: Dict) -> str:
        intent = analysis["intent"]
        query = analysis["sub_queries"][0]
        agent = self.agents.get(intent, self.agents["education"])

        if intent == "video_analysis" and analysis.get("video_url"):
            return agent.process(query, analysis["video_url"])
        return agent.process(query)


supervisor = MultiAgentSupervisor(hf_model) if hf_model else None


# ═══════════════════════════════════════════════════════════════
# FOLLOW-UP SUGGESTIONS
# ═══════════════════════════════════════════════════════════════
def generate_followup_suggestions(query: str, agent: str) -> list:
    """Generate 3 context-aware follow-up questions."""
    suggestions_map = {
        "education": [
            f"Give me real-world examples of {query}",
            f"What are common misconceptions about {query}?",
            f"How does {query} compare to related concepts?"
        ],
        "code": [
            "Can you optimize this code for performance?",
            "Write comprehensive unit tests for this",
            "Explain the time and space complexity"
        ],
        "interview": [
            "What are common mistakes candidates make?",
            "Give me a STAR model answer for this",
            "What follow-up questions might the interviewer ask?"
        ],
        "health": [
            f"What evidence-based lifestyle changes support {query}?",
            f"What are common risk factors or causes?",
            f"What questions should I ask my doctor about {query}?"
        ],
        "research": [
            f"What are the latest breakthroughs in {query}?",
            f"Who are the leading authors in {query}?",
            f"What are open research questions in this area?"
        ],
        "resume": [
            "How should I quantify achievements for this role?",
            "What are the best power action verbs to use?",
            "How do I tailor this for ATS screening systems?"
        ],
        "news": [
            f"What is the background context for {query}?",
            f"How does this impact the wider industry?",
            f"What are experts forecasting next?"
        ],
        "video_analysis": [
            "Summarize the key takeaways from the video",
            "What action items or conclusions were presented?",
            "What were the main arguments made?"
        ]
    }
    return suggestions_map.get(agent, [
        f"Can you provide more examples of {query}?",
        f"What are the practical applications of {query}?",
        f"Explain this from an advanced perspective"
    ])


# ═══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════
def render_agent_badge(agent: str):
    meta = AGENT_META.get(agent, AGENT_META["general"])
    st.markdown(
        f'<span class="agent-badge {meta["badge"]}">{meta["icon"]} {meta["label"]}</span>',
        unsafe_allow_html=True
    )

def display_chat_message(message: str, is_user: bool = True, agent: str = None, timestamp: str = None):
    if is_user:
        with st.chat_message("user", avatar="🧑‍💻"):
            if timestamp:
                st.caption(f"🕒 {timestamp}")
            st.markdown(message)
    else:
        meta = AGENT_META.get(agent, AGENT_META["general"])
        with st.chat_message("assistant", avatar=meta.get("icon", "🤖")):
            render_agent_badge(agent)
            if timestamp:
                st.caption(f"🕒 {timestamp}")
            st.markdown(message)

def display_video_info(video_url: str):
    video_id = extract_video_id(video_url)
    if video_id:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(get_video_thumbnail(video_id), use_container_width=True)
        with col2:
            st.markdown(f"🎬 **Video**: [{video_url}]({video_url})")

def display_pdf_tab():
    st.subheader("📑 PDF Q&A with RAG")
    st.markdown("Upload a PDF document to index it and ask questions using local embeddings.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to analyze"
    )

    if "pdf_agent" not in st.session_state:
        st.session_state.pdf_agent = PDFRAGAgent()

    if uploaded_file is not None:
        if st.button("📤 Process PDF", type="primary"):
            with st.spinner("Indexing PDF..."):
                success = st.session_state.pdf_agent.load_pdf(uploaded_file)
            if success:
                st.success("✅ PDF loaded successfully! You can now ask questions.")
                st.session_state.pdf_processed = True
            else:
                st.error("❌ Failed to process PDF. Please try again.")
                st.session_state.pdf_processed = False

    if getattr(st.session_state, 'pdf_processed', False):
        st.markdown("---")
        st.subheader("Ask Questions")
        user_question = st.text_input(
            "What would you like to know about the document?",
            placeholder="e.g., What is the main thesis? Summarize chapter 1."
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear PDF", use_container_width=True):
                st.session_state.pdf_processed = False
                st.session_state.pdf_agent = PDFRAGAgent()
                st.rerun()

        if ask_button and user_question.strip():
            with st.spinner("Analyzing document..."):
                answer = st.session_state.pdf_agent.answer_question(user_question, hf_model)
            st.markdown("### 📝 Answer")
            st.markdown(answer)
            timestamp = time.strftime("%H:%M")
            st.session_state.chat_history.append((
                f"📑 PDF Q&A: {user_question}", answer, "pdf", timestamp
            ))
    else:
        st.info("👆 Upload and process a PDF file to begin.")


# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════
def main():
    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown('<div class="hero-title" style="font-size:1.5rem;">🧠 IntelliAgent</div>', unsafe_allow_html=True)
        st.markdown("---")

        # Model status
        model_name = st.session_state.get("model_name", "Not Connected")
        if hf_model:
            st.markdown(
                f'<span class="status-pill status-online">● {model_name}</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-pill status-offline">● Not Connected</span>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Agent mode selector
        st.markdown("##### 🎯 Agent Mode")
        agent_options = {
            "auto": "🤖 Auto-Detect",
            "education": "🎓 Education",
            "code": "💻 Code Assistant",
            "interview": "🎤 Interview Prep",
            "health": "🏥 Health Advisor",
            "research": "🔬 Research",
            "news": "📰 News",
            "resume": "📄 Resume",
            "video_analysis": "🎥 Video Analysis",
        }
        selected = st.selectbox(
            "Force agent",
            options=list(agent_options.keys()),
            format_func=lambda x: agent_options[x],
            label_visibility="collapsed"
        )
        st.session_state.force_agent = selected

        st.markdown("---")

        # Session stats
        st.markdown("##### 📊 Session Stats")
        st.markdown(f"**Messages**: {len(st.session_state.chat_history)}")
        if st.session_state.chat_history:
            agents_used = set(item[2] for item in st.session_state.chat_history if len(item) > 2)
            st.markdown(f"**Specialists used**: {len(agents_used)}")

        st.markdown("---")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── MAIN CONTENT ──
    st.markdown(
        '<div class="hero-title">🧠 IntelliAgent</div>'
        '<div class="hero-subtitle">Multi-Agent AI Assistant — Education • Code • Interview • Health • Research • News • Resume • PDF</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # Tabs
    tab_chat, tab_pdf, tab_help = st.tabs([
        "💬 Chat", "📑 PDF Analysis", "📖 Help"
    ])

    # ── CHAT TAB ──
    with tab_chat:
        if st.session_state.chat_history:
            for i, (user_msg, bot_msg, agent, timestamp) in enumerate(st.session_state.chat_history[-15:]):
                display_chat_message(user_msg, is_user=True, timestamp=timestamp)
                if "youtube.com" in user_msg or "youtu.be" in user_msg:
                    display_video_info(user_msg)
                display_chat_message(bot_msg, is_user=False, agent=agent, timestamp=timestamp)
                st.markdown("")

            # Follow-up suggestions
            last_query = st.session_state.chat_history[-1][0]
            last_agent = st.session_state.chat_history[-1][2]
            suggestions = generate_followup_suggestions(last_query, last_agent)

            st.markdown("**💡 Follow-up suggestions:**")
            cols = st.columns(len(suggestions))
            for idx, (col, suggestion) in enumerate(zip(cols, suggestions)):
                with col:
                    if st.button(suggestion, key=f"followup_{idx}", use_container_width=True):
                        st.session_state.followup_query = suggestion
                        st.rerun()

        st.markdown("---")

        default_value = st.session_state.pop("followup_query", "")

        with st.container():
            user_input = st.text_area(
                "Ask me anything:",
                height=110,
                value=default_value,
                placeholder="Try:\n• Explain quantum computing\n• Write a Python script to sort a list\n• Prepare me for a software engineer interview\n• What are evidence-based tips for better sleep?",
                key="user_input"
            )

            col1, col2 = st.columns([3, 1])

            with col1:
                send_button = st.button("🚀 Send Message", type="primary", use_container_width=True)

            with col2:
                if st.button("✨ Example", use_container_width=True):
                    st.session_state.followup_query = "Write a Python function to check for palindromes with unit tests"
                    st.rerun()

        # Process input
        if send_button and user_input.strip() and supervisor:
            if not st.session_state.processing:
                st.session_state.processing = True
                timestamp = time.strftime("%H:%M")

                force = st.session_state.get("force_agent", "auto")
                if force != "auto":
                    meta = AGENT_META.get(force, AGENT_META["general"])
                    st.info(f"{meta['icon']} Routing to **{meta['label']} Agent**")

                with st.spinner("🧠 IntelliAgent is thinking..."):
                    try:
                        response = supervisor.process_query(user_input)
                        agent = st.session_state.current_agent
                        st.session_state.chat_history.append((user_input, response, agent, timestamp))
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error processing your request: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append((user_input, error_msg, "error", timestamp))
                    finally:
                        st.session_state.processing = False

        elif send_button and not supervisor:
            st.error("⚠️ AI model is not initialized. Please verify `GROQ_API_KEY` in your `.env` file.")

    # ── PDF TAB ──
    with tab_pdf:
        display_pdf_tab()

    # ── HELP TAB ──
    with tab_help:
        st.header("📖 Help & Documentation")

        st.subheader("Available Specialist Agents")
        agents_info = {
            "💻 Code Assistant": "Generates, explains, reviews, and debugs code across any programming language",
            "🎤 Interview Prep": "Role-specific interview preparation, STAR method model answers, and coaching",
            "🏥 Health Advisor": "Explains medical and wellness topics in plain language (educational, not medical advice)",
            "🎓 Education Agent": "Breaks down complex academic and technical topics with step-by-step clarity",
            "🔬 Research Agent": "Fetches and synthesizes academic papers from Semantic Scholar",
            "📰 News Agent": "Searches and summarizes recent real-world news and developments",
            "📄 Resume Agent": "Generates professional ATS-optimized LaTeX resumes",
            "🎥 Video Analysis": "Answers specific questions about any YouTube video using subtitles/transcripts",
            "📑 PDF Q&A": "Answers questions based on uploaded PDF documents using local vector embeddings"
        }
        for agent, description in agents_info.items():
            st.markdown(f"**{agent}**: {description}")

        st.markdown("---")
        st.subheader("Tips for Best Results")
        st.markdown("""
        - **Automatic Intent Detection**: Simply type naturally. The supervisor will detect whether your query is about code, interviews, health, education, etc.
        - **Manual Override**: Want to guarantee a specific specialist? Choose it from the **Agent Mode** dropdown in the sidebar.
        - **YouTube Video Analysis**: Paste any YouTube link directly into your query along with your question!
        """)


if __name__ == "__main__":
    main()
