import streamlit as st
import re
import tempfile
import json
import requests
from typing import List, Dict, Annotated, TypedDict, Literal
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from langchain_community.tools.tavily_search import TavilySearchResults
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import urlparse, parse_qs
import traceback
import pdfplumber
# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
YOUTUBE_API_KEY= os.getenv("YOUTUBE_API_KEY")
GroQ_API_KEY= os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="AI Multi-Agent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# st.markdown("""
# <style>
# /* 🇮🇳 Main Chat Container */
# .chat-container {
#     background: linear-gradient(135deg, #FF9933, #FFFFFF, #138808) !important;
#     background-size: 300% 300%;
#     animation: gradientBG 15s ease infinite;
#     border-radius: 18px;
#     padding: 15px;
#     border: none !important;
#     color: black;
# }

# /* Gradient animation */
# @keyframes gradientBG {
#     0% {background-position: 0% 50%;}
#     50% {background-position: 100% 50%;}
#     100% {background-position: 0% 50%;}
# }

# /* 👤 User Message */
# .user-message {
#     background: linear-gradient(135deg, #FF9933, #FFFFFF);
#     color: black !important;
#     padding: 12px 18px;
#     border-radius: 15px 15px 0px 15px;
#     margin: 8px 0;
#     box-shadow: 0 4px 10px rgba(255, 153, 51, 0.4);
#     animation: floatUp 0.6s ease;
# }

# /* 🤖 Bot Message */
# .bot-message {
#     background: linear-gradient(135deg, #FFFFFF, #138808);
#     color: black !important;
#     padding: 12px 18px;
#     border-radius: 15px 15px 15px 0px;
#     margin: 8px 0;
#     border-left: 4px solid #FF9933;
#     box-shadow: 0 4px 10px rgba(19, 136, 8, 0.4);
#     animation: floatUp 0.6s ease;
# }

# /* Floating animation */
# @keyframes floatUp {
#     from {opacity: 0; transform: translateY(10px);}
#     to {opacity: 1; transform: translateY(0);}
# }

# /* 📊 Metric Cards */
# .metric-card {
#     background: linear-gradient(135deg, #FF9933, #FFFFFF, #138808);
#     color: black !important;
#     border-radius: 15px;
#     padding: 15px;
#     text-align: center;
#     box-shadow: 0 4px 15px rgba(255, 153, 51, 0.4);
#     transition: transform 0.3s ease;
# }
# .metric-card:hover {
#     transform: scale(1.05);
# }

# /* 🎭 Agent Badge */
# .agent-badge {
#     display: inline-block;
#     background: linear-gradient(45deg, #FF9933, #FFFFFF, #138808);
#     color: black;
#     padding: 0.3rem 0.8rem;
#     border-radius: 20px;
#     font-size: 0.8rem;
#     font-weight: bold;
#     margin-bottom: 0.5rem;
# }

# /* Sidebar styling */
# section[data-testid="stSidebar"] {
#     background: linear-gradient(135deg, #FF9933, #FFFFFF, #138808) !important;
#     color: black !important;
# }
# section[data-testid="stSidebar"] .css-1v3fvcr, 
# section[data-testid="stSidebar"] .css-1d391kg {
#     color: black !important;
# }

# /* Buttons */
# div.stButton > button {
#     background: linear-gradient(135deg, #FF9933, #FFFFFF, #138808) !important;
#     color: black !important;
#     border-radius: 25px;
#     border: none;
#     padding: 0.6rem 1.2rem;
#     font-weight: bold;
#     box-shadow: 0 4px 10px rgba(255, 153, 51, 0.4);
#     transition: transform 0.2s ease;
# }
# div.stButton > button:hover {
#     transform: scale(1.05);
#     background: linear-gradient(135deg, #FF9933, #FFFFFF, #138808) !important;
# }

# /* Text Inputs */
# input, textarea {
#     border-radius: 10px !important;
#     border: 1px solid #FF9933 !important;
#     padding: 0.5rem !important;
# }
# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>
/* Import beautiful fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styling */
* {
    font-family: 'Inter', sans-serif !important;
}

/* Main app background with subtle pattern */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
}

/* Main Chat Container with glassmorphism */
.chat-container {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 25px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
}

.chat-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, #FF9933, #FFFFFF, #138808);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    opacity: 0.1;
    z-index: -1;
}

/* Enhanced gradient animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    25% { background-position: 100% 50%; }
    50% { background-position: 100% 100%; }
    75% { background-position: 0% 100%; }
    100% { background-position: 0% 50%; }
}

/* User Message with enhanced styling */
.user-message {
    background: linear-gradient(135deg, #FF9933 0%, #FFB366 50%, #FFFFFF 100%);
    color: #2c3e50 !important;
    padding: 16px 24px;
    border-radius: 25px 25px 8px 25px;
    margin: 12px 0;
    box-shadow: 
        0 8px 25px rgba(255, 153, 51, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.6);
    animation: slideInRight 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.3);
    font-weight: 500;
    letter-spacing: 0.3px;
}

.user-message::before {
    content: '👤';
    position: absolute;
    top: -8px;
    right: -8px;
    background: #FF9933;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

/* Bot Message with enhanced styling */
.bot-message {
    background: linear-gradient(135deg, #FFFFFF 0%, #E8F5E8 50%, #138808 100%);
    color: #2c3e50 !important;
    padding: 16px 24px;
    border-radius: 25px 25px 25px 8px;
    margin: 12px 0;
    border-left: 5px solid #138808;
    box-shadow: 
        0 8px 25px rgba(19, 136, 8, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.6);
    animation: slideInLeft 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.3);
    font-weight: 400;
    letter-spacing: 0.2px;
    line-height: 1.6;
}

.bot-message::before {
    content: '🤖';
    position: absolute;
    top: -8px;
    left: -8px;
    background: #138808;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

/* Enhanced animations */
@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(30px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-30px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
    }
}

/* Enhanced Metric Cards with 3D effect */
.metric-card {
    background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
    color: #2c3e50 !important;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    box-shadow: 
        0 15px 35px rgba(255, 153, 51, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.6);
    transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.3);
    backdrop-filter: blur(10px);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    transition: left 0.5s;
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 
        0 25px 50px rgba(255, 153, 51, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

.metric-card:hover::before {
    left: 100%;
}

/* Enhanced Agent Badge with pulsing effect */
.agent-badge {
    display: inline-block;
    background: linear-gradient(45deg, #FF9933, #FFFFFF, #138808);
    color: #2c3e50;
    padding: 0.4rem 1rem;
    border-radius: 25px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 15px rgba(255, 153, 51, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.3);
    animation: pulse 2s infinite;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

/* Enhanced Sidebar with glassmorphism */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
}

section[data-testid="stSidebar"] .css-1v3fvcr, 
section[data-testid="stSidebar"] .css-1d391kg,
section[data-testid="stSidebar"] * {
    color: #2c3e50 !important;
    font-weight: 500 !important;
}

/* Enhanced Buttons with modern styling */
div.stButton > button {
    background: linear-gradient(135deg, #FF9933, #FFB366, #138808) !important;
    color: #2c3e50 !important;
    border-radius: 30px !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    font-weight: 600 !important;
    box-shadow: 
        0 8px 25px rgba(255, 153, 51, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    position: relative;
    overflow: hidden;
}

div.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    transition: left 0.5s;
}

div.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 
        0 15px 35px rgba(255, 153, 51, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
}

div.stButton > button:hover::before {
    left: 100%;
}

div.stButton > button:active {
    transform: translateY(-1px) scale(0.98) !important;
}

/* Enhanced Text Inputs with better visibility */
input, textarea, .stTextInput input, .stTextArea textarea {
    border-radius: 15px !important;
    border: 2px solid #FF9933 !important;
    padding: 0.8rem !important;
    background: #2c3e50 !important;
    color: #ffffff !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 
        0 4px 15px rgba(0, 0, 0, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
    font-size: 16px !important;
}

input::placeholder, textarea::placeholder, 
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: #bdc3c7 !important;
    opacity: 0.8 !important;
}

input:focus, textarea:focus, .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #FFB366 !important;
    background: #34495e !important;
    color: #ffffff !important;
    box-shadow: 
        0 8px 25px rgba(255, 153, 51, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-2px) !important;
    outline: none !important;
}

/* Enhanced File Uploader */
.stFileUploader {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    border: 2px dashed rgba(255, 153, 51, 0.5) !important;
    padding: 2rem !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}

.stFileUploader:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    border-color: #FF9933 !important;
    transform: scale(1.02) !important;
}

/* Enhanced Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 25px !important;
    padding: 0.5rem !important;
    backdrop-filter: blur(10px) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 20px !important;
    color: #2c3e50 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #FF9933, #FFB366, #138808) !important;
    color: #2c3e50 !important;
    box-shadow: 0 4px 15px rgba(255, 153, 51, 0.4) !important;
}

/* Loading spinner enhancement */
.stSpinner > div {
    border-top-color: #FF9933 !important;
    border-right-color: #FF9933 !important;
}

/* Success/Error message enhancement */
.stSuccess {
    background: linear-gradient(135deg, rgba(19, 136, 8, 0.1), rgba(255, 255, 255, 0.1)) !important;
    border-left: 4px solid #138808 !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
}

.stError {
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(255, 255, 255, 0.1)) !important;
    border-left: 4px solid #e74c3c !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #FF9933, #138808);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #FFB366, #4CAF50);
}

/* Responsive design */
@media (max-width: 768px) {
    .user-message, .bot-message {
        padding: 12px 16px;
        margin: 8px 0;
    }
    
    .metric-card {
        padding: 15px;
    }
    
    div.stButton > button {
        padding: 0.6rem 1.5rem !important;
    }
}

/* Additional enhancements */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #2c3e50 !important;
    font-weight: 700 !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.stMarkdown code {
    background: rgba(255, 153, 51, 0.1) !important;
    border: 1px solid rgba(255, 153, 51, 0.3) !important;
    border-radius: 6px !important;
    color: #2c3e50 !important;
}

/* Hover effects for interactive elements */
.stSelectbox > div, .stMultiSelect > div {
    transition: all 0.3s ease !important;
}

.stSelectbox > div:hover, .stMultiSelect > div:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255, 153, 51, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)



# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = "general"
    if 'processing' not in st.session_state:
        st.session_state.processing = False

initialize_session_state()

# Initialize the model
@st.cache_resource
def initialize_model():
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize AI model: {str(e)}")
        return None

hf_model = initialize_model()

# Utility functions
def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    try:
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    except:
        return None

def is_valid_youtube_url(url):
    """Validate YouTube URL format."""
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=([^&\s]+)',
        r'https?://youtu\.be/([^&\s]+)',
        r'https?://(?:www\.)?youtube\.com/embed/([^&\s]+)'
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)

def get_video_thumbnail(video_id):
    """Get YouTube video thumbnail URL."""
    if video_id:
        return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    return None

# Enhanced Tools
@st.cache_data(ttl=300)
def youtube_search_cached(query: str) -> str:
    """Search YouTube for videos with caching."""
    try:
        url = f"https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 5,
            "key": YOUTUBE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        res = response.json()
        
        videos = []
        for item in res.get("items", []):
            title = item['snippet']['title']
            video_id = item['id']['videoId']
            url = f"https://www.youtube.com/watch?v={video_id}"
            videos.append(f"**{title}**: {url}")
        
        return "\n".join(videos) if videos else "No related videos found."
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"

@tool
def youtube_search(query: str) -> str:
    """Search YouTube for videos related to the query."""
    return youtube_search_cached(query)

@tool
def topic_explanation(query: str) -> str:
    """Return a comprehensive conceptual explanation of the topic."""
    if not hf_model:
        return "AI model is not available. Please check configuration."
    
    try:
        prompt = f"""
        Provide a detailed explanation of '{query}' covering:
        1. Core concept and definition
        2. Key components or principles
        3. Real-world applications
        4. Why it's important
        
        Make it beginner-friendly but comprehensive.
        """
        return hf_model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

@tool
def generate_resume(job_description: str, candidate_info: str = "") -> str:
    """Generate a professional LaTeX resume optimized for ATS systems."""
    if not hf_model:
        return "AI model is not available. Please check configuration."
    
    try:
        prompt = f"""
        Create a modern, ATS-friendly LaTeX resume using clean formatting.
        
        Requirements:
        - Use standard LaTeX packages (no exotic dependencies)
        - Include proper sections: Contact, Summary, Skills, Experience, Education, Projects
        - Optimize for keyword matching
        - Professional formatting with clear hierarchy
        
        Job Description: {job_description}
        Candidate Info: {candidate_info if candidate_info else "Entry-level candidate"}
        
        Return only valid LaTeX code.
        """
        return hf_model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        return f"Error generating resume: {str(e)}"

# Initialize other tools
try:
    ss = SemanticScholarAPIWrapper(top_k_results=5, load_max_docs=5)
except:
    ss = None

@tool
def semantic_scholar_research(query: str) -> List[Dict]:
    """Fetch and summarize top research papers from Semantic Scholar."""
    if not ss:
        return [{"error": "Semantic Scholar not available"}]
    
    try:
        raw = ss.run(query)
        papers = raw.split("\n\n")
        result = []
        
        for p in papers[:3]:
            if "abstract:" in p.lower():
                if hf_model:
                    summary = hf_model.invoke([
                        HumanMessage(content=f"Summarize this research paper in 3-4 sentences:\n{p}")
                    ]).content
                    result.append({"raw_info": p, "summary": summary})
                else:
                    result.append({"raw_info": p, "summary": "AI model unavailable for summarization"})
        
        return result
    except Exception as e:
        return [{"error": f"Error fetching papers: {str(e)}"}]

# YouTube QA functionality
prompt_template = """
You are a helpful assistant designed to answer questions about a YouTube video based on its transcript.
Answer the user's question using ONLY the provided transcript context.
If the information is not in the context, explicitly say "I cannot find information about that in the video transcript."

Transcript:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

def get_transcript(video_id: str):
    """Fetch transcript safely with enhanced error handling."""
    if not video_id:
        return None, "Invalid video ID provided."
    
    try:
        # Try Hindi first
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["hi"])
    except NoTranscriptFound:
        try:
            # Fallback to English
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        except NoTranscriptFound:
            try:
                # Try auto-generated captions
                transcript_list = YouTubeTranscriptApi().list_transcripts(video_id)
                transcript = transcript_list.find_generated_transcript(['hi', 'en'])
                transcript_list = transcript.fetch()
            except:
                return None, "No transcript available in Hindi, English, or auto-generated captions."
    except (TranscriptsDisabled, VideoUnavailable) as e:
        return None, f"Video transcript unavailable: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error fetching transcript: {str(e)}"

    try:
        texts = []
        for snippet in transcript_list:
            if isinstance(snippet, dict):
                texts.append(snippet.get("text", ""))
            else:
                texts.append(getattr(snippet, "text", ""))

        transcript = " ".join(texts)
        if len(transcript.strip()) < 10:
            return None, "Retrieved transcript is too short or empty."
            
        return transcript, None
    except Exception as e:
        return None, f"Error processing transcript: {str(e)}"

@tool
def youtube_qa(video_url: str, question: str):
    """Given a YouTube URL and question, return answer based on transcript."""
    if not hf_model:
        return "AI model is not available. Please check configuration."
    
    try:
        if not is_valid_youtube_url(video_url):
            return "Invalid YouTube URL format. Please provide a valid YouTube link."
        
        video_id = extract_video_id(video_url)
        if not video_id:
            return "Could not extract video ID from the provided URL."
        
        transcript, error = get_transcript(video_id)
        
        if error:
            return f"Transcript Error: {error}"

        if not transcript or len(transcript.strip()) < 10:
            return "Retrieved transcript is empty or too short to analyze."

        # Limit transcript size
        if len(transcript) > 8000:
            transcript = transcript[:8000] + "... (transcript truncated)"

        rag_runnable = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | hf_model
        )
        answer = rag_runnable.invoke({"context": transcript, "question": question})
        
        return answer.content if hasattr(answer, 'content') else str(answer)

    except Exception as e:
        return f"Error processing video: {str(e)}"

# Initialize search tools
try:
    duck_tool = DuckDuckGoSearchRun()
    tavily_tool =TavilySearchResults(max_results=5)

except:
    duck_tool = None
    tavily_tool = None

# *****************************************************
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

class PDFRAGAgent:
    def __init__(self, hf_token):
        self.hf_token = hf_token
        self.retriever = None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_embedding_model():
        """Load HuggingFaceEmbeddings once and cache it"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_pdf(self, pdf_file):
        """Load and index a single PDF"""
        # Create temporary file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        try:
            # Load PDF documents
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=900, 
                chunk_overlap=30
            )
            split_docs = text_splitter.split_documents(documents)

            # Get cached embedding model - CORRECTED: Remove parameter
            embedding_model = self.get_embedding_model()

            # Create vector database - CORRECTED: Use embedding_model directly
            db = Chroma.from_documents(
                split_docs,
                embedding=embedding_model,  # Use the HuggingFaceEmbeddings object directly
                collection_name="student_pdf"
            )

            self.retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # top 3 similar chunks
            )

            return True

        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

    def answer_question(self, query, model):
        """Answer a question based on the uploaded PDF"""
        if not self.retriever:
            return "⚠️ Please upload a PDF first."

        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content for d in docs])

            if not context.strip():
                return "❌ No relevant content found in the PDF for your question."

            # Prepare prompt
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant for students.\n"
                "Answer the question based only on the provided document context.\n"
                "If the information is not clearly present in the context, say so.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )

            # Run chain
            chain = prompt | model
            response = chain.invoke({"context": context, "question": query})
            
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            return f"❌ Error processing question: {str(e)}"

# Updated tab5 section for the main app


# Specialized Agents
class EducationAgent:
    def __init__(self, model):
        self.model = model
        self.tools = [topic_explanation, youtube_search]
        
    def process(self, query: str) -> str:
        try:
            explanation = topic_explanation.invoke({"query": query})
            videos = youtube_search.invoke({"query": query + " tutorial explanation"})
            
            response = f"""
## Educational Response for: {query}

### Conceptual Explanation:
{explanation}

### Related Video Resources:
{videos}

### Study Suggestions:
- Start with the conceptual understanding above
- Watch the recommended videos for visual learning
- Practice with hands-on examples if applicable
"""
            return response
        except Exception as e:
            return f"Error in educational processing: {str(e)}"

class ResearchAgent:
    def __init__(self, model):
        self.model = model
        self.tools = [semantic_scholar_research, youtube_search]
        
    def process(self, query: str) -> str:
        try:
            papers = semantic_scholar_research.invoke({"query": query})
            videos = youtube_search.invoke({"query": query + " research paper explanation"})
            
            response = f"## Research Analysis for: {query}\n\n"
            
            if papers:
                response += "### Academic Papers:\n"
                for i, paper in enumerate(papers, 1):
                    if "summary" in paper:
                        response += f"{i}. **Summary**: {paper['summary']}\n\n"
            
            response += f"### Educational Videos:\n{videos}\n"
            return response
        except Exception as e:
            return f"Error in research processing: {str(e)}"

class ResumeAgent:
    def __init__(self, model):
        self.model = model
        self.tools = [generate_resume]
        
    def process(self, query: str, job_desc: str = "", candidate_info: str = "") -> str:
        try:
            resume_latex = generate_resume.invoke({
                "job_description": job_desc or query,
                "candidate_info": candidate_info
            })
            
            return f"""
## Resume Generation Complete

### LaTeX Resume Code:
```latex
{resume_latex}
```

### Instructions:
1. Copy the LaTeX code above
2. Paste into Overleaf or any LaTeX editor
3. Compile to generate PDF
4. Customize further as needed
"""
        except Exception as e:
            return f"Error generating resume: {str(e)}"

class NewsAgent:
    def __init__(self, model):
        self.model = model
        self.tools = [tavily_tool, duck_tool]
        
    def process(self, query: str) -> str:
        if not tavily_tool:
            return "News search tools are not available."
        
        try:
            news_results = tavily_tool.invoke({"query": query + " latest news today"})
            summary_prompt = f"""
            Summarize the latest news about '{query}' based on this information:
            {news_results}
            
            Provide:
            1. Key headlines
            2. Important developments
            3. Implications or analysis
            """
            
            summary = self.model.invoke([HumanMessage(content=summary_prompt)]).content
            return f"## Latest News: {query}\n\n{summary}"
        except Exception as e:
            return f"Error fetching news: {str(e)}"

class VideoAnalysisAgent:
    def __init__(self, model):
        self.model = model
        self.tools = [youtube_qa, youtube_search]
        
    def process(self, query: str, video_url: str = None) -> str:
        try:
            if video_url:
                analysis = youtube_qa.invoke({"video_url": video_url, "question": query})
                return f"## Video Analysis\n\n**Question**: {query}\n\n**Answer**: {analysis}"
            else:
                videos = youtube_search.invoke({"query": query})
                return f"## Video Resources for: {query}\n\n{videos}"
        except Exception as e:
            return f"Error in video analysis: {str(e)}"

# Query Analyzer
class QueryAnalyzer:
    def __init__(self, model):
        self.model = model
        
    def analyze_intent(self, query: str) -> Dict[str, any]:
        """
        Analyze user query for intent, sub-queries, video suggestion, and complexity.
        Returns a dictionary with:
        - intent: one of education, research, resume, news, video_analysis, general
        - sub_queries: list of split queries if complex
        - video_url: YouTube URL if present
        - suggest_video: True/False
        - complexity: simple/complex
        """
        try:
            # Step 1: Extract YouTube video URL (if any)
            video_url = None
            video_match = re.search(
                r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([^&\s]+)', query
            )
            if video_match:
                video_url = video_match.group(0)

            # Step 2: Split query into sub-queries if needed
            sub_queries = re.split(r'[;|\n]|and also|also', query)
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
            complexity = "complex" if len(sub_queries) > 1 else "simple"

            # Step 3: Prepare robust semantic LLM prompt
            prompt = f"""
You are an AI assistant that classifies user queries into intent. 
The possible intents are: education, research, resume, news, video_analysis, general.

Rules:
- Queries asking how to learn, understand, study, or get started with a subject should be classified as "education".
- Queries asking for latest studies, papers, or academic research should be "research".
- Queries about job applications, resumes, or CVs should be "resume".
- Queries asking for latest news or trends should be "news".
- Queries asking to analyze videos or provide summaries of videos should be "video_analysis".
- All other queries default to "general".

Also decide if a video would help explain or teach the query. 

Return ONLY JSON like: 
{{ "intent": "education", "suggest_video": "yes" }}

Query: {query}
"""
            # Step 4: Call LLM for semantic classification
            result = self.model.invoke([HumanMessage(content=prompt)]).content.strip()

            # Step 5: Parse JSON safely
            try:
                result_json = json.loads(result)
                intent = result_json.get("intent", "general").lower()
                suggest_video = result_json.get("suggest_video", "no").lower() == "yes"
            except:
                intent = "general"
                suggest_video = False

            # Step 6: Override video_analysis if URL exists or video suggested
            if video_url:
                intent = "video_analysis"
                suggest_video = True
            elif intent == "education" and suggest_video:
                suggest_video = True

            return {
                "intent": intent,
                "sub_queries": sub_queries if len(sub_queries) > 1 else [query],
                "video_url": video_url if video_url else None,
                "suggest_video": suggest_video,
                "complexity": complexity
            }

        except Exception as e:
            # Fallback in case anything goes wrong
            return {
                "intent": "general",
                "sub_queries": [query],
                "video_url": None,
                "suggest_video": False,
                "complexity": "simple"
            }



# Multi-Agent Supervisor
class MultiAgentSupervisor:
    def __init__(self, model):
        self.model = model
        self.analyzer = QueryAnalyzer(model)
        # Register all agents for full auto-routing
        self.agents = {
            "education": EducationAgent(model),
            "research": ResearchAgent(model),
            "resume": ResumeAgent(model),
            "news": NewsAgent(model),
            "video_analysis": VideoAnalysisAgent(model),
            "pdf": PDFRAGAgent(model)
        }

    def process_query(self, query: str, uploaded_file=None) -> str:
        if uploaded_file:
            return self.agents["pdf"].process(uploaded_file)
        analysis = self.analyzer.analyze_intent(query)
        st.session_state.current_agent = analysis["intent"]
        if len(analysis["sub_queries"]) > 1:
            return self._process_multi_query(analysis)
        else:
            return self._process_single_query(analysis)

    def _process_single_query(self, analysis: Dict) -> str:
        intent = analysis["intent"]
        query = analysis["sub_queries"][0]
        agent = self.agents.get(intent)
        if agent:
            if intent == "video_analysis" and analysis.get("video_url"):
                return agent.process(query, analysis["video_url"])
            elif intent == "resume" and analysis.get("job_description"):
                return agent.process(query, job_desc=analysis.get("job_description"), candidate_info=analysis.get("candidate_info", ""))
            else:
                return agent.process(query)
        return self._general_response(query)

    def _process_multi_query(self, analysis: Dict) -> str:
        main_topic = analysis["sub_queries"][0]  # treat first sub-query as main topic
        responses = []
        for sub_query in analysis["sub_queries"]:
            # If sub-query is about YouTube/video, reuse main topic
            if re.search(r'youtube|video|link', sub_query, re.IGNORECASE):
                single_analysis = {
                    "intent": "education",
                    "sub_queries": [main_topic],
                    "video_url": None,
                    "complexity": "simple"
                }
            else:
                single_analysis = {
                    "intent": analysis["intent"],
                    "sub_queries": [sub_query],
                    "video_url": analysis.get("video_url"),
                    "complexity": "simple"
                }
            responses.append(self._process_single_query(single_analysis))
        return "\n\n---\n\n".join(responses)

    def _general_response(self, query: str) -> str:
        if not hf_model:
            return "AI model unavailable."
        return hf_model.invoke([HumanMessage(content=f"Provide a helpful response: {query}")]).content

# ----------------- Main App -----------------
supervisor = MultiAgentSupervisor(hf_model) if hf_model else None

# Main App Functions
def display_chat_message(message, is_user=True, agent=None):
    if is_user:
        st.markdown(f'<div class="user-message">{message}</div>', unsafe_allow_html=True)
    else:
        agent_badge = f'<div class="agent-badge">{agent.upper() if agent else "ASSISTANT"}</div>' if agent else ""
        st.markdown(f'<div class="bot-message">{agent_badge}{message}</div>', unsafe_allow_html=True)

def display_video_info(video_url):
    video_id = extract_video_id(video_url)
    if video_id:
        col1, col2 = st.columns([1, 2])
        with col1:
            thumbnail_url = get_video_thumbnail(video_id)
            if thumbnail_url:
                try:
                    st.image(thumbnail_url, caption="Video Thumbnail", use_container_width=True)
                except:
                    st.info("Could not load thumbnail")
        with col2:
            st.info(f"**Video ID:** {video_id}")
            st.info(f"**URL:** {video_url}")

# Initialize supervisor
if hf_model:
    supervisor = MultiAgentSupervisor(hf_model)
else:
    supervisor = None

# ===================================================

def display_pdf_tab():
    st.subheader("📑 PDF Q&A with RAG")
    st.write("Upload a PDF document and ask questions about its content!")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"],
        help="Upload a PDF document to analyze and ask questions about"
    )
    
    # Initialize PDF agent in session state
    if "pdf_agent" not in st.session_state:
        st.session_state.pdf_agent = PDFRAGAgent(hf_token)
    
    # Process uploaded file
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF... This may take a moment."):
                success = st.session_state.pdf_agent.load_pdf(uploaded_file)
                
            if success:
                st.success("✅ PDF loaded successfully! You can now ask questions.")
                st.session_state.pdf_processed = True
            else:
                st.error("❌ Failed to process PDF. Please try again.")
                st.session_state.pdf_processed = False
    
    # Question input (only show if PDF is processed)
    if getattr(st.session_state, 'pdf_processed', False):
        st.markdown("---")
        st.subheader("Ask Questions")
        
        user_question = st.text_input(
            "What would you like to know about the document?",
            placeholder="e.g., What is the main topic discussed? Summarize chapter 1."
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ask_button = st.button("Get Answer", type="primary", use_container_width=True)
        
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.pdf_processed = False
                st.session_state.pdf_agent = PDFRAGAgent(hf_token)
                st.rerun()
        
        if ask_button and user_question.strip():
            with st.spinner("Analyzing document..."):
                answer = st.session_state.pdf_agent.answer_question(user_question, hf_model)
            
            st.markdown("### Answer:")
            st.markdown(answer)
            
            # Add to chat history if desired
            timestamp = time.strftime("%H:%M")
            st.session_state.chat_history.append((
                f"PDF Q&A: {user_question}", 
                answer, 
                "pdf", 
                timestamp
            ))
    
    else:
        st.info("👆 Please upload and process a PDF file first to start asking questions.")




# Main App Layout
def main():
    st.title("🤖 AI Multi-Agent Assistant")
    st.markdown("*Your intelligent companion for education, research, video analysis, and more!*")
    
    # Create tabs for different functionalities
    tab1, tab5,tab2, tab3,tab4 = st.tabs(["💬 Chat","📄 PDF Analysis" ,"📊 Analytics", "⚙️ Settings", "📖 Help"])
    

    with tab1:
        # Chat Interface
        st.header("Chat Interface")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Conversation History")
            for i, (user_msg, bot_msg, agent, timestamp) in enumerate(st.session_state.chat_history[-10:]):
                display_chat_message(user_msg, is_user=True)
                display_video_info(user_msg) if "youtube.com" in user_msg or "youtu.be" in user_msg else None
                display_chat_message(bot_msg, is_user=False, agent=agent)
                st.markdown("---")
        
        # Input area
        st.subheader("Ask me anything!")
        
        with st.container():
            user_input = st.text_area(
                "Your message:",
                height=120,
                placeholder="Try:\n• Explain quantum computing\n• Analyze this video: https://youtube.com/watch?v=...\n• Generate a resume for software engineer\n• What are the latest AI trends?",
                key="user_input"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                send_button = st.button("Send Message", type="primary", use_container_width=True)
            
            with col2:
                clear_button = st.button("Clear Chat", use_container_width=True)
            
            with col3:
                if st.button("Example", use_container_width=True):
                    st.session_state.user_input = "Explain machine learning in simple terms"
        
        # Handle button clicks
        if clear_button:
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        if send_button and user_input.strip() and supervisor:
            if not st.session_state.processing:
                st.session_state.processing = True
                
                timestamp = time.strftime("%H:%M")
                
                with st.spinner("AI is processing your request..."):
                    try:
                        response = supervisor.process_query(user_input)
                        agent = st.session_state.current_agent
                        
                        # Add to history
                        st.session_state.chat_history.append((user_input, response, agent, timestamp))
                        
                        # Display new response
                        st.success("Response generated!")
                        display_chat_message(user_input, is_user=True)
                        display_video_info(user_input) if "youtube.com" in user_input or "youtu.be" in user_input else None
                        display_chat_message(response, is_user=False, agent=agent)
                        
                    except Exception as e:
                        error_msg = f"Error processing your request: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append((user_input, error_msg, "error", timestamp))
                    
                    finally:
                        st.session_state.processing = False
        
        elif send_button and not supervisor:
            st.error("AI model is not initialized. Please check your configuration.")
    with tab5:  # assuming 0=Chat,1=Analytics,2=Settings,3=Hel
        display_pdf_tab()

    
    with tab2:
        # Analytics
        st.header("📊 Usage Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f'<div class="metric-card"><h3>{len(st.session_state.chat_history)}</h3><p>Total Conversations</p></div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f'<div class="metric-card"><h3>{st.session_state.current_agent.title()}</h3><p>Current Agent</p></div>',
                unsafe_allow_html=True
            )
        
        with col3:
            status = "Online" if hf_model else "Offline"
            st.markdown(
                f'<div class="metric-card"><h3>{status}</h3><p>Model Status</p></div>',
                unsafe_allow_html=True
            )
        
        # Agent usage statistics
        if st.session_state.chat_history:
            st.subheader("Agent Usage Distribution")
            agents = [item[2] for item in st.session_state.chat_history if len(item) > 2]
            if agents:
                agent_counts = {}
                for agent in agents:
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
                
                for agent, count in agent_counts.items():
                    st.metric(f"{agent.title()} Agent", count)
    
    with tab3:
        # Settings
        st.header("⚙️ Configuration")
        
        st.subheader("Model Configuration")
        model_status = "✅ Connected" if hf_model else "❌ Not Connected"
        st.info(f"Google Gemini Status: {model_status}")
        
        st.subheader("Chat Settings")
        max_history = st.slider("Maximum chat history to display", 5, 50, 10)
        
        st.subheader("Agent Settings")
        default_agent = st.selectbox(
            "Default Agent",
            ["auto", "education", "research", "resume", "news", "video_analysis"]
        )
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
        
        st.subheader("System Information")
        st.json({
            "Model": "Google Gemini 1.5 Flash",
            "Version": "1.0.0",
            "Status": "Active" if hf_model else "Inactive"
        })
    
    with tab4:
        # Help
        st.header("📖 Help & Documentation")
        
        st.subheader("Available Agents")
        
        agents_info = {
            "🎓 Education Agent": "Provides detailed explanations of complex topics with related video resources",
            "🔬 Research Agent": "Fetches and summarizes academic papers from Semantic Scholar",
            "📄 Resume Agent": "Generates professional LaTeX resumes optimized for ATS systems",
            "📰 News Agent": "Fetches and summarizes latest news on any topic",
            "🎥 Video Analysis Agent": "Analyzes YouTube videos and answers questions based on transcripts"
        }
        
        for agent, description in agents_info.items():
            st.markdown(f"**{agent}**: {description}")
        
        st.subheader("Usage Examples")
        
        examples = [
            "**Education**: 'Explain quantum computing in simple terms'",
            "**Video Analysis**: 'Summarize this video: https://youtube.com/watch?v=xyz'",
            "**Research**: 'Find papers about machine learning applications'",
            "**Resume**: 'Create a resume for data scientist position'",
            "**News**: 'What are the latest AI trends today?'",
            "**General**: 'Give me tips to improve productivity'"
        ]

        for ex in examples:
            st.markdown(f"- {ex}")

        st.subheader("FAQ")
        faq_items = {
            "How do I clear the chat history?": "Go to the **Chat tab** and click **Clear Chat**.",
            "Why do I get transcript errors on some videos?": 
                "Not all YouTube videos have transcripts enabled. Try a different video.",
            "How do I generate a resume PDF?": 
                "Copy the LaTeX code output from the Resume Agent, paste it into Overleaf or TeX editor, and compile.",
            "Can I ask multiple things in one query?": 
                "Yes! Use semicolons or 'and also'. The assistant will split into sub-queries."
        }

        for q, a in faq_items.items():
            st.markdown(f"**{q}**")
            st.write(a)
            st.markdown("---")

        st.success("You’re all set! Use the tabs to explore different assistants 🚀")
        


# Run the app
if __name__ == "__main__":
    main()
