from typing import List, Dict, Annotated, TypedDict, Literal
import re
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from langchain_tavily import TavilySearch
import operator
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# # hf_model = ChatHuggingFace(...)  # Configure your model here
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#     task="text-generation",
#     do_sample=False,
#     top_p=1.0,
#     top_k=0,
#     provider="auto"  
# )
# hf_model = ChatHuggingFace(llm=llm)
# 
# State Defined for Multi-Agent System 

hf_model=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash")



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_query: str
    sub_queries: List[str]
    agent_responses: Dict[str, str]
    next_agent: str
    final_response: str
    context: Dict[str, any]

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Tools
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = "AIzaSyBNBTgze_5FR5VHzfZlxc38iLwr7xyYaHE"

@tool
def youtube_search(query: str) -> str:
    """Search YouTube for videos related to the query and return video links with titles."""
    try:
        url = f"https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 5,
            "key": API_KEY
        }
        res = requests.get(url, params=params).json()
        
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
def topic_explanation(query: str) -> str:
    """Return a comprehensive conceptual explanation of the topic."""
    prompt = f"""
    Provide a detailed explanation of '{query}' covering:
    1. Core concept and definition
    2. Key components or principles
    3. Real-world applications
    4. Why it's important
    
    Make it beginner-friendly but comprehensive.
    """
    return hf_model.invoke([HumanMessage(content=prompt)]).content

@tool
def generate_resume(job_description: str, candidate_info: str = "") -> str:
    """Generate a professional LaTeX resume optimized for ATS systems."""
    prompt = f"""
    Create a modern, ATS-friendly LaTeX resume using clean formatting.
    
    Requirements:
    - Use standard LaTeX packages (no exotic dependencies)
    - Include proper sections: Contact, Summary, Skills, Experience, Education, Projects
    - Optimize for keyword matching
    - Professional formatting with clear hierarchy
    - Include quantifiable achievements where possible
    
    Job Description: {job_description}
    Candidate Info: {candidate_info if candidate_info else "Entry-level candidate"}
    
    Return only valid LaTeX code.
    """
    return hf_model.invoke([HumanMessage(content=prompt)]).content

# Initialize other tools
ss = SemanticScholarAPIWrapper(top_k_results=5, load_max_docs=5)

@tool
def semantic_scholar_research(query: str) -> List[Dict]:
    """Fetch and summarize top research papers from Semantic Scholar."""
    try:
        raw = ss.run(query)
        papers = raw.split("\n\n")
        result = []
        
        for p in papers[:3]:  # Limit to top 3 papers
            if "abstract:" in p.lower():
                summary = hf_model.invoke([
                    HumanMessage(content=f"Summarize this research paper in 3-4 sentences:\n{p}")
                ]).content
                result.append({"raw_info": p, "summary": summary})
        
        return result
    except Exception as e:
        return [{"error": f"Error fetching papers: {str(e)}"}]

def get_transcript(video_id: str):
    """Extract transcript from YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
        texts = [snippet.get("text", "") for snippet in transcript_list]
        return " ".join(texts), None
    except Exception as e:
        return None, str(e)

# @tool
# def youtube_qa(video_url: str, question: str) -> str:
#     """Answer questions based on YouTube video transcript."""
#     try:
#         vid_id = video_url.split("v=")[-1].split("&")[0]
#         transcript, error = get_transcript(vid_id)
        
#         if error:
#             return f"Error getting transcript: {error}"
        
#         prompt = f"""
#         Based on this video transcript, answer the question:
        
#         Transcript: {transcript[:3000]}...  # Limit context
        
#         Question: {question}
        
#         Provide a detailed answer based on the video content.
#         """
        
#         return hf_model.invoke([HumanMessage(content=prompt)]).content
#     except Exception as e:
#         return f"Error processing video: {str(e)}"
youtube_prompt_template = PromptTemplate(
    template="""
You are a helpful assistant designed to answer questions about a YouTube video based on its transcript.
Answer the user's question using ONLY the provided transcript context.
If the information is not in the transcript, explicitly say "I cannot find information about that in the video transcript."

Transcript:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

@tool
def youtube_qa(video_url: str, question: str) -> str:
    """Answer questions based on YouTube video transcript using structured prompts."""
    try:
        # Extract video ID
        vid_id = video_url.split("v=")[-1].split("&")[0]
        
        # Fetch transcript
        transcript, error = get_transcript(vid_id)
        if error:
            return f"Error getting transcript: {error}"
        
        # Limit transcript context to first 3000 chars
        context = transcript[:3000]
        
        # Format prompt using the template
        formatted_prompt = youtube_prompt_template.format(context=context, question=question)
        
        # Invoke LLM
        return hf_model.invoke([HumanMessage(content=formatted_prompt)]).content
    except Exception as e:
        return f"Error processing video: {str(e)}"

# Initialize search tools
duck_tool = DuckDuckGoSearchRun()
tavily_tool = TavilySearch()

# ─────────────────────────────────────────────────────────────────────────────
# Specialized Agents
# ─────────────────────────────────────────────────────────────────────────────
class EducationAgent:
    def __init__(self, model):
        self.model = model
        self.tools = [topic_explanation, youtube_search]
        
    def process(self, query: str) -> str:
        """Handle educational queries with explanations and video resources."""
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

class ResearchAgent:
    def __init__(self, model):
        self.model = model
        self.tools = [semantic_scholar_research, youtube_search]
        
    def process(self, query: str) -> str:
        """Handle research queries with academic papers and videos."""
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

class ResumeAgent:
    def __init__(self, model):
        self.model = model
        self.tools = [generate_resume]
        
    def process(self, query: str, job_desc: str = "", candidate_info: str = "") -> str:
        """Handle resume creation and optimization."""
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

### Tips:
- Ensure all information is accurate
- Tailor keywords to match job description
- Proofread before submitting
"""

class NewsAgent:
    def __init__(self, model):
        self.model = model
        self.tools = [tavily_tool, duck_tool]
        
    def process(self, query: str) -> str:
        """Handle news and current affairs queries."""
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
        """Handle video-specific queries."""
        if video_url:
            # Direct video analysis
            analysis = youtube_qa.invoke({"video_url": video_url, "question": query})
            return f"## Video Analysis\n\n**Question**: {query}\n\n**Answer**: {analysis}"
        else:
            # Search for relevant videos
            videos = youtube_search.invoke({"query": query})
            return f"## Video Resources for: {query}\n\n{videos}"

# ─────────────────────────────────────────────────────────────────────────────
# Query Analysis and Routing System
# ─────────────────────────────────────────────────────────────────────────────
class QueryAnalyzer:
    def __init__(self, model):
        self.model = model
        
    def analyze_intent(self, query: str) -> Dict[str, any]:
        """Use LLM to analyze query intent and extract components."""
        analysis_prompt = f"""
        Analyze this user query and respond with a JSON-like structure:
        
        Query: "{query}"
        
        Determine:
        1. Primary intent (education, research, resume, news, video_analysis, general)
        2. Sub-queries if it's a multi-part query
        3. Specific requirements (video_url if present, job_description, etc.)
        4. Complexity level (simple, medium, complex)
        
        Format your response as:
        Intent: [primary intent]
        Sub_queries: [list of sub-queries if multi-part, otherwise just the main query]
        Requirements: [any specific requirements]
        Complexity: [complexity level]
        Video_URL: [if YouTube URL present, extract it]
        """
        
        response = self.model.invoke([HumanMessage(content=analysis_prompt)]).content
        return self._parse_analysis(response, query)
    
    def _parse_analysis(self, response: str, original_query: str) -> Dict[str, any]:
        """Parse the LLM analysis response."""
        intent_patterns = {
            "education": ["explain", "tutorial", "learn", "teach", "concept", "difference"],
            "research": ["research", "paper", "academic", "study", "journal"],
            "resume": ["resume", "cv", "job", "application"],
            "news": ["news", "latest", "current", "trend", "update"],
            "video_analysis": ["video", "youtube", "watch", "summarize video"]
        }
        
        # Extract intent from response or fallback to pattern matching
        intent = "general"
        for key, patterns in intent_patterns.items():
            if any(pattern in response.lower() or pattern in original_query.lower() for pattern in patterns):
                intent = key
                break
        
        # Extract video URL if present
        video_url = None
        video_match = re.search(r'https?://(?:www\.)?youtube\.com/watch\?v=([^&\s]+)', original_query)
        if video_match:
            video_url = video_match.group(0)
            intent = "video_analysis"
        
        # Split multi-part queries
        sub_queries = re.split(r'[;|\n]|and also|also', original_query)
        sub_queries = [q.strip() for q in sub_queries if q.strip()]
        
        return {
            "intent": intent,
            "sub_queries": sub_queries if len(sub_queries) > 1 else [original_query],
            "video_url": video_url,
            "complexity": "complex" if len(sub_queries) > 1 else "simple"
        }

# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical Supervisor System
# ─────────────────────────────────────────────────────────────────────────────
class MultiAgentSupervisor:
    def __init__(self, model):
        self.model = model
        self.analyzer = QueryAnalyzer(model)
        
        # Initialize specialized agents
        self.agents = {
            "education": EducationAgent(model),
            "research": ResearchAgent(model),
            "resume": ResumeAgent(model),
            "news": NewsAgent(model),
            "video_analysis": VideoAnalysisAgent(model)
        }
        
    def process_query(self, query: str) -> str:
        """Main entry point for processing any query."""
        # Analyze the query
        analysis = self.analyzer.analyze_intent(query)
        
        # Handle multi-part queries
        if len(analysis["sub_queries"]) > 1:
            return self._process_multi_query(analysis)
        else:
            return self._process_single_query(analysis)
    
    def _process_single_query(self, analysis: Dict) -> str:
        """Process a single query."""
        intent = analysis["intent"]
        query = analysis["sub_queries"][0]
        
        if intent in self.agents:
            agent = self.agents[intent]
            if intent == "video_analysis" and analysis.get("video_url"):
                return agent.process(query, analysis["video_url"])
            else:
                return agent.process(query)
        else:
            # Fallback to general response
            return self._general_response(query)
    
    def _process_multi_query(self, analysis: Dict) -> str:
        """Process multiple sub-queries."""
        responses = []
        
        for i, sub_query in enumerate(analysis["sub_queries"], 1):
            # Re-analyze each sub-query
            sub_analysis = self.analyzer.analyze_intent(sub_query)
            response = self._process_single_query(sub_analysis)
            
            responses.append(f"## Part {i}: {sub_query}\n{response}\n")
        
        final_response = "# Multi-Part Query Response\n\n" + "\n---\n\n".join(responses)
        
        # Add synthesis if needed
        if len(responses) > 2:
            synthesis = self._synthesize_responses(analysis["sub_queries"], responses)
            final_response += f"\n---\n\n## Synthesis\n{synthesis}"
        
        return final_response
    
    def _synthesize_responses(self, queries: List[str], responses: List[str]) -> str:
        """Synthesize multiple responses into a coherent summary."""
        synthesis_prompt = f"""
        The user asked multiple related questions:
        {queries}
        
        Provide a brief synthesis that connects these topics and highlights:
        1. Common themes
        2. How these topics relate to each other
        3. Key takeaways
        
        Keep it concise but insightful.
        """
        
        return self.model.invoke([HumanMessage(content=synthesis_prompt)]).content
    
    def _general_response(self, query: str) -> str:
        """Fallback for general queries."""
        prompt = f"""
        Provide a helpful response to this query: {query}
        
        If you need more specific information or tools, suggest what the user should specify.
        """
        return self.model.invoke([HumanMessage(content=prompt)]).content

# ─────────────────────────────────────────────────────────────────────────────
# Main Interface
# ─────────────────────────────────────────────────────────────────────────────
class IntelligentAssistant:
    def __init__(self, model):
        self.supervisor = MultiAgentSupervisor(model)
    
    def process(self, query: str) -> str:
        """Main interface for processing user queries."""
        try:
            response = self.supervisor.process_query(query)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error processing your request: {str(e)}\nPlease try rephrasing your query or contact support if the issue persists."

# ─────────────────────────────────────────────────────────────────────────────
# Usage Example
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize the assistant
    assistant = IntelligentAssistant(hf_model)
    
    # Example queries
    test_queries = [
        "Explain attention mechanism in transformers and show me related videos",
        "Create a resume for a software engineer position; Also explain what skills are most important for this role",
        "What are the latest AI trends today?",
        "Summarize this video https://www.youtube.com/watch?v=dQw4w9WgXcQ and explain the key concepts",
        "Find research papers about quantum computing and explain the basic concepts"
    ]
    
    # Process queries
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        response = assistant.process(query)
        print(response)
        print("\n")
