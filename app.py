# ========================================
# IMPORTS
# ========================================
import streamlit as st
import requests
import json
import time
import uuid
import threading
import logging
import pandas as pd
from datetime import datetime, timedelta
import re
import html
import weakref
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, Optional, List, Tuple, Union

# ========================================
# LOAD ENVIRONMENT VARIABLES
# ========================================
load_dotenv()

# ========================================
# LOGGING CONFIGURATION
# ========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION
# ========================================
def get_config():
    """Get configuration from environment or Streamlit secrets"""
    try:
        return {
            'MAX_CONCURRENT_USERS': int(st.secrets.get('MAX_CONCURRENT_USERS', 25)),
            'DEFAULT_MAX_COMMENTS': int(st.secrets.get('DEFAULT_MAX_COMMENTS', 100)),
            'MAX_SAFE_COMMENTS': 2000,
            'SESSION_TIMEOUT_MINUTES': 30,
            'YOUTUBE_API_KEY': st.secrets.get('YOUTUBE_API_KEY', ''),
            'GEMINI_API_KEY': st.secrets.get('GEMINI_API_KEY', ''),
            'REQUEST_TIMEOUT': (60, 1800),
            'DEBUG': st.secrets.get('DEBUG', 'False').lower() == 'true',
        }
    except:
        return {
            'MAX_CONCURRENT_USERS': int(os.getenv('MAX_CONCURRENT_USERS', 25)),
            'DEFAULT_MAX_COMMENTS': int(os.getenv('DEFAULT_MAX_COMMENTS', 100)),
            'MAX_SAFE_COMMENTS': 2000,
            'SESSION_TIMEOUT_MINUTES': 30,
            'YOUTUBE_API_KEY': os.getenv('YOUTUBE_API_KEY', ''),
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', ''),
            'REQUEST_TIMEOUT': (60, 1800),
            'DEBUG': os.getenv('DEBUG', 'False').lower() == 'true',
        }

APP_CONFIG = get_config()

# ========================================
# INITIALIZE GEMINI CLIENT
# ========================================
if APP_CONFIG['GEMINI_API_KEY']:
    genai.configure(api_key=APP_CONFIG['GEMINI_API_KEY'])

# ========================================
# MULTIUSER SESSION MANAGEMENT
# ========================================
class MultiUserSessionManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.active_sessions = weakref.WeakValueDictionary()
            self.session_data = {}
            self.session_locks = {}
            self.max_users = APP_CONFIG['MAX_CONCURRENT_USERS']
            self.cleanup_interval = 300
            self.last_cleanup = time.time()
            self._initialized = True
    
    def create_session(self) -> str:
        self._cleanup_expired_sessions()
        with self._lock:
            if len(self.active_sessions) >= self.max_users:
                raise Exception("🚫 Server at capacity. Please try again later.")
            
            session_id = str(uuid.uuid4())
            self.session_data[session_id] = {
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'analysis_running': False,
                'errors': []
            }
            self.session_locks[session_id] = threading.Lock()
            logger.info(f"New session: {session_id[:8]}")
            return session_id
    
    def update_session_activity(self, session_id: str):
        if session_id in self.session_data:
            self.session_data[session_id]['last_activity'] = datetime.now()
    
    def set_analysis_status(self, session_id: str, running: bool):
        if session_id in self.session_data:
            self.session_data[session_id]['analysis_running'] = running
    
    def record_error(self, session_id: str, error: str):
        if session_id in self.session_data:
            self.session_data[session_id]['errors'].append({
                'timestamp': datetime.now(),
                'error': error
            })
    
    def _cleanup_expired_sessions(self):
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        expired_threshold = datetime.now() - timedelta(minutes=APP_CONFIG['SESSION_TIMEOUT_MINUTES'])
        expired = [sid for sid, d in self.session_data.items() if d['last_activity'] < expired_threshold]
        for sid in expired:
            self.session_data.pop(sid, None)
            self.session_locks.pop(sid, None)
        self.last_cleanup = current_time

session_manager = MultiUserSessionManager()


# ========================================
# ENHANCED YOUTUBE COMMENT PROCESSOR
# ========================================
class EnhancedYouTubeProcessor:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.youtube_api_key = APP_CONFIG['YOUTUBE_API_KEY']
        self.session = requests.Session()
        self.session.timeout = APP_CONFIG['REQUEST_TIMEOUT']
        
        if not self.youtube_api_key:
            raise ValueError("YouTube API Key is required! Set YOUTUBE_API_KEY in environment variables or config.")

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        if not url or not isinstance(url, str):
            raise ValueError("Invalid URL")

        url = url.strip()
        patterns = [
            r'(?:youtube\.com/watch\?v=)([^&\n]+)',
            r'(?:youtu\.be/)([^&\n]+)',
            r'(?:youtube\.com/embed/)([^&\n]+)',
            r'(?:youtube\.com/shorts/)([^&\n]+)',
            r'(?:youtube\.com/v/)([^&\n]+)',
            r'(?:youtube\.com/live/)([^&\n]+)'
        ]
        for p in patterns:
            m = re.search(p, url, re.IGNORECASE)
            if m:
                vid = m[1].split('&')[0]
                if len(vid) == 11:
                    return vid
        if len(url) == 11:
            return url
        return None

    def get_video_info(self, video_id: str) -> Dict:
        """Get metadata of a video"""
        try:
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                'part': 'snippet,statistics',
                'id': video_id,
                'key': self.youtube_api_key
            }
            r = self.session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                d = r.json()
                if d.get('items'):
                    video = d['items'][0]
                    snip = video.get('snippet', {})
                    stats = video.get('statistics', {})
                    return {
                        'video_id': video_id,
                        'video_title': snip.get('title', 'Unknown'),
                        'channel_name': snip.get('channelTitle', 'Unknown'),
                        'channel_id': snip.get('channelId', ''),
                        'published_at': snip.get('publishedAt', ''),
                        'description': snip.get('description', ''),
                        'view_count': int(stats.get('viewCount', 0)),
                        'like_count': int(stats.get('likeCount', 0)),
                        'comment_count': int(stats.get('commentCount', 0)),
                        'video_url': f"https://youtube.com/watch?v={video_id}"
                    }
                else:
                    raise ValueError("Video not found or deleted")
            else:
                raise ValueError(f"YouTube API error {r.status_code}")
        except Exception as e:
            logger.error(f"Video info error: {e}")
            return {'video_id': video_id, 'video_title': 'Unknown', 'error': str(e)}

    def fetch_youtube_comments(self, video_id: str, max_comments: int = 100) -> Dict:
        """Fetch top-level comments via YouTube API"""
        try:
            video_info = self.get_video_info(video_id)
            if 'error' in video_info:
                return {'success': False, 'error': video_info['error'], 'video_info': video_info}
            
            comments, next_page, calls = [], None, 0
            while len(comments) < max_comments and calls < 20:
                params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'maxResults': min(100, max_comments - len(comments)),
                    'order': 'relevance',
                    'key': self.youtube_api_key
                }
                if next_page:
                    params['pageToken'] = next_page
                r = self.session.get(
                    "https://www.googleapis.com/youtube/v3/commentThreads",
                    params=params,
                    timeout=30
                )
                calls += 1
                if r.status_code == 200:
                    d = r.json()
                    for item in d.get('items', []):
                        if len(comments) >= max_comments:
                            break
                        snip = item['snippet']['topLevelComment']['snippet']
                        comments.append({
                            'comment_id': item['id'],
                            'author': snip.get('authorDisplayName', 'Unknown'),
                            'comment_text': html.unescape(snip.get('textDisplay', '')),
                            'full_text': html.unescape(snip.get('textOriginal', '')),
                            'like_count': int(snip.get('likeCount', 0)),
                            'published_at': snip.get('publishedAt', '')
                        })
                    next_page = d.get('nextPageToken')
                    if not next_page:
                        break
                elif r.status_code == 403:
                    return {
                        'success': False,
                        'error': "Comments disabled or quota exceeded",
                        'video_info': video_info
                    }
                else:
                    logger.error(f"Comment fetch failed {r.status_code}")
                    break
            
            valid = [c for c in comments if c['comment_text'].strip()]
            return {
                'success': True,
                'comments': valid,
                'video_info': video_info,
                'total_fetched': len(valid),
                'api_requests_made': calls
            }
        except Exception as e:
            logger.error(f"Comment fetch error: {e}")
            return {'success': False, 'error': str(e), 'comments': []}
# ========================================
# LLM ANALYZER - GEMINI
# ========================================
import google.generativeai as genai

def setup_gemini():
    """Configure Gemini with API key"""
    api_key = APP_CONFIG.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Gemini API Key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    genai.configure(api_key=api_key)

def analyze_comments_with_llm(video_info: Dict, comments: list, custom_prompt: str = None) -> Dict:
    """
    Send YouTube comments to Gemini and get structured analysis in 
    'Video Comment Analysis Report' format or custom analysis.
    """
    try:
        setup_gemini()

        # Prepare comment block
        comments_text = "\n".join(
            [f"- {c['author']}: {c['comment_text']}" for c in comments[:200]]
        )

        # Use custom prompt if provided, otherwise use default
        if custom_prompt:
            prompt = f"""
Video Title: "{video_info.get('video_title', 'Unknown')}"
Channel: {video_info.get('channel_name', 'Unknown')}
Total Comments Analyzed: {len(comments)}
Video Stats - Views: {video_info.get('view_count', 0):,} | Likes: {video_info.get('like_count', 0):,}

{custom_prompt}

Here are the comments to analyze:
{comments_text}
"""
        else:
            prompt = f"""
You are an expert EdTech comment analyst. Analyze the following YouTube video comments
and return a structured report in EXACTLY this format:

CONTEXT: The video title is "{video_info.get('video_title', 'Unknown')}" - use this to understand what subject/topic is being taught.

Video Comment Analysis Report


Video: "{video_info.get('video_title', 'Unknown')}" | Channel: {video_info.get('channel_name', 'Unknown')}
Total Comments: {len(comments)} | Views: {video_info.get('view_count', 0):,} | Likes: {video_info.get('like_count', 0):,} 
Analysis Date: {datetime.now().strftime("%B %d, %Y")}
Analysis Confidence: Estimate percentage confidence in insights
Overall Sentiment Score: (scale 1–10 with justification)
Positive: X% (count) | Neutral: X% (count) | Negative: X% (count)

Top Positive Themes:
- Theme (mentions) – example comment

Top Negative/Concern Themes:
- Theme (mentions) – example comment

Engagement Quality:
- Key bullet points about engagement style, tone, and interaction

👨‍🏫 TEACHER PERFORMANCE

What Students Appreciate:
- List key positive feedback about teaching style, explanations, or content

Areas for Improvement:
- List constructive feedback or concerns mentioned by students

💡 ACTIONABLE INSIGHTS
🚨 HIGH PRIORITY ACTIONS
Format each action as:
Problem: [Specific issue from comments] (X comments) → Impact: [Why this matters and consequences] → Action: [Specific steps to address it]

IMPORTANT: Count and include the number of comments that mention or relate to each problem in parentheses.

Examples:
1. Problem: Students perceive unfairness in winner selection (15 comments) → Impact: Reduces engagement and damages credibility → Action: Explain selection criteria publicly (speed, accuracy, completeness)
2. Problem: Content discrepancies with NCERT causing confusion (8 comments) → Impact: Undermines teacher credibility and hinders learning → Action: Review specific NCERT content, provide corrections or explanations

⚠️ MEDIUM PRIORITY ACTIONS
Format each action as:
Problem: [Specific issue] (X comments) → Impact: [Consequences] → Action: [Steps to take]

Now here are the comments to analyze:

{comments_text}
"""

        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)

        return {
            "success": True,
            "report_text": response.text
        }

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
# ========================================
# REPORT POLISHING
# ========================================

def polish_report_text(report_text: str) -> str:
    """
    Cleans and formats Gemini's plain text report so it renders beautifully in Streamlit.
    - Ensures headings are bold
    - Converts bullet points into Markdown lists
    - Adds spacing between sections
    """
    if not report_text:
        return "⚠️ No report generated."


    # Normalize line breaks
    text = report_text.replace("\r\n", "\n").strip()


    # Replace markers with Streamlit-friendly markdown
    text = text.replace("📊 SENTIMENT ANALYSIS", "### 📊 Sentiment Analysis")
    text = text.replace("👨‍🏫 TEACHER PERFORMANCE", "### 👨‍🏫 Teacher Performance")
    text = text.replace("💡 ACTIONABLE INSIGHTS", "### 💡 Actionable Insights")
    text = text.replace("🚨 HIGH PRIORITY ACTIONS", "#### 🚨 High Priority Actions")
    text = text.replace("⚠️ MEDIUM PRIORITY ACTIONS", "#### ⚠️ Medium Priority Actions")
    text = text.replace("What Students Appreciate:", "**What Students Appreciate:**")
    text = text.replace("Areas for Improvement:", "**Areas for Improvement:**")
    
    # Highlight Problem, Impact, and Action
    text = text.replace("Problem:", "**Problem:**")
    text = text.replace("→ Impact:", "→ **Impact:**")
    text = text.replace("→ Action:", "→ **Action:**")


    # Add spacing after major sections
    text = re.sub(r"(### .+)", r"\1\n", text)
    text = re.sub(r"(#### .+)", r"\1\n", text)


    # Convert " - " to proper bullets
    text = text.replace(" - ", "\n- ")


    return text


# ========================================
# STREAMLIT PAGE CONFIG (MUST BE FIRST)
# ========================================
st.set_page_config(page_title="YouTube Comment Analysis", layout="wide")

# ========================================
# STREAMLIT UI - MAIN APP

def display_final_report(report_text: str):
    """Display the full analysis report returned by Gemini"""
    st.markdown("## 📄 Video Comment Analysis Report")
    # Polish the report before displaying
    polished_text = polish_report_text(report_text)
    st.markdown(polished_text)


def main():
    st.title("🎓 YouTube Comment Analyzer")

    # --- User Input ---
    video_url = st.text_input("Enter YouTube Video URL:", "")
    max_comments = st.number_input(
        "Max Comments to Analyze", 10, APP_CONFIG['MAX_SAFE_COMMENTS'], 100
    )

    # --- Analysis Type Selection ---
    st.markdown("### Analysis Type")
    analysis_type = st.radio(
        "Choose analysis type:",
        ["📚 Standard Teacher Performance Analysis", "🎯 Custom Analysis", "🔍 Analyze Previous Insights"],
        index=0
    )
    
    custom_prompt = None
    if analysis_type == "🎯 Custom Analysis":
        st.markdown("**Enter your custom analysis prompt:**")
        custom_prompt = st.text_area(
            "Describe what you want to analyze in the comments",
            placeholder="Example: Analyze the comments to find the most common questions students are asking about the topic. Group similar questions together and suggest answers for each group.",
            height=120
        )
        st.info("💡 Tip: Be specific about what insights you want from the comments!")
    
    elif analysis_type == "🔍 Analyze Previous Insights":
        if "last_analysis_report" in st.session_state and st.session_state.last_analysis_report:
            st.info("✅ Previous analysis found. Enter your custom prompt to analyze those insights deeper.")
            st.markdown("**What would you like to know about the previous analysis?**")
            custom_prompt = st.text_area(
                "Enter your question about the insights",
                placeholder="Example: Based on the teacher performance analysis, what are the top 3 most urgent changes the teacher should make? Or: Summarize all student concerns in order of frequency.",
                height=120
            )
        else:
            st.warning("⚠️ No previous analysis found. Please run a standard analysis first.")

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    with col2:
        if analysis_type == "🎯 Custom Analysis" and not custom_prompt:
            st.warning("Please enter a custom prompt")
    
    # View Previous Report Button
    if "stored_report" in st.session_state and st.session_state.stored_report:
        view_report_btn = st.button("📋 View Previous Report", use_container_width=True)
    else:
        view_report_btn = None
    
    if analyze_btn:
        try:
            # --- Create Session ---
            session_id = st.session_state.get("session_id")
            if not session_id:
                session_id = session_manager.create_session()
                st.session_state["session_id"] = session_id

            processor = EnhancedYouTubeProcessor(session_id)

            # --- Step 1: Fetch comments ---
            with st.spinner("⏳ Fetching comments from YouTube..."):
                result = processor.fetch_youtube_comments(
                    processor.extract_video_id(video_url), max_comments
                )

            if not result.get("success"):
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                return

            video_info = result["video_info"]
            comments = result["comments"]

            # --- Step 2: Run Gemini Analysis ---
            with st.spinner("🤖 Running Gemini analysis..."):
                if analysis_type == "🔍 Analyze Previous Insights":
                    if not custom_prompt:
                        st.error("Please provide a prompt for analyzing the insights")
                        return
                    if "last_analysis_report" not in st.session_state:
                        st.error("No previous analysis found. Run a standard analysis first.")
                        return
                    
                    # Analyze both previous insights AND original comments
                    setup_gemini()
                    
                    # Prepare comments text
                    comments_text = "\n".join([
                        f"- {c['author']}: {c['comment_text']}" 
                        for c in st.session_state.last_comments[:200]
                    ])
                    
                    video_info = st.session_state.last_video_info
                    
                    prompt = f"""
Video Information:
Title: {video_info.get('video_title', 'Unknown')}
Channel: {video_info.get('channel_name', 'Unknown')}
Views: {video_info.get('view_count', 0):,} | Likes: {video_info.get('like_count', 0):,}

Previous Analysis Report:
{st.session_state.last_analysis_report}

Original Comments for Reference:
{comments_text}

User's Question:
{custom_prompt}

Based on the previous analysis and original comments, please provide a DIRECT and CONCISE answer to the user's question.

REQUIREMENTS:
- Give a simple, focused answer - no long explanations
- Answer ONLY what was asked, nothing extra
- Use bullet points or numbered lists if listing multiple items
- Keep response under 5-7 sentences unless specifically asked for more detail
- Be specific and actionable
- Focus on constructive insights only
"""
                    model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    response = model.generate_content(prompt)
                    
                    st.markdown("## 🔍 Quick Answer")
                    st.markdown(response.text)
                    # Note: Quick answers are not stored in the report button since they are follow-up queries
                    return
                
                elif analysis_type == "🎯 Custom Analysis" and not custom_prompt:
                    st.error("Please provide a custom analysis prompt")
                    
                analysis = analyze_comments_with_llm(video_info, comments, custom_prompt)

            if analysis.get("success"):
                # Store the analysis and comments for potential re-analysis
                st.session_state.last_analysis_report = analysis["report_text"]
                st.session_state.last_video_info = video_info
                st.session_state.last_comments = comments
                
                if custom_prompt:
                    st.markdown("## 🎯 Custom Analysis Results")
                    st.markdown(analysis["report_text"])
                    # Store custom analysis
                    st.session_state.stored_report = analysis["report_text"]
                    st.session_state.stored_report_type = "custom"
                else:
                    display_final_report(analysis["report_text"])
                    # Store standard analysis
                    st.session_state.stored_report = analysis["report_text"]
                    st.session_state.stored_report_type = "standard"
            else:
                st.error(f"LLM Analysis Failed: {analysis.get('error')}")

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            if APP_CONFIG['DEBUG']:
                st.exception(e)

    # Display stored report when button is clicked
    if view_report_btn and "stored_report" in st.session_state:
        st.markdown("---")
        st.markdown("## 📋 Stored Report")
        if st.session_state.stored_report_type == "standard":
            display_final_report(st.session_state.stored_report)
        else:
            st.markdown(st.session_state.stored_report)
    
    # Placeholder Help Text
    if not video_url and not view_report_btn:
        st.info("👆 Enter a YouTube URL and click **Start Analysis**")


# ========================================
# RUN APP
# ========================================
if __name__ == "__main__":
    main()
