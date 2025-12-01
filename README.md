# YouTube Comment Analyzer - Streamlit Deployment

This is a YouTube comment analyzer that uses Gemini AI to analyze teaching performance from video comments.

## Features
- Fetches YouTube video comments
- Analyzes sentiment and teaching performance
- Provides actionable insights
- Simplified teacher performance summary

## Deployment on Streamlit Cloud

### 1. Prepare Your Repository
- Push this folder to a GitHub repository
- Make sure to NOT include .env files

### 2. Set Up Secrets in Streamlit Cloud
In your Streamlit app settings, go to Secrets and add:

```toml
# YouTube API Configuration
YOUTUBE_API_KEY = "your-youtube-api-key-here"

# Gemini API Configuration  
GEMINI_API_KEY = "your-gemini-api-key-here"

# App Configuration
MAX_CONCURRENT_USERS = 25
DEFAULT_MAX_COMMENTS = 100
DEBUG = "False"
```

### 3. Deploy
- Connect your GitHub repo to Streamlit Cloud
- Select `app.py` as the main file
- Deploy!

## Local Development

1. Create a `.env` file with your API keys:
```
YOUTUBE_API_KEY=your-key
GEMINI_API_KEY=your-key
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Required API Keys
- **YouTube Data API v3**: Get from [Google Cloud Console](https://console.cloud.google.com/)
- **Gemini API**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
