import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake
from datetime import datetime
import time

headers = {
    'Authorization": st.secrets["auth_token"],
    'Content-Type': 'application/json'
}

NEWS_API_ENDPOINT = "https://newsapi.org/v2/top-headlines"
COUNTRY = "us" 

analyzer = SentimentIntensityAnalyzer()

kw_extractor = yake.KeywordExtractor(
    lan="en",
    n=2,
    dedupLim=0.3,
    top=5,
)

def get_news():
    """Fetch news from News API"""
    params = {
        "country": COUNTRY,
        "apiKey": NEWS_API_KEY
    }
    
    response = requests.get(NEWS_API_ENDPOINT, params=params)
    return response.json()

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return "Positive", compound_score
    elif compound_score <= -0.05:
        return "Negative", compound_score
    else:
        return "Neutral", compound_score

def extract_keywords(text):
    """Extract keywords from text using YAKE"""
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords] 

st.title("Real-Time News Sentiment Analysis")

sentiment_filter = st.selectbox(
    "Filter headlines by sentiment",
    ["All", "Positive", "Negative", "Neutral"]
)

col1, col2 = st.columns([2, 1])
with col1:
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=True)
with col2:
    refresh = st.button("Refresh Now")

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

current_time = datetime.now()
should_refresh = (
    refresh or 
    (auto_refresh and (current_time - st.session_state.last_refresh).total_seconds() > 300)
)

if should_refresh:
    st.session_state.last_refresh = current_time
    
    with st.spinner("Fetching latest headlines..."):
        news_data = get_news()
        
        if news_data["status"] == "ok":
            processed_articles = []
            for article in news_data["articles"]:
                text_to_analyze = article["title"] + " " + (article["description"] or "")
                sentiment, score = analyze_sentiment(text_to_analyze)
                
                processed_article = {
                    **article,
                    "sentiment": sentiment,
                    "score": score,
                    "keywords": extract_keywords(article["description"]) if article["description"] else []
                }
                processed_articles.append(processed_article)
        
            if sentiment_filter != "All":
                processed_articles = [
                    article for article in processed_articles 
                    if article["sentiment"] == sentiment_filter
                ]

            st.write(f"Showing {len(processed_articles)} {sentiment_filter.lower() if sentiment_filter != 'All' else ''} headlines")

            for article in processed_articles:
                with st.container():
                    st.markdown("---")
 
                    st.subheader(article["title"])
                    st.text(f"Source: {article['source']['name']} | "
                           f"Published: {article['publishedAt']}")
                    
                    sentiment_color = {
                        "Positive": "green",
                        "Negative": "red",
                        "Neutral": "gray"
                    }[article["sentiment"]]
                    
                    st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}'>"
                              f"{article['sentiment']}</span> (Score: {article['score']:.3f})",
                              unsafe_allow_html=True)
     
                    if article["keywords"]:
                        st.markdown("**Keywords:** " + ", ".join(article["keywords"]))

                    if article["description"]:
                        st.markdown(f"*{article['description']}*")
  
                    st.markdown(f"[Read full article]({article['url']})")
        
        else:
            st.error("Error fetching news. Please check your API key and try again.")

    st.sidebar.text(f"Last refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

st.sidebar.title("About")
st.sidebar.info(
    "This app fetches real-time news headlines and performs sentiment analysis using VADER. You can filter headlines by sentiment and see detailed sentiment scores for each article."
)

st.sidebar.title("Sentiment Scores")
st.sidebar.markdown("""
- **Positive:** Score ≥ 0.05
- **Negative:** Score ≤ -0.05
- **Neutral:** -0.05 < Score < 0.05
""")