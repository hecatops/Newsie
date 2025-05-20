import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import yake
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
import time
import numpy as np

st.set_page_config(page_title="Newsie", page_icon="📰", layout="wide")

NEWS_API_KEY = st.secrets["auth_token"]
NEWS_API_ENDPOINT = "https://newsapi.org/v2/top-headlines"
COUNTRY = "us"

# Initialize sentiment analyzers
analyzer = SentimentIntensityAnalyzer()
distilbert_analyzer = pipeline("sentiment-analysis")

kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.3, top=5)

@st.cache_data(ttl=300)
def get_news():
    """Fetch news from News API"""
    params = {"country": COUNTRY, "apiKey": NEWS_API_KEY}
    response = requests.get(NEWS_API_ENDPOINT, params=params)
    return response.json()

@st.cache_data()
def analyze_sentiment(text):
    """Analyze sentiment using VADER and DistilBERT"""
    vader_scores = analyzer.polarity_scores(text)
    bert_result = distilbert_analyzer(text[:512])[0]
    
    compound_score = vader_scores['compound']
    if compound_score >= 0.05:
        vader_sentiment = "Positive"
    elif compound_score <= -0.05:
        vader_sentiment = "Negative"
    else:
        vader_sentiment = "Neutral"
    
    bert_sentiment = bert_result['label']
    confidence = bert_result['score']
    
    return vader_sentiment, compound_score, bert_sentiment, confidence

def extract_keywords(text):
    """Extract keywords from text using YAKE"""
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

st.title("Newsie Time")

sentiment_filter = st.selectbox("Filter headlines by sentiment", ["All", "Positive", "Negative", "Neutral"])
sentiment_threshold = st.slider("Confidence threshold (BERT)", 0.5, 1.0, 0.7, 0.05)
topic_filter = st.text_input("Filter by keyword (optional)")

col1, col2 = st.columns([2, 1])
with col1:
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=True)
with col2:
    refresh = st.button("Refresh Now")

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

current_time = datetime.now()
should_refresh = refresh or (auto_refresh and (current_time - st.session_state.last_refresh).total_seconds() > 300)

if should_refresh:
    st.session_state.last_refresh = current_time
    with st.spinner("Fetching latest headlines..."):
        news_data = get_news()
        
        if news_data["status"] == "ok":
            processed_articles = []
            for article in news_data["articles"]:
                text_to_analyze = article["title"] + " " + (article["description"] or "")
                vader_sentiment, vader_score, bert_sentiment, confidence = analyze_sentiment(text_to_analyze)
                
                if confidence < sentiment_threshold:
                    continue
                
                processed_article = {
                    **article,
                    "vader_sentiment": vader_sentiment,
                    "vader_score": vader_score,
                    "bert_sentiment": bert_sentiment,
                    "confidence": confidence,
                    "keywords": extract_keywords(article["description"]) if article["description"] else []
                }
                processed_articles.append(processed_article)
            
            if sentiment_filter != "All":
                processed_articles = [
                    article for article in processed_articles 
                    if article["vader_sentiment"] == sentiment_filter or article["bert_sentiment"].lower() == sentiment_filter.lower()
                ]
            
            if topic_filter:
                processed_articles = [
                    article for article in processed_articles if topic_filter.lower() in article["title"].lower()
                ]
            
            st.write(f"Showing {len(processed_articles)} headlines matching criteria")
            
            all_keywords = [kw for article in processed_articles for kw in article["keywords"]]
            if all_keywords:
                wordcloud = WordCloud(width=600, height=300, background_color='white').generate(" ".join(all_keywords))
                st.image(wordcloud.to_array(), use_column_width=True)
            
            sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
            for article in processed_articles:
                sentiment_counts[article["vader_sentiment"]] += 1
            
            fig, ax = plt.subplots()
            ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red', 'gray'])
            st.pyplot(fig)

            for article in processed_articles:
                with st.container():
                    st.markdown("---")
                    st.subheader(article["title"])
                    st.text(f"Source: {article['source']['name']} | Published: {article['publishedAt']}")
                    sentiment_color = {"Positive": "green", "Negative": "red", "Neutral": "gray"}[article["vader_sentiment"]]
                    st.markdown(f"**Sentiment (VADER):** <span style='color: {sentiment_color}'>{article['vader_sentiment']}</span> (Score: {article['vader_score']:.3f})", unsafe_allow_html=True)
                    st.markdown(f"**Sentiment (BERT):** {article['bert_sentiment']} (Confidence: {article['confidence']:.2f})")
                    if article["keywords"]:
                        st.markdown("**Keywords:** " + ", ".join(article["keywords"]))
                    if article["description"]:
                        st.markdown(f"*{article['description']}*")
                    st.markdown(f"[Read full article]({article['url']})")
        else:
            st.error("Error fetching news. Please check your API key and try again.")
    
st.sidebar.text(f"Last refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

st.sidebar.title("About")
st.sidebar.info("This app fetches real-time news headlines and performs sentiment analysis using VADER and BERT. You can filter headlines by sentiment, confidence level, and keywords.")
