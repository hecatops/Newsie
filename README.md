# Newsie

Newsie is a Streamlit application that aggregates real-time news headlines, groups similar articles into stories, and analyzes their sentiment using RoBERTa. Instead of showing duplicate headlines, it presents a single story with coverage from multiple sources.

## Features

* Real-time news from NewsAPI
* Story clustering using TF-IDF and cosine similarity
* Sentiment analysis with RoBERTa
* Keyword extraction with YAKE
* Search for additional coverage
* Filter stories by sentiment or keyword
* Export stories as CSV

## Installation

```bash
git clone https://github.com/yourusername/newsie.git
cd newsie
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
auth_token = "YOUR_NEWSAPI_KEY"
```

Run the app:

```bash
streamlit run app.py
```

## How It Works

1. Fetches headlines from NewsAPI.
2. Analyzes article sentiment with RoBERTa.
3. Groups related articles using TF-IDF and cosine similarity.
4. Displays one story with links to coverage from multiple sources.
