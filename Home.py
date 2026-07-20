import streamlit as st
import requests
import pandas as pd
import hashlib
from transformers import pipeline
import yake
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Newsie", layout="wide")

NEWS_API_KEY = st.secrets["auth_token"]
TOP_HEADLINES_ENDPOINT = "https://newsapi.org/v2/top-headlines"
EVERYTHING_ENDPOINT = "https://newsapi.org/v2/everything"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

CATEGORIES = ["general", "business", "entertainment", "health", "science", "sports", "technology"]
SIMILARITY_THRESHOLD = 0.30
SENTIMENT_COLOR = {"Positive": "green", "Negative": "red", "Neutral": "gray"}


@st.cache_resource(show_spinner="Loading RoBERTa sentiment model...")
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL)


@st.cache_resource
def get_keyword_extractor():
    return yake.KeywordExtractor(lan="en", n=2, dedupLim=0.3, top=5)


@st.cache_data(ttl=300, show_spinner=False)
def get_news(category: str):
    """Fetch news from News API. Returns (articles, error_message)."""
    params = {
        "country": "us",
        "category": category,
        "apiKey": NEWS_API_KEY,
        "pageSize": 50,
    }
    try:
        response = requests.get(TOP_HEADLINES_ENDPOINT, params=params, timeout=10)
    except requests.exceptions.RequestException as e:
        return [], f"Network error reaching NewsAPI: {e}"

    if response.status_code == 401:
        return [], "Invalid API key. Please check your NewsAPI key in secrets."
    if response.status_code == 426:
        return [], "NewsAPI upgrade required for this request."
    if response.status_code == 429:
        return [], "Rate limit hit — you've made too many requests. Try again shortly."
    if response.status_code != 200:
        return [], f"NewsAPI returned an error (status {response.status_code})."

    data = response.json()
    if data.get("status") != "ok":
        return [], data.get("message", "Unknown error fetching news.")

    return data.get("articles", []), None


@st.cache_data(ttl=1800, show_spinner=False)
def find_more_coverage(query_text: str, published_at: str, exclude_domains: str = ""):
    """
    Use /v2/everything to actively search for more coverage of a specific
    story, since /v2/top-headlines rarely returns multiple outlets on the
    same event. Searches a window around the story's publish date.
    """
    try:
        pub_dt = dateparser.parse(published_at)
    except Exception:
        pub_dt = datetime.utcnow()

    from_date = (pub_dt - timedelta(days=3)).strftime("%Y-%m-%d")
    to_date = (pub_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "q": query_text[:490],
        "from": from_date,
        "to": to_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 15,
        "apiKey": NEWS_API_KEY,
    }
    if exclude_domains.strip():
        params["excludeDomains"] = exclude_domains.strip()

    try:
        response = requests.get(EVERYTHING_ENDPOINT, params=params, timeout=10)
    except requests.exceptions.RequestException:
        return [], "Couldn't reach NewsAPI to search for more coverage."

    if response.status_code != 200:
        return [], None

    data = response.json()
    if data.get("status") != "ok":
        return [], None

    return data.get("articles", []), None


@st.cache_data(show_spinner=False)
def analyze_sentiment(text: str):
    """Analyze sentiment with RoBERTa. Returns (label, signed_score, confidence). Cached per unique text."""
    model = get_sentiment_pipeline()
    result = model(text[:512])[0]
    label = result["label"].capitalize()
    confidence = result["score"]

    if label == "Positive":
        signed_score = confidence
    elif label == "Negative":
        signed_score = -confidence
    else:
        signed_score = 0.0

    return label, signed_score, confidence


@st.cache_data(show_spinner=False)
def extract_keywords(text: str):
    kw_extractor = get_keyword_extractor()
    return [kw[0] for kw in kw_extractor.extract_keywords(text)]


def friendly_date(iso_str: str) -> str:
    try:
        dt = dateparser.parse(iso_str)
        return dt.strftime("%b %d, %Y at %I:%M %p")
    except Exception:
        return iso_str or "Unknown date"


def article_id(article: dict) -> str:
    key = article.get("url") or article.get("title") or ""
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def sentiment_label_from_score(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"


def cluster_into_stories(articles: list):
    """
    Groups articles covering the same event using TF-IDF + cosine similarity.
    Returns (list_of_clusters, similarity_matrix), where each cluster is a
    list of indices into `articles`.
    """
    n = len(articles)
    if n == 0:
        return [], np.zeros((0, 0))
    if n == 1:
        return [[0]], np.array([[1.0]])

    texts = [f"{a['title']} {a.get('description', '')}" for a in articles]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    try:
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
    except ValueError:
        return [[i] for i in range(n)], np.eye(n)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= SIMILARITY_THRESHOLD:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values()), sim_matrix


def build_stories(articles: list, clusters: list, sim_matrix: np.ndarray) -> list:
    """Turn raw clusters into story summary dicts with aggregate sentiment."""
    stories = []
    for member_indices in clusters:
        members = [articles[i] for i in member_indices]

        lead_local_idx = max(
            range(len(members)),
            key=lambda k: len(members[k]["title"]) + len(members[k].get("description", "")),
        )
        lead_global_idx = member_indices[lead_local_idx]
        lead = members[lead_local_idx]

        scores = [m["score"] for m in members]
        avg_score = sum(scores) / len(scores)
        spread = float(np.std(scores)) if len(scores) > 1 else 0.0

        latest_dt = None
        for m in members:
            try:
                dt = dateparser.parse(m.get("publishedAt", ""))
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
            except Exception:
                pass

        stories.append({
            "story_id": article_id(lead),
            "lead": lead,
            "lead_global_idx": lead_global_idx,
            "member_indices": member_indices,
            "members": members,
            "size": len(members),
            "avg_score": avg_score,
            "avg_sentiment": sentiment_label_from_score(avg_score),
            "spread": spread,
            "latest_dt": latest_dt or datetime.min,
        })

    stories.sort(key=lambda s: s["latest_dt"], reverse=True)
    return stories


st.markdown("""
<style>
.newsie-title {
    font-weight: 700;
    font-size: 2.7rem;
    line-height: 1.1;
    background: linear-gradient(90deg, #4F46E5, #C026D3 55%, #F97316);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    display: inline-block;
    margin-bottom: 0.1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="newsie-title">Newsie</div>', unsafe_allow_html=True)
st.caption("Real-time headlines, de-duplicated into stories, with sentiment analysis powered by RoBERTa.")

if "view" not in st.session_state:
    st.session_state.view = "list"
if "selected_story_id" not in st.session_state:
    st.session_state.selected_story_id = None

with st.sidebar:
    st.title("Settings")
    category = st.selectbox("Category", CATEGORIES, index=0)

    st.divider()
    exclude_domains = st.text_input(
        "Exclude domains (optional)",
        placeholder="e.g. dailymail.co.uk, tabloid.com",
        help="Applied when searching for extra coverage on a story's detail page.",
    )

    st.divider()
    st.title("About")
    st.info("Headlines are grouped into stories using text similarity, so one trending event doesn't flood your feed. Sentiment shown on the list is the story's average tone. Open a story to search for more coverage of it across other outlets. Currently limited to US headlines — NewsAPI's free tier only supports US top-headlines.")

col1, col2, col3 = st.columns(3)
with col1:
    sentiment_filter = st.selectbox("Filter by sentiment", ["All", "Positive", "Negative", "Neutral"])
with col2:
    sentiment_threshold = st.slider("Confidence threshold", 0.5, 1.0, 0.6, 0.05)
with col3:
    topic_filter = st.text_input("Filter by keyword (optional)")

with st.spinner("Fetching latest headlines..."):
    raw_articles, error = get_news(category)

if error:
    st.error(f"{error}")
    st.stop()

if not raw_articles:
    st.info("No headlines came back for this category right now. Try a different one!")
    st.stop()

processed_articles = []
with st.spinner("Analyzing sentiment..."):
    for article in raw_articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        source_name = (article.get("source") or {}).get("name") or ""

        if title.strip() in ("", "[Removed]") or source_name.strip() in ("", "[Removed]"):
            continue
        if "google news" in source_name.lower():
            continue
        if not article.get("url"):
            continue

        text_to_analyze = f"{title} {description}".strip()
        if len(text_to_analyze) < 12:
            continue

        sentiment, score, confidence = analyze_sentiment(text_to_analyze)
        if confidence < sentiment_threshold:
            continue

        processed_articles.append({
            **article,
            "title": title,
            "description": description,
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "keywords": extract_keywords(description) if description else [],
        })

if not processed_articles:
    st.warning("Nothing passed the confidence threshold — try lowering it.")
    st.stop()

clusters, sim_matrix = cluster_into_stories(processed_articles)
all_stories = build_stories(processed_articles, clusters, sim_matrix)
stories_by_id = {s["story_id"]: s for s in all_stories}

if st.session_state.view == "detail" and st.session_state.selected_story_id in stories_by_id:
    story = stories_by_id[st.session_state.selected_story_id]

    if st.button("← Back to headlines"):
        st.session_state.view = "list"
        st.rerun()

    st.header(story["lead"]["title"])

    existing_urls = {m.get("url") for m in story["members"]}
    with st.spinner("Searching for more coverage of this story..."):
        found_articles, enrich_error = find_more_coverage(
            story["lead"]["title"], story["lead"].get("publishedAt", ""), exclude_domains
        )

    enriched_members = []
    if found_articles:
        for a in found_articles:
            title = a.get("title") or ""
            source_name = (a.get("source") or {}).get("name") or ""
            if a.get("url") in existing_urls or title.strip() in ("", "[Removed]"):
                continue
            if "google news" in source_name.lower():
                continue
            existing_urls.add(a["url"])
            text = f"{title} {a.get('description') or ''}".strip()
            if len(text) < 12:
                continue
            sentiment, score, confidence = analyze_sentiment(text)
            enriched_members.append({
                **a,
                "title": title,
                "description": a.get("description") or "",
                "sentiment": sentiment,
                "score": score,
                "confidence": confidence,
            })

    all_members = story["members"] + enriched_members
    combined_scores = [m["score"] for m in all_members]
    combined_avg = sum(combined_scores) / len(combined_scores)
    combined_spread = float(np.std(combined_scores)) if len(combined_scores) > 1 else 0.0
    combined_sentiment = sentiment_label_from_score(combined_avg)

    total_sources = len(all_members)
    st.caption(
        f"{total_sources} source{'s' if total_sources != 1 else ''} covering this story"
        + (f" ({len(enriched_members)} found via broader search)" if enriched_members else "")
    )

    agree_note = "Sources are closely aligned in tone." if combined_spread < 0.15 else \
                 "Sources are somewhat split in tone." if combined_spread < 0.35 else \
                 "Sources diverge significantly in tone — worth reading a few to compare framing."
    color = SENTIMENT_COLOR[combined_sentiment]
    st.markdown(
        f"**Overall tone:** <span style='color:{color}'>{combined_sentiment}</span> "
        f"(avg score: {combined_avg:.3f}) — {agree_note}",
        unsafe_allow_html=True,
    )
    if enrich_error:
        st.caption(f"{enrich_error}")
    st.markdown("---")

    members_sorted = sorted(
        zip(story["member_indices"], story["members"]),
        key=lambda pair: sim_matrix[story["lead_global_idx"]][pair[0]],
        reverse=True,
    )

    def render_article_card(member, sim_pct=None, is_lead=False, found_via_search=False):
        source_name = member.get("source", {}).get("name", "Unknown source")
        m_color = SENTIMENT_COLOR[member["sentiment"]]
        with st.container():
            img_col, text_col = st.columns([1, 3])
            with img_col:
                if member.get("urlToImage"):
                    try:
                        st.image(member["urlToImage"], use_container_width=True)
                    except Exception:
                        st.write("(image unavailable)")
                else:
                    st.write("(no image)")
            with text_col:
                st.subheader(member["title"])
                st.caption(f"{source_name} · {friendly_date(member.get('publishedAt', ''))}")
                if sim_pct is not None and not is_lead:
                    st.caption(f"🔗 {sim_pct:.0f}% textual match to lead article")
                if found_via_search:
                    st.caption("🔎 Found via broader search")
                st.markdown(
                    f"**Sentiment:** <span style='color:{m_color}'>{member['sentiment']}</span> "
                    f"(score: {member['score']:.3f}, {member['confidence']:.0%} confidence)",
                    unsafe_allow_html=True,
                )
                if member["description"]:
                    st.markdown(f"*{member['description']}*")
                st.markdown(f"[Read full article →]({member['url']})")
            st.markdown("---")

    for global_idx, member in members_sorted:
        sim_pct = sim_matrix[story["lead_global_idx"]][global_idx] * 100
        render_article_card(member, sim_pct=sim_pct, is_lead=(global_idx == story["lead_global_idx"]))

    if enriched_members:
        st.subheader("More coverage found via search")
        for member in enriched_members:
            render_article_card(member, found_via_search=True)

    st.stop()

stories = all_stories
if sentiment_filter != "All":
    stories = [s for s in stories if s["avg_sentiment"] == sentiment_filter]
if topic_filter:
    stories = [
        s for s in stories
        if any(topic_filter.lower() in m["title"].lower() for m in s["members"])
    ]

st.write(f"**Showing {len(stories)} stories** (from {len(processed_articles)} articles) matching your filters")

if not stories:
    st.warning("Nothing matches these filters right now — try loosening the confidence threshold or clearing the keyword filter.")
    st.stop()

export_df = pd.DataFrame([{
    "story_title": s["lead"]["title"],
    "sources_covering": s["size"],
    "avg_sentiment": s["avg_sentiment"],
    "avg_score": s["avg_score"],
    "lead_url": s["lead"]["url"],
} for s in stories])
st.download_button(
    "Download story list as CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"newsie_stories_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)

st.markdown("---")

for story in stories:
    lead = story["lead"]
    with st.container():
        img_col, text_col = st.columns([1, 3])
        with img_col:
            if lead.get("urlToImage"):
                try:
                    st.image(lead["urlToImage"], use_container_width=True)
                except Exception:
                    st.write("(image unavailable)")
            else:
                st.write("(no image)")

        with text_col:
            st.subheader(lead["title"])
            source_name = lead.get("source", {}).get("name", "Unknown source")
            badge = f" · {story['size']} sources" if story["size"] >= 3 else \
                    (f" · {story['size']} sources" if story["size"] > 1 else "")
            st.caption(f"{source_name} · {friendly_date(lead.get('publishedAt', ''))}{badge}")

            color = SENTIMENT_COLOR[story["avg_sentiment"]]
            score_label = "Story tone" if story["size"] > 1 else "Sentiment"
            st.markdown(
                f"**{score_label}:** <span style='color:{color}'>{story['avg_sentiment']}</span> "
                f"(avg score: {story['avg_score']:.3f})",
                unsafe_allow_html=True,
            )

            if lead["keywords"]:
                st.markdown("**Keywords:** " + ", ".join(lead["keywords"]))
            if lead["description"]:
                st.markdown(f"*{lead['description']}*")

            btn_label = f"View full coverage ({story['size']} sources) →" if story["size"] > 1 else "View story →"
            if st.button(btn_label, key=f"story_{story['story_id']}"):
                st.session_state.view = "detail"
                st.session_state.selected_story_id = story["story_id"]
                st.rerun()
        st.markdown("---")
