import streamlit as st
import pandas as pd
import re, nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="E-Commerce Recommendation", layout="wide")
st.title("🛒 E-Commerce Product Recommendation System")

# ---------- Load Dataset ----------
file = st.file_uploader("Upload Product Review CSV", type="csv")

if not file:
    st.info("Upload your CSV dataset to continue.")
    st.stop()

df = pd.read_csv(file)

# ---------- Automatically find columns ----------
def find_column(names):
    for c in df.columns:
        if c.lower().strip() in names:
            return c
    return None

product_col = find_column({
    "product", "product_name", "productname", "name",
    "title", "product_title", "item"
})

review_col = find_column({
    "review", "review_text", "reviewtext", "comment",
    "comments", "text", "description"
})

rating_col = find_column({
    "rating", "ratings", "score", "stars", "review_rating"
})

if not product_col or not review_col:
    st.error(
        "Dataset must contain a product column and a review column."
    )
    st.write("Available columns:", list(df.columns))
    st.stop()

# ---------- Clean Reviews ----------
df[review_col] = df[review_col].fillna("").astype(str)

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df["clean_review"] = df[review_col].apply(clean)

# ---------- Sentiment ----------
sia = SentimentIntensityAnalyzer()

df["sentiment"] = df["clean_review"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

# ---------- Product Summary ----------
summary = df.groupby(product_col).agg(
    Reviews=("clean_review", "count"),
    Sentiment=("sentiment", "mean")
).reset_index()

if rating_col:
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    ratings = df.groupby(product_col)[rating_col].mean().reset_index()
    summary = summary.merge(ratings, on=product_col)
    summary["Score"] = (
        (summary[rating_col] / 5) * 0.5 +
        ((summary["Sentiment"] + 1) / 2) * 0.5
    )
else:
    summary["Score"] = (summary["Sentiment"] + 1) / 2

# ---------- Product Review Documents ----------
products = df.groupby(product_col)["clean_review"].apply(
    lambda x: " ".join(x)
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf = vectorizer.fit_transform(products)

similarity = cosine_similarity(tfidf)

product_names = products.index.tolist()

# ---------- Product Selection ----------
selected = st.selectbox(
    "Select a product",
    product_names
)

index = product_names.index(selected)

# ---------- Recommendations ----------
results = pd.DataFrame({
    "Product": product_names,
    "Similarity": similarity[index]
})

results = results[results["Product"] != selected]

results = results.merge(
    summary[[product_col, "Score"]],
    left_on="Product",
    right_on=product_col
)

results["Recommendation Score"] = (
    results["Similarity"] * 0.6 +
    results["Score"] * 0.4
)

results = results.sort_values(
    "Recommendation Score",
    ascending=False
).head(5)

# ---------- Display ----------
st.subheader("⭐ Recommended Products")

show = results[
    ["Product", "Similarity", "Score", "Recommendation Score"]
].copy()

show.columns = [
    "Product",
    "Review Similarity",
    "Product Score",
    "Recommendation Score"
]

st.dataframe(
    show,
    use_container_width=True,
    hide_index=True
)

# ---------- Product Information ----------
selected_info = summary[
    summary[product_col] == selected
].iloc[0]

c1, c2, c3 = st.columns(3)

c1.metric("Reviews", int(selected_info["Reviews"]))
c2.metric("Sentiment", round(selected_info["Sentiment"], 2))
c3.metric("Product Score", round(selected_info["Score"] * 100, 1))

st.subheader("💬 Customer Reviews")

st.dataframe(
    df[df[product_col] == selected][
        [review_col] +
        ([rating_col] if rating_col else [])
    ],
    use_container_width=True,
    hide_index=True
)
