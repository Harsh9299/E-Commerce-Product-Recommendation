import streamlit as st
import pandas as pd
import requests
import re

from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="E-Commerce Product Recommendation",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Product Recommendation System")

st.write(
    "A product recommendation and review analysis system "
    "using ratings, customer reviews, sentiment and product features."
)


# =========================================================
# PART 1 — PRODUCT URL ANALYZER
# =========================================================

st.header("🔗 Analyze Any Product URL")

st.write(
    "Paste a product URL to analyze its rating and customer "
    "reviews and get a BUY / MAYBE / NOT RECOMMENDED decision."
)

product_url = st.text_input(
    "Paste Product URL",
    placeholder="https://example.com/product"
)


# =========================================================
# SENTIMENT ANALYZER
# =========================================================

analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(review):

    score = analyzer.polarity_scores(review)["compound"]

    if score >= 0.05:
        return "Positive", score

    elif score <= -0.05:
        return "Negative", score

    else:
        return "Neutral", score


# =========================================================
# EXTRACT PRODUCT DATA FROM URL
# =========================================================

def extract_product_data(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/142.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    if response.status_code != 200:
        return None, None, []

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # -----------------------------------------------------
    # PRODUCT TITLE
    # -----------------------------------------------------

    title = None

    if soup.find("h1"):

        title = soup.find("h1").get_text(
            " ",
            strip=True
        )

    # Try meta title if h1 is unavailable
    if not title:

        meta_title = soup.find(
            "meta",
            property="og:title"
        )

        if meta_title:

            title = meta_title.get(
                "content"
            )

    # -----------------------------------------------------
    # PRODUCT RATING
    # -----------------------------------------------------

    rating = None

    rating_patterns = [
        r"([0-5](?:\.[0-9])?)\s*(?:out of\s*5|/5)",
        r"([0-5](?:\.[0-9])?)\s*stars?",
        r"([0-5](?:\.[0-9])?)"
    ]

    # Search visible page text
    page_text = soup.get_text(
        " ",
        strip=True
    )

    for pattern in rating_patterns:

        matches = re.findall(
            pattern,
            page_text,
            flags=re.IGNORECASE
        )

        for match in matches:

            try:

                value = float(match)

                if 0 <= value <= 5:

                    rating = value
                    break

            except:

                pass

        if rating is not None:
            break

    # -----------------------------------------------------
    # REVIEWS
    # -----------------------------------------------------

    reviews = []

    review_selectors = [

        '[class*="review"]',

        '[class*="Review"]',

        '[id*="review"]',

        '[id*="Review"]',

        '[data-hook*="review"]'

    ]

    for selector in review_selectors:

        elements = soup.select(
            selector
        )

        for element in elements:

            text = element.get_text(
                " ",
                strip=True
            )

            # Avoid collecting tiny labels
            if len(text) >= 40:

                reviews.append(text)

    # Remove duplicates
    reviews = list(
        dict.fromkeys(reviews)
    )

    # Keep first 20 reviews
    reviews = reviews[:20]

    return title, rating, reviews


# =========================================================
# ANALYZE URL BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Product URL",
    type="primary"
):

    if not product_url:

        st.warning(
            "Please paste a product URL first."
        )

    elif not (
        product_url.startswith("http://")
        or
        product_url.startswith("https://")
    ):

        st.error(
            "Please enter a valid URL starting with "
            "http:// or https://"
        )

    else:

        with st.spinner(
            "Analyzing product..."
        ):

            try:

                title, rating, reviews = (
                    extract_product_data(
                        product_url
                    )
                )

                # -----------------------------------------
                # PRODUCT
                # -----------------------------------------

                st.subheader(
                    "📦 Product Information"
                )

                if title:

                    st.write(
                        f"**Product:** {title}"
                    )

                else:

                    st.info(
                        "Product name could not be extracted."
                    )

                # -----------------------------------------
                # RATING
                # -----------------------------------------

                if rating is not None:

                    st.metric(
                        "⭐ Product Rating",
                        f"{rating:.1f}/5"
                    )

                else:

                    st.warning(
                        "Product rating could not be extracted."
                    )

                # -----------------------------------------
                # REVIEW ANALYSIS
                # -----------------------------------------

                positive = 0
                negative = 0
                neutral = 0

                sentiment_scores = []

                for review in reviews:

                    sentiment, score = (
                        analyze_sentiment(
                            review
                        )
                    )

                    sentiment_scores.append(
                        score
                    )

                    if sentiment == "Positive":

                        positive += 1

                    elif sentiment == "Negative":

                        negative += 1

                    else:

                        neutral += 1

                # -----------------------------------------
                # AVERAGE SENTIMENT
                # -----------------------------------------

                if sentiment_scores:

                    average_sentiment = (
                        sum(sentiment_scores)
                        /
                        len(sentiment_scores)
                    )

                else:

                    average_sentiment = 0

                # -----------------------------------------
                # REVIEW RESULTS
                # -----------------------------------------

                st.subheader(
                    "💬 Customer Review Analysis"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Positive Reviews",
                        positive
                    )

                with col2:

                    st.metric(
                        "Negative Reviews",
                        negative
                    )

                with col3:

                    st.metric(
                        "Neutral Reviews",
                        neutral
                    )

                st.metric(
                    "Average Sentiment",
                    f"{average_sentiment:.2f}"
                )

                # -----------------------------------------
                # BUY SCORE
                # -----------------------------------------

                if rating is not None:

                    rating_score = (
                        rating / 5
                    )

                else:

                    rating_score = 0

                sentiment_score = (
                    average_sentiment + 1
                ) / 2

                buy_score = (
                    0.60 * rating_score
                    +
                    0.40 * sentiment_score
                )

                buy_percentage = (
                    buy_score * 100
                )

                # -----------------------------------------
                # PURCHASE DECISION
                # -----------------------------------------

                st.subheader(
                    "🛍️ Purchase Recommendation"
                )

                st.metric(
                    "Buy Score",
                    f"{buy_percentage:.2f}%"
                )

                if buy_percentage >= 75:

                    st.success(
                        "✅ BUY — The product has "
                        "a strong rating and positive "
                        "customer sentiment."
                    )

                elif buy_percentage >= 55:

                    st.warning(
                        "⚠️ MAYBE — The product has "
                        "mixed or moderate feedback. "
                        "Check the reviews before buying."
                    )

                else:

                    st.error(
                        "❌ NOT RECOMMENDED — The product "
                        "has relatively low rating or "
                        "negative customer sentiment."
                    )

                # -----------------------------------------
                # FORMULA
                # -----------------------------------------

                st.info(
                    "Buy Score = "
                    "60% Rating Score + "
                    "40% Review Sentiment Score"
                )

                # -----------------------------------------
                # DISPLAY REVIEWS
                # -----------------------------------------

                if reviews:

                    st.subheader(
                        "📝 Reviews Analyzed"
                    )

                    for i, review in enumerate(
                        reviews,
                        1
                    ):

                        sentiment, score = (
                            analyze_sentiment(
                                review
                            )
                        )

                        st.write(
                            f"**Review {i}: "
                            f"{sentiment}**"
                        )

                        st.write(review)

                        st.divider()

                else:

                    st.warning(
                        "No customer review text could "
                        "be extracted from this webpage."
                    )

            except Exception as e:

                st.error(
                    "Unable to analyze this product."
                )

                st.write(
                    f"Error: {e}"
                )


# =========================================================
# DIVIDER
# =========================================================

st.markdown("---")


# =========================================================
# PART 2 — EXISTING CSV RECOMMENDATION SYSTEM
# =========================================================

st.header(
    "📁 Product Recommendation from Dataset"
)

st.write(
    "Upload your CSV dataset to find similar products."
)


# =========================================================
# DATASET UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload the CSV dataset",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Upload your CSV dataset to use "
        "the product recommendation system."
    )

else:

    df = pd.read_csv(
        uploaded_file
    )

    # -----------------------------------------------------
    # PRODUCT ID
    # -----------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    df["Product ID"] = (
        df.index + 1
    )

    # -----------------------------------------------------
    # REQUIRED FEATURES
    # -----------------------------------------------------

    numeric_features = [

        "Number of clicks on similar products",

        "Number of similar products purchased so far",

        "Average rating given to similar products",

        "Median purchasing price (in rupees)",

        "Rating of the product",

        "Customer review sentiment score (overall)",

        "Price of the product"

    ]

    categorical_features = [

        "Gender",

        "Brand of the product",

        "Holiday",

        "Season",

        "Geographical locations"

    ]

    required_columns = (
        numeric_features
        +
        categorical_features
    )

    # -----------------------------------------------------
    # CHECK COLUMNS
    # -----------------------------------------------------

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]

    if missing_columns:

        st.error(
            "The following columns are missing:"
        )

        st.write(
            missing_columns
        )

        st.stop()

    # -----------------------------------------------------
    # DATA PREPROCESSING
    # -----------------------------------------------------

    for column in numeric_features:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    for column in categorical_features:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )

    # -----------------------------------------------------
    # FEATURE ENCODING
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",

                StandardScaler(),

                numeric_features
            ),

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                categorical_features
            )

        ]
    )

    feature_matrix = (
        preprocessor.fit_transform(
            df[required_columns]
        )
    )

    # -----------------------------------------------------
    # COSINE SIMILARITY
    # -----------------------------------------------------

    similarity_matrix = (
        cosine_similarity(
            feature_matrix
        )
    )

    # -----------------------------------------------------
    # PRODUCT SELECTION
    # -----------------------------------------------------

    st.header(
        "🔍 Select a Product"
    )

    selected_id = st.selectbox(
        "Choose Product ID",
        df["Product ID"].tolist()
    )

    selected_index = df.index[
        df["Product ID"]
        ==
        selected_id
    ][0]

    selected_product = df.loc[
        selected_index
    ]

    # -----------------------------------------------------
    # SELECTED PRODUCT DETAILS
    # -----------------------------------------------------

    st.subheader(
        "Selected Product"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Brand",
            selected_product[
                "Brand of the product"
            ]
        )

    with col2:

        st.metric(
            "Rating",
            f"{selected_product['Rating of the product']:.1f}/5"
        )

    with col3:

        st.metric(
            "Price",
            f"₹{selected_product['Price of the product']:.0f}"
        )

    with col4:

        st.metric(
            "Review Sentiment",
            f"{selected_product['Customer review sentiment score (overall)']:.2f}"
        )

    # -----------------------------------------------------
    # CALCULATE RECOMMENDATION SCORES
    # -----------------------------------------------------

    similarities = (
        similarity_matrix[
            selected_index
        ]
    )

    recommendations = pd.DataFrame({

        "Product ID":
            df["Product ID"],

        "Similarity":
            similarities

    })

    # -----------------------------------------------------
    # REMOVE SELECTED PRODUCT
    # -----------------------------------------------------

    recommendations = (
        recommendations[
            recommendations[
                "Product ID"
            ]
            != selected_id
        ]
        .copy()
    )

    # -----------------------------------------------------
    # ADD PRODUCT INFORMATION
    # -----------------------------------------------------

    recommendations = (
        recommendations.merge(
            df,
            on="Product ID",
            how="left"
        )
    )

    # -----------------------------------------------------
    # PRODUCT QUALITY SCORE
    # -----------------------------------------------------

    rating_score = (
        recommendations[
            "Rating of the product"
        ]
        /
        5
    )

    sentiment_score = (

        recommendations[
            "Customer review sentiment score (overall)"
        ]
        +
        1
    ) / 2

    recommendations[
        "Product Quality Score"
    ] = (

        0.5 * rating_score

        +

        0.5 * sentiment_score

    )

    # -----------------------------------------------------
    # FINAL RECOMMENDATION SCORE
    # -----------------------------------------------------

    recommendations[
        "Recommendation Score"
    ] = (

        0.60
        *
        recommendations[
            "Similarity"
        ]

        +

        0.40
        *
        recommendations[
            "Product Quality Score"
        ]

    )

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    recommendations = (
        recommendations.sort_values(
            "Recommendation Score",
            ascending=False
        )
    )

    # -----------------------------------------------------
    # TOP 5
    # -----------------------------------------------------

    top_recommendations = (
        recommendations.head(5)
        .copy()
    )

    # -----------------------------------------------------
    # DISPLAY TOP 5
    # -----------------------------------------------------

    st.header(
        "⭐ Top 5 Recommended Products"
    )

    display = top_recommendations[

        [

            "Product ID",

            "Brand of the product",

            "Rating of the product",

            "Price of the product",

            "Customer review sentiment score (overall)",

            "Recommendation Score"

        ]

    ].copy()

    display.columns = [

        "Product ID",

        "Brand",

        "Rating",

        "Price",

        "Review Sentiment",

        "Recommendation Score"

    ]

    display[
        "Recommendation Score"
    ] = (

        display[
            "Recommendation Score"
        ]
        *
        100
    ).round(2)

    st.dataframe(

        display,

        use_container_width=True,

        hide_index=True

    )

    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

    st.header(
        "📊 Recommendation Score"
    )

    chart_data = (
        display.copy()
    )

    chart_data["Product"] = (

        chart_data["Brand"]

        +

        " - ID "

        +

        chart_data[
            "Product ID"
        ].astype(str)

    )

    chart_data = (
        chart_data.set_index(
            "Product"
        )[
            "Recommendation Score"
        ]
    )

    st.bar_chart(
        chart_data
    )

    # -----------------------------------------------------
    # PRODUCT COMPARISON
    # -----------------------------------------------------

    st.header(
        "📈 Product Comparison"
    )

    comparison = (
        top_recommendations[
            [

                "Product ID",

                "Brand of the product",

                "Rating of the product",

                "Customer review sentiment score (overall)",

                "Price of the product"

            ]
        ]
        .copy()
    )

    comparison.columns = [

        "Product ID",

        "Brand",

        "Rating",

        "Review Sentiment",

        "Price"

    ]

    st.dataframe(

        comparison,

        use_container_width=True,

        hide_index=True

    )

    # -----------------------------------------------------
    # EXPLANATION
    # -----------------------------------------------------

    st.header(
        "💡 Why Are These Products Recommended?"
    )

    st.write(
        """
        The system uses a content-based recommendation approach.

        Each product is represented using numerical and categorical
        characteristics such as rating, price, review sentiment,
        brand, customer behavior, season, holiday and location.

        Cosine similarity is used to identify products with similar
        characteristics.

        The final recommendation score combines product similarity
        with product quality based on rating and customer review
        sentiment.
        """
    )

    # -----------------------------------------------------
    # BEST RECOMMENDATION
    # -----------------------------------------------------

    best = (
        top_recommendations.iloc[0]
    )

    st.success(

        f"🏆 Best Recommendation: "
        f"{best['Brand of the product']} "
        f"(Product ID {int(best['Product ID'])}) "
        f"with a recommendation score of "
        f"{best['Recommendation Score'] * 100:.2f}%"

    )

    # -----------------------------------------------------
    # SYSTEM EVALUATION
    # -----------------------------------------------------

    st.header(
        "📊 System Evaluation"
    )

    average_score = (

        top_recommendations[
            "Recommendation Score"
        ]
        .mean()
        *
        100

    )

    average_similarity = (

        top_recommendations[
            "Similarity"
        ]
        .mean()
        *
        100

    )

    best_score = (

        top_recommendations[
            "Recommendation Score"
        ]
        .max()
        *
        100

    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Average Top-5 Score",
            f"{average_score:.2f}%"
        )

    with col2:

        st.metric(
            "Average Similarity",
            f"{average_similarity:.2f}%"
        )

    with col3:

        st.metric(
            "Best Recommendation",
            f"{best_score:.2f}%"
        )

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    st.header(
        "📁 Dataset Information"
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        st.metric(
            "Total Products",
            len(df)
        )

    with col2:

        st.metric(
            "Features Used",
            len(required_columns)
        )

    with st.expander(
        "View Complete Dataset"
    ):

        st.dataframe(

            df,

            use_container_width=True,

            hide_index=True

        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "E-Commerce Product Recommendation System | "
    "Content-Based Recommendation + Product Review Analysis"
)
