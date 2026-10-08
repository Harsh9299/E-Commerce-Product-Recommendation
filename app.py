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
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="E-Commerce AI Advisor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #667eea 0%,
            #764ba2 100%
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.95;
    }

    .section-title {
        font-size: 27px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .info-card {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        background-color: rgba(128,128,128,0.06);
        margin-bottom: 12px;
    }

    .score-card {
        padding: 20px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
        background-color: rgba(128,128,128,0.06);
    }

    .score-number {
        font-size: 32px;
        font-weight: 800;
    }

    .small-text {
        font-size: 14px;
        opacity: 0.75;
    }

    .alternative-card {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛒 E-Commerce AI Advisor</h1>
        <p>
        Analyze products, understand customer reviews,
        and discover better-rated alternatives.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
    "Use the product URL analyzer to evaluate a product, "
    "or upload your dataset to discover similar products."
)


# =========================================================
# SENTIMENT ANALYZER
# =========================================================

analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):

    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive", score

    elif score <= -0.05:
        return "Negative", score

    else:
        return "Neutral", score


# =========================================================
# EXTRACT PRICE
# =========================================================

def extract_price(text):

    if not text:
        return None

    patterns = [
        r"(?:₹|Rs\.?|INR)\s*([0-9,]+(?:\.[0-9]+)?)",
        r"([0-9,]+(?:\.[0-9]+)?)\s*(?:₹|INR)"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            try:

                value = float(
                    match.replace(",", "")
                )

                if value > 0:
                    return value

            except:
                pass

    return None


# =========================================================
# EXTRACT RATING
# =========================================================

def extract_rating(text):

    if not text:
        return None

    patterns = [
        r"([0-5](?:\.[0-9])?)\s*(?:out of\s*5|/5)",
        r"([0-5](?:\.[0-9])?)\s*stars?"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            try:

                value = float(match)

                if 0 <= value <= 5:
                    return value

            except:
                pass

    return None


# =========================================================
# EXTRACT PRODUCT BRAND
# =========================================================

def extract_brand(title):

    if not title:
        return None

    known_brands = [
        "Nike",
        "Adidas",
        "Puma",
        "Reebok",
        "Apple",
        "Samsung",
        "OnePlus",
        "Sony",
        "LG",
        "HP",
        "Dell",
        "Lenovo",
        "Asus",
        "Acer",
        "Levi's",
        "Pepe Jeans",
        "Flying Machine",
        "Max",
        "Roadster",
        "H&M"
    ]

    title_lower = title.lower()

    for brand in known_brands:

        if brand.lower() in title_lower:
            return brand

    return None


# =========================================================
# PRODUCT URL EXTRACTION
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

        return {
            "title": None,
            "rating": None,
            "price": None,
            "brand": None,
            "reviews": []
        }

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # -----------------------------------------------------
    # PRODUCT TITLE
    # -----------------------------------------------------

    title = None

    h1 = soup.find("h1")

    if h1:

        title = h1.get_text(
            " ",
            strip=True
        )

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
    # PAGE TEXT
    # -----------------------------------------------------

    page_text = soup.get_text(
        " ",
        strip=True
    )

    # -----------------------------------------------------
    # RATING
    # -----------------------------------------------------

    rating = extract_rating(
        page_text
    )

    # Try structured rating information
    if rating is None:

        rating_meta = soup.find(
            attrs={
                "itemprop": "ratingValue"
            }
        )

        if rating_meta:

            try:

                rating = float(
                    rating_meta.get(
                        "content"
                    )
                )

            except:
                pass

    # -----------------------------------------------------
    # PRICE
    # -----------------------------------------------------

    price = None

    price_meta = soup.find(
        attrs={
            "itemprop": "price"
        }
    )

    if price_meta:

        try:

            price = float(
                price_meta.get(
                    "content"
                )
            )

        except:
            pass

    if price is None:

        price = extract_price(
            page_text
        )

    # -----------------------------------------------------
    # REVIEWS
    # -----------------------------------------------------

    reviews = []

    review_selectors = [

        '[class*="review"]',
        '[class*="Review"]',
        '[id*="review"]',
        '[id*="Review"]',
        '[data-hook*="review"]',
        '[itemprop="reviewBody"]'

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

            # Avoid tiny pieces of text
            if len(text) >= 50:

                reviews.append(
                    text
                )

    # Remove duplicates
    reviews = list(
        dict.fromkeys(
            reviews
        )
    )

    # Limit number of reviews
    reviews = reviews[:30]

    brand = extract_brand(
        title
    )

    return {
        "title": title,
        "rating": rating,
        "price": price,
        "brand": brand,
        "reviews": reviews
    }


# =========================================================
# LOAD DATASET
# =========================================================

st.sidebar.header(
    "⚙️ Data & Controls"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Recommendation Dataset",
    type=["csv"]
)


# =========================================================
# DATASET PROCESSING
# =========================================================

df = None
feature_matrix = None
similarity_matrix = None

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        df = df.reset_index(
            drop=True
        )

        # Product ID
        df["Product ID"] = (
            df.index + 1
        )

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

        missing_columns = [

            column
            for column in required_columns
            if column not in df.columns

        ]

        if missing_columns:

            st.sidebar.error(
                "Dataset is missing required columns."
            )

            st.sidebar.write(
                missing_columns
            )

            df = None

        else:

            # Numeric cleaning
            for column in numeric_features:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).fillna(0)

            # Categorical cleaning
            for column in categorical_features:

                df[column] = (
                    df[column]
                    .fillna("Unknown")
                    .astype(str)
                )

            # ------------------------------------------------
            # PREPROCESSING
            # ------------------------------------------------

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

            # ------------------------------------------------
            # COSINE SIMILARITY
            # ------------------------------------------------

            similarity_matrix = (
                cosine_similarity(
                    feature_matrix
                )
            )

    except Exception as e:

        st.sidebar.error(
            f"Dataset error: {e}"
        )

        df = None


# =========================================================
# PRODUCT URL ANALYZER
# =========================================================

st.markdown(
    '<div class="section-title">🔗 Analyze a Product</div>',
    unsafe_allow_html=True
)

st.write(
    "Paste the URL of an actual product page."
)

product_url = st.text_input(
    "Product URL",
    placeholder="https://www.example.com/product/..."
)

analyze_button = st.button(
    "🔍 Analyze Product",
    type="primary",
    use_container_width=True
)


if analyze_button:

    if not product_url:

        st.warning(
            "Please paste a product URL."
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
            "Analyzing product and customer reviews..."
        ):

            try:

                product_data = (
                    extract_product_data(
                        product_url
                    )
                )

                title = product_data[
                    "title"
                ]

                rating = product_data[
                    "rating"
                ]

                price = product_data[
                    "price"
                ]

                brand = product_data[
                    "brand"
                ]

                reviews = product_data[
                    "reviews"
                ]

                # Save in session state
                st.session_state[
                    "url_product"
                ] = product_data

            except Exception as e:

                st.error(
                    f"Unable to analyze the URL: {e}"
                )


# =========================================================
# SHOW URL PRODUCT
# =========================================================

if "url_product" in st.session_state:

    product_data = st.session_state[
        "url_product"
    ]

    title = product_data[
        "title"
    ]

    rating = product_data[
        "rating"
    ]

    price = product_data[
        "price"
    ]

    brand = product_data[
        "brand"
    ]

    reviews = product_data[
        "reviews"
    ]

    st.markdown(
        '<div class="section-title">📦 Product Analysis</div>',
        unsafe_allow_html=True
    )

    if title:

        st.subheader(
            title
        )

    # -----------------------------------------------------
    # BASIC PRODUCT METRICS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        if rating is not None:

            st.metric(
                "⭐ Rating",
                f"{rating:.1f}/5"
            )

        else:

            st.metric(
                "⭐ Rating",
                "Not found"
            )

    with c2:

        if price is not None:

            st.metric(
                "💰 Price",
                f"₹{price:,.0f}"
            )

        else:

            st.metric(
                "💰 Price",
                "Not found"
            )

    with c3:

        st.metric(
            "💬 Reviews Found",
            len(reviews)
        )

    with c4:

        st.metric(
            "🏷️ Brand",
            brand if brand else "Unknown"
        )

    # -----------------------------------------------------
    # SENTIMENT ANALYSIS
    # -----------------------------------------------------

    positive = 0
    negative = 0
    neutral = 0

    sentiment_scores = []

    review_results = []

    for review in reviews:

        sentiment, score = (
            analyze_sentiment(
                review
            )
        )

        sentiment_scores.append(
            score
        )

        review_results.append(
            {
                "Review": review,
                "Sentiment": sentiment,
                "Score": score
            }
        )

        if sentiment == "Positive":

            positive += 1

        elif sentiment == "Negative":

            negative += 1

        else:

            neutral += 1

    if sentiment_scores:

        average_sentiment = (
            sum(sentiment_scores)
            /
            len(sentiment_scores)
        )

    else:

        average_sentiment = 0

    total_reviews = (
        positive
        +
        negative
        +
        neutral
    )

    if total_reviews > 0:

        positive_percentage = (
            positive
            /
            total_reviews
            *
            100
        )

        negative_percentage = (
            negative
            /
            total_reviews
            *
            100
        )

        neutral_percentage = (
            neutral
            /
            total_reviews
            *
            100
        )

    else:

        positive_percentage = 0
        negative_percentage = 0
        neutral_percentage = 0

    st.markdown(
        '<div class="section-title">💬 Customer Review Insights</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "😊 Positive",
            f"{positive_percentage:.1f}%"
        )

    with s2:

        st.metric(
            "😐 Neutral",
            f"{neutral_percentage:.1f}%"
        )

    with s3:

        st.metric(
            "😞 Negative",
            f"{negative_percentage:.1f}%"
        )

    with s4:

        st.metric(
            "📊 Avg Sentiment",
            f"{average_sentiment:.2f}"
        )

    # -----------------------------------------------------
    # BUY SCORE
    # -----------------------------------------------------

    if rating is not None:

        rating_score = (
            rating / 5
        )

    else:

        rating_score = 0

    sentiment_score = (
        average_sentiment + 1
    ) / 2

    if total_reviews >= 20:

        review_confidence = 1.0

    elif total_reviews >= 10:

        review_confidence = 0.8

    elif total_reviews >= 5:

        review_confidence = 0.6

    elif total_reviews > 0:

        review_confidence = 0.4

    else:

        review_confidence = 0

    if negative_percentage <= 10:

        risk_score = 1.0

    elif negative_percentage <= 20:

        risk_score = 0.8

    elif negative_percentage <= 35:

        risk_score = 0.5

    else:

        risk_score = 0.2

    buy_score = (

        0.40 * rating_score
        +
        0.30 * sentiment_score
        +
        0.15 * review_confidence
        +
        0.15 * risk_score

    )

    buy_percentage = (
        buy_score * 100
    )

    st.markdown(
        '<div class="section-title">🛍️ Purchase Decision</div>',
        unsafe_allow_html=True
    )

    b1, b2, b3 = st.columns(3)

    with b1:

        st.metric(
            "🛒 Buy Score",
            f"{buy_percentage:.1f}/100"
        )

    with b2:

        st.metric(
            "⭐ Rating Contribution",
            f"{rating_score * 100:.1f}%"
        )

    with b3:

        st.metric(
            "💬 Sentiment Contribution",
            f"{sentiment_score * 100:.1f}%"
        )

    if buy_percentage >= 75:

        st.success(
            "✅ BUY — This product has strong "
            "overall customer feedback."
        )

    elif buy_percentage >= 55:

        st.warning(
            "⚠️ MAYBE — The product has mixed "
            "or moderate customer feedback."
        )

    else:

        st.error(
            "❌ NOT RECOMMENDED — The product shows "
            "weak rating or customer sentiment."
        )

    st.caption(
        "Buy Score = 40% Rating + 30% Sentiment + "
        "15% Review Confidence + 15% Risk"
    )

    # -----------------------------------------------------
    # SENTIMENT CHART
    # -----------------------------------------------------

    if total_reviews > 0:

        sentiment_chart = pd.DataFrame(
            {
                "Sentiment": [
                    "Positive",
                    "Neutral",
                    "Negative"
                ],

                "Reviews": [
                    positive,
                    neutral,
                    negative
                ]
            }
        )

        st.bar_chart(
            sentiment_chart.set_index(
                "Sentiment"
            )
        )

    # =====================================================
    # BETTER ALTERNATIVES
    # =====================================================

    if df is not None:

        st.markdown(
            '<div class="section-title">🔄 Better Similar Alternatives</div>',
            unsafe_allow_html=True
        )

        if rating is None:

            st.info(
                "A rating could not be extracted from "
                "the URL, so same/higher-rating filtering "
                "cannot be performed."
            )

        else:

            alternatives = df[
                df[
                    "Rating of the product"
                ] >= rating
            ].copy()

            # Don't recommend products with zero/invalid ratings
            alternatives = alternatives[
                alternatives[
                    "Rating of the product"
                ] > 0
            ]

            if alternatives.empty:

                st.warning(
                    "No products in your dataset have "
                    "the same or higher rating."
                )

            else:

                # -----------------------------------------
                # RATING SCORE
                # -----------------------------------------

                alternatives[
                    "Rating Score"
                ] = (
                    alternatives[
                        "Rating of the product"
                    ]
                    /
                    5
                )

                # -----------------------------------------
                # SENTIMENT SCORE
                # -----------------------------------------

                alternatives[
                    "Sentiment Score"
                ] = (

                    alternatives[
                        "Customer review sentiment score (overall)"
                    ]
                    +
                    1

                ) / 2

                # -----------------------------------------
                # PRICE SCORE
                # -----------------------------------------

                if price is not None:

                    price_difference = (
                        abs(
                            alternatives[
                                "Price of the product"
                            ]
                            -
                            price
                        )
                    )

                    price_score = 1 / (
                        1
                        +
                        price_difference
                        /
                        max(price, 1)
                    )

                else:

                    price_score = 0.5

                alternatives[
                    "Price Score"
                ] = price_score

                # -----------------------------------------
                # BRAND MATCH
                # -----------------------------------------

                if brand:

                    alternatives[
                        "Brand Match"
                    ] = (

                        alternatives[
                            "Brand of the product"
                        ]
                        .str.lower()
                        ==
                        brand.lower()

                    ).astype(float)

                else:

                    alternatives[
                        "Brand Match"
                    ] = 0

                # -----------------------------------------
                # ALTERNATIVE SCORE
                # -----------------------------------------

                alternatives[
                    "Alternative Score"
                ] = (

                    0.40
                    *
                    alternatives[
                        "Rating Score"
                    ]

                    +

                    0.25
                    *
                    alternatives[
                        "Sentiment Score"
                    ]

                    +

                    0.20
                    *
                    alternatives[
                        "Price Score"
                    ]

                    +

                    0.15
                    *
                    alternatives[
                        "Brand Match"
                    ]

                )

                # -----------------------------------------
                # SORT
                # -----------------------------------------

                alternatives = (
                    alternatives
                    .sort_values(
                        "Alternative Score",
                        ascending=False
                    )
                )

                alternatives = (
                    alternatives
                    .head(5)
                    .copy()
                )

                # -----------------------------------------
                # ALTERNATIVE TABLE
                # -----------------------------------------

                alternative_display = alternatives[
                    [
                        "Product ID",
                        "Brand of the product",
                        "Rating of the product",
                        "Price of the product",
                        "Customer review sentiment score (overall)",
                        "Alternative Score"
                    ]
                ].copy()

                alternative_display.columns = [

                    "Product ID",
                    "Brand",
                    "Rating",
                    "Price",
                    "Review Sentiment",
                    "Alternative Score"

                ]

                alternative_display[
                    "Alternative Score"
                ] = (

                    alternative_display[
                        "Alternative Score"
                    ]
                    *
                    100

                ).round(2)

                st.dataframe(
                    alternative_display,
                    use_container_width=True,
                    hide_index=True
                )

                # -----------------------------------------
                # BEST ALTERNATIVE
                # -----------------------------------------

                best_alternative = (
                    alternatives.iloc[0]
                )

                best_rating = (
                    best_alternative[
                        "Rating of the product"
                    ]
                )

                best_price = (
                    best_alternative[
                        "Price of the product"
                    ]
                )

                best_brand = (
                    best_alternative[
                        "Brand of the product"
                    ]
                )

                rating_difference = (
                    best_rating
                    -
                    rating
                )

                st.success(
                    f"🏆 Best Alternative: "
                    f"{best_brand} "
                    f"(Product ID "
                    f"{int(best_alternative['Product ID'])}) "
                    f"— ⭐ {best_rating:.1f}/5"
                )

                if price is not None:

                    if best_price < price:

                        saving = (
                            price
                            -
                            best_price
                        )

                        st.info(
                            f"💰 This alternative is "
                            f"₹{saving:,.0f} cheaper "
                            f"and has "
                            f"{rating_difference:+.1f} "
                            f"higher rating points."
                        )

                    elif best_price > price:

                        extra = (
                            best_price
                            -
                            price
                        )

                        st.info(
                            f"💰 This alternative costs "
                            f"₹{extra:,.0f} more but has "
                            f"{rating_difference:+.1f} "
                            f"rating points."
                        )

                    else:

                        st.info(
                            "💰 This alternative has "
                            "approximately the same price "
                            "with a higher/equal rating."
                        )

                # -----------------------------------------
                # ALTERNATIVE CHART
                # -----------------------------------------

                chart = alternative_display.copy()

                chart["Product"] = (

                    chart[
                        "Brand"
                    ]
                    +
                    " - ID "
                    +
                    chart[
                        "Product ID"
                    ].astype(str)

                )

                chart = chart.set_index(
                    "Product"
                )[
                    "Alternative Score"
                ]

                st.subheader(
                    "📊 Alternative Comparison"
                )

                st.bar_chart(
                    chart
                )

                st.caption(
                    "Alternative Score combines rating, "
                    "review sentiment, price similarity "
                    "and brand match."
                )

    # -----------------------------------------------------
    # REVIEWS
    # -----------------------------------------------------

    if reviews:

        st.markdown(
            '<div class="section-title">📝 Reviews Analyzed</div>',
            unsafe_allow_html=True
        )

        for i, result in enumerate(
            review_results,
            1
        ):

            sentiment = result[
                "Sentiment"
            ]

            score = result[
                "Score"
            ]

            review = result[
                "Review"
            ]

            if sentiment == "Positive":

                icon = "😊"

            elif sentiment == "Negative":

                icon = "😞"

            else:

                icon = "😐"

            with st.expander(
                f"{icon} Review {i} — {sentiment}"
            ):

                st.write(
                    review
                )

                st.caption(
                    f"Sentiment score: {score:.2f}"
                )

    else:

        st.warning(
            "No customer review text could be extracted "
            "from this webpage."
        )


# =========================================================
# DIVIDER
# =========================================================

st.markdown("---")


# =========================================================
# DATASET RECOMMENDATION SYSTEM
# =========================================================

st.markdown(
    '<div class="section-title">📁 Dataset Product Recommendation</div>',
    unsafe_allow_html=True
)

if df is None:

    st.info(
        "Upload your CSV dataset from the sidebar "
        "to use the content-based recommendation system."
    )

else:

    st.write(
        "Select a product from your dataset to find "
        "the top 5 similar products."
    )

    # -----------------------------------------------------
    # DATASET SUMMARY
    # -----------------------------------------------------

    d1, d2, d3 = st.columns(3)

    with d1:

        st.metric(
            "📦 Total Products",
            len(df)
        )

    with d2:

        st.metric(
            "🔢 Features Used",
            12
        )

    with d3:

        st.metric(
            "⭐ Average Rating",
            f"{df['Rating of the product'].mean():.2f}/5"
        )

    # -----------------------------------------------------
    # PRODUCT SELECTION
    # -----------------------------------------------------

    selected_id = st.selectbox(
        "🔍 Select Product ID",
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

    st.subheader(
        "Selected Product"
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        st.metric(
            "🏷️ Brand",
            selected_product[
                "Brand of the product"
            ]
        )

    with p2:

        st.metric(
            "⭐ Rating",
            f"{selected_product['Rating of the product']:.1f}/5"
        )

    with p3:

        st.metric(
            "💰 Price",
            f"₹{selected_product['Price of the product']:,.0f}"
        )

    with p4:

        st.metric(
            "💬 Sentiment",
            f"{selected_product['Customer review sentiment score (overall)']:.2f}"
        )

    # -----------------------------------------------------
    # SIMILARITY
    # -----------------------------------------------------

    similarities = (
        similarity_matrix[
            selected_index
        ]
    )

    recommendations = pd.DataFrame(
        {
            "Product ID":
                df["Product ID"],

            "Similarity":
                similarities
        }
    )

    recommendations = (
        recommendations[
            recommendations[
                "Product ID"
            ]
            != selected_id
        ]
        .copy()
    )

    recommendations = (
        recommendations.merge(
            df,
            on="Product ID",
            how="left"
        )
    )

    # -----------------------------------------------------
    # PRODUCT QUALITY
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
    # FINAL SCORE
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

    recommendations = (
        recommendations
        .sort_values(
            "Recommendation Score",
            ascending=False
        )
    )

    top_recommendations = (
        recommendations
        .head(5)
        .copy()
    )

    # -----------------------------------------------------
    # TOP 5
    # -----------------------------------------------------

    st.subheader(
        "⭐ Top 5 Similar Products"
    )

    display = top_recommendations[
        [
            "Product ID",
            "Brand of the product",
            "Rating of the product",
            "Price of the product",
            "Customer review sentiment score (overall)",
            "Similarity",
            "Recommendation Score"
        ]
    ].copy()

    display.columns = [

        "Product ID",
        "Brand",
        "Rating",
        "Price",
        "Review Sentiment",
        "Similarity",
        "Recommendation Score"

    ]

    display[
        "Similarity"
    ] = (

        display[
            "Similarity"
        ]
        *
        100

    ).round(2)

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

    st.subheader(
        "📊 Recommendation Scores"
    )

    chart_data = display.copy()

    chart_data[
        "Product"
    ] = (

        chart_data[
            "Brand"
        ]
        +
        " - ID "
        +
        chart_data[
            "Product ID"
        ].astype(str)

    )

    chart_data = (
        chart_data
        .set_index(
            "Product"
        )[
            "Recommendation Score"
        ]
    )

    st.bar_chart(
        chart_data
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
        f"(Product ID "
        f"{int(best['Product ID'])}) "
        f"with a recommendation score of "
        f"{best['Recommendation Score'] * 100:.2f}%"
    )

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    st.subheader(
        "📈 Recommendation Performance"
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

    e1, e2, e3 = st.columns(3)

    with e1:

        st.metric(
            "Average Top-5 Score",
            f"{average_score:.2f}%"
        )

    with e2:

        st.metric(
            "Average Similarity",
            f"{average_similarity:.2f}%"
        )

    with e3:

        st.metric(
            "Best Recommendation",
            f"{best_score:.2f}%"
        )

    st.caption(
        "These are recommendation/similarity scores, "
        "not classification accuracy."
    )

    # -----------------------------------------------------
    # EXPLANATION
    # -----------------------------------------------------

    with st.expander(
        "💡 How does the recommendation system work?"
    ):

        st.write(
            """
            The system uses a content-based recommendation approach.

            Each product is represented using numerical and
            categorical characteristics such as:

            • Customer behavior
            • Product rating
            • Product price
            • Review sentiment
            • Brand
            • Gender
            • Holiday
            • Season
            • Geographical location

            Numerical features are standardized using StandardScaler.

            Categorical features are converted into numerical
            representations using OneHotEncoder.

            Cosine similarity is then used to find products
            with similar characteristics.

            The final Recommendation Score combines:

            60% Product Similarity

            40% Product Quality

            Product Quality is based on:

            50% Rating

            50% Review Sentiment
            """
        )

    # -----------------------------------------------------
    # COMPLETE DATASET
    # -----------------------------------------------------

    with st.expander(
        "📁 View Complete Dataset"
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
    "🛒 E-Commerce AI Advisor | "
    "Content-Based Recommendation | "
    "Review Sentiment Analysis | "
    "Better Product Alternatives"
)
