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
    page_title="E-Commerce Product Recommendation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# NEUTRAL DARK THEME
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background: #1f2328;
}

.main {
    padding-top: 1rem;
}


/* =====================================================
   MAIN PROJECT HEADER
   ===================================================== */

.app-header {
    padding: 28px 30px;
    border-radius: 14px;
    background: #292e34;
    color: #ffffff;
    margin-bottom: 30px;
    border: 1px solid #41474e;
}

.app-header h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: 0.3px;
    background: #3a4047;
    padding: 11px 17px;
    border-radius: 8px;
    display: inline-block;
    border-left: 4px solid #c4c8cc;
}

.app-header p {
    margin-top: 14px;
    margin-bottom: 0;
    color: #c2c6ca;
    font-size: 16px;
    line-height: 1.6;
}


/* =====================================================
   SECTION TITLES
   ===================================================== */

.section-title {
    font-size: 30px;
    font-weight: 900;
    color: #f2f3f4;
    margin-top: 30px;
    margin-bottom: 18px;
    letter-spacing: 0.2px;
}


/* =====================================================
   REMOVE METRIC BOXES
   ===================================================== */

[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 4px !important;
    box-shadow: none !important;
}

[data-testid="stMetricLabel"] {
    color: #aeb4ba !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    color: #f2f3f4 !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    color: #aeb4ba !important;
}


/* =====================================================
   GENERAL TEXT
   ===================================================== */

.stMarkdown,
.stText {
    color: #d5d8db;
}

h1,
h2,
h3 {
    color: #f1f2f3 !important;
}


/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {
    background: #181b1f;
    border-right: 1px solid #34393f;
}

section[data-testid="stSidebar"] * {
    color: #d7dadd;
}


/* =====================================================
   BUTTONS
   ===================================================== */

.stButton > button {
    border-radius: 8px;
    font-weight: 700;
    border: 1px solid #5b6269;
    background: #343a40;
    color: #f1f2f3;
}

.stButton > button:hover {
    border-color: #8b9299;
    background: #3d4349;
    color: #ffffff;
}


/* =====================================================
   TEXT INPUT
   ===================================================== */

.stTextInput input {
    background: #292e34 !important;
    color: #f1f2f3 !important;
    border: 1px solid #555c63 !important;
    border-radius: 8px !important;
}

.stTextInput input::placeholder {
    color: #8e959c !important;
}


/* =====================================================
   SELECT BOX
   ===================================================== */

div[data-baseweb="select"] > div {
    background: #292e34 !important;
    border-color: #555c63 !important;
    color: #f1f2f3 !important;
}


/* =====================================================
   DATA TABLE
   ===================================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #3a4046;
    border-radius: 10px;
}


/* =====================================================
   EXPANDER
   ===================================================== */

[data-testid="stExpander"] {
    border: 1px solid #3a4046;
    border-radius: 10px;
    background: #24282d;
}


/* =====================================================
   DIVIDERS
   ===================================================== */

hr {
    border: none;
    border-top: 1px solid #3a4046;
    margin: 32px 0;
}


/* =====================================================
   ALERTS
   ===================================================== */

[data-testid="stAlert"] {
    border-radius: 9px;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# MAIN PROJECT HEADER
# =========================================================

st.markdown(
    """
<div class="app-header">
<h1>🛒 E-Commerce Product Recommendation Using Customer Reviews</h1>
<p>Analyze customer feedback and discover similar products with equal or higher ratings.</p>
</div>
""",
    unsafe_allow_html=True
)


st.write(
    "The system analyzes product ratings and customer "
    "review sentiment to provide a purchase recommendation "
    "and find similar products with the same or higher rating."
)


# =========================================================
# SENTIMENT ANALYZER
# =========================================================

analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):

    score = analyzer.polarity_scores(
        text
    )["compound"]

    if score >= 0.05:
        return "Positive", score

    elif score <= -0.05:
        return "Negative", score

    return "Neutral", score


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
# EXTRACT BRAND
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

        "Accept-Language":
            "en-US,en;q=0.9"

    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    if response.status_code != 200:

        raise Exception(
            f"Website returned status code "
            f"{response.status_code}"
        )

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )


    # =====================================================
    # PRODUCT TITLE
    # =====================================================

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


    # =====================================================
    # PAGE TEXT
    # =====================================================

    page_text = soup.get_text(
        " ",
        strip=True
    )


    # =====================================================
    # RATING
    # =====================================================

    rating = extract_rating(
        page_text
    )


    # Try structured rating

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


    # =====================================================
    # REVIEWS
    # =====================================================

    reviews = []

    review_selectors = [

        '[itemprop="reviewBody"]',

        '[class*="review-text"]',

        '[class*="reviewText"]',

        '[class*="review-content"]',

        '[class*="reviewContent"]',

        '[data-hook="review-body"]',

        '[class*="review"]',

        '[id*="review"]'

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


    # Analyze maximum 30 reviews

    reviews = reviews[:30]


    # =====================================================
    # BRAND
    # =====================================================

    brand = extract_brand(
        title
    )


    return {

        "title": title,

        "rating": rating,

        "brand": brand,

        "reviews": reviews

    }


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "⚙️ Controls"
)

st.sidebar.write(
    "Upload your recommendation dataset."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)


# =========================================================
# DATASET VARIABLES
# =========================================================

df = None

preprocessor = None

feature_matrix = None

similarity_matrix = None


# =========================================================
# DATASET FEATURE DEFINITIONS
# =========================================================

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


# =========================================================
# LOAD DATASET
# =========================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        df = df.reset_index(
            drop=True
        )

        df["Product ID"] = (
            df.index + 1
        )


        # =================================================
        # CHECK COLUMNS
        # =================================================

        missing_columns = [

            column

            for column in required_columns

            if column not in df.columns

        ]


        if missing_columns:

            st.sidebar.error(
                "Required columns are missing."
            )

            st.sidebar.write(
                missing_columns
            )

            df = None


        else:

            # =============================================
            # CLEAN NUMERICAL DATA
            # =============================================

            for column in numeric_features:

                df[column] = pd.to_numeric(

                    df[column],

                    errors="coerce"

                ).fillna(0)


            # =============================================
            # CLEAN CATEGORICAL DATA
            # =============================================

            for column in categorical_features:

                df[column] = (

                    df[column]

                    .fillna("Unknown")

                    .astype(str)

                )


            # =============================================
            # PREPROCESSOR
            # =============================================

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


            # =============================================
            # DATASET FEATURE MATRIX
            # =============================================

            feature_matrix = (

                preprocessor.fit_transform(

                    df[
                        required_columns
                    ]

                )

            )


            # =============================================
            # NORMAL DATASET SIMILARITY
            # =============================================

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
# URL PRODUCT ANALYZER
# =========================================================

st.markdown(
    '<div class="section-title">🔗 Analyze a Product URL</div>',
    unsafe_allow_html=True
)

st.write(
    "Paste a product URL. The system will analyze its "
    "rating and customer reviews, then find similar "
    "products from your uploaded dataset."
)


product_url = st.text_input(
    "Product URL",
    placeholder="https://www.example.com/product/..."
)


analyze_button = st.button(
    "🔍 Analyze Product & Find Similar Products",
    type="primary",
    use_container_width=True
)


# =========================================================
# ANALYZE URL
# =========================================================

if analyze_button:

    if not product_url:

        st.warning(
            "Please paste a product URL first."
        )

    elif not (

        product_url.startswith(
            "http://"
        )

        or

        product_url.startswith(
            "https://"
        )

    ):

        st.error(
            "Please enter a valid URL beginning with "
            "http:// or https://"
        )

    elif df is None:

        st.warning(
            "Please upload your CSV dataset from the "
            "sidebar before analyzing the URL."
        )

    else:

        with st.spinner(
            "Analyzing product and finding similar products..."
        ):

            try:

                product_data = (
                    extract_product_data(
                        product_url
                    )
                )

                st.session_state[
                    "url_product"
                ] = product_data


                # =========================================
                # RESET OLD RESULTS
                # =========================================

                st.session_state[
                    "url_recommendations"
                ] = None


            except Exception as e:

                st.error(
                    "Unable to analyze this product."
                )

                st.info(
                    f"Reason: {e}"
                )


# =========================================================
# URL PRODUCT RESULTS
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

    brand = product_data[
        "brand"
    ]

    reviews = product_data[
        "reviews"
    ]


    # =====================================================
    # PRODUCT INFORMATION
    # =====================================================

    st.markdown(
        '<div class="section-title">📦 Product Information</div>',
        unsafe_allow_html=True
    )


    if title:

        st.subheader(
            title
        )

    else:

        st.info(
            "Product name could not be extracted."
        )


    p1, p2, p3 = st.columns(3)


    with p1:

        if rating is not None:

            st.metric(
                "⭐ Product Rating",
                f"{rating:.1f}/5"
            )

        else:

            st.metric(
                "⭐ Product Rating",
                "Not found"
            )


    with p2:

        st.metric(
            "💬 Reviews Analyzed",
            len(reviews)
        )


    with p3:

        st.metric(
            "🏷️ Brand",
            brand if brand else "Unknown"
        )


    # =====================================================
    # SENTIMENT ANALYSIS
    # =====================================================

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


    total_reviews = (

        positive

        +

        negative

        +

        neutral

    )


    if sentiment_scores:

        average_sentiment = (

            sum(
                sentiment_scores
            )

            /

            len(
                sentiment_scores
            )

        )

    else:

        average_sentiment = 0


    if total_reviews > 0:

        positive_percentage = (

            positive
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

        negative_percentage = (

            negative
            /
            total_reviews
            *
            100

        )

    else:

        positive_percentage = 0

        neutral_percentage = 0

        negative_percentage = 0


    # =====================================================
    # CUSTOMER REVIEW INSIGHTS
    # =====================================================

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
            "📊 Average Sentiment",
            f"{average_sentiment:.2f}"
        )


    # =====================================================
    # SENTIMENT CHART
    # =====================================================

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
    # BUY SCORE
    # =====================================================

    if rating is not None:

        rating_score = (

            rating
            /
            5

        )

    else:

        rating_score = 0


    sentiment_score = (

        average_sentiment
        +
        1

    ) / 2


    # =====================================================
    # REVIEW CONFIDENCE
    # =====================================================

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


    # =====================================================
    # RISK SCORE
    # =====================================================

    if negative_percentage <= 10:

        risk_score = 1.0

    elif negative_percentage <= 20:

        risk_score = 0.8

    elif negative_percentage <= 35:

        risk_score = 0.5

    else:

        risk_score = 0.2


    # =====================================================
    # BUY SCORE
    # =====================================================

    buy_score = (

        0.40
        *
        rating_score

        +

        0.30
        *
        sentiment_score

        +

        0.15
        *
        review_confidence

        +

        0.15
        *
        risk_score

    )


    buy_percentage = (

        buy_score
        *
        100

    )


    # =====================================================
    # PURCHASE DECISION
    # =====================================================

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
            "⭐ Rating Score",
            f"{rating_score * 100:.1f}%"
        )


    with b3:

        st.metric(
            "💬 Sentiment Score",
            f"{sentiment_score * 100:.1f}%"
        )


    if buy_percentage >= 75:

        st.success(
            "✅ BUY — The product has strong "
            "overall customer feedback."
        )

    elif buy_percentage >= 55:

        st.warning(
            "⚠️ MAYBE — The product has mixed "
            "or moderate customer feedback."
        )

    else:

        st.error(
            "❌ NOT RECOMMENDED — The product has "
            "relatively weak rating or sentiment."
        )


    st.caption(
        "Buy Score = 40% Rating + 30% Sentiment + "
        "15% Review Confidence + 15% Review Risk"
    )


    # =====================================================
    # URL → DATASET RECOMMENDATION
    # =====================================================

    st.markdown(
        '<div class="section-title">🔄 Similar & Better-Rated Products</div>',
        unsafe_allow_html=True
    )


    if rating is None:

        st.warning(
            "The product rating could not be extracted "
            "from this URL. Similar-product recommendations "
            "cannot be calculated reliably."
        )

    else:

        # =================================================
        # CREATE A COPY OF DATASET
        # =================================================

        recommendation_df = df.copy()


        # =================================================
        # FILTER SAME OR HIGHER RATING
        # =================================================

        better_products = recommendation_df[

            recommendation_df[
                "Rating of the product"
            ]

            >=

            rating

        ].copy()


        # Remove impossible ratings

        better_products = better_products[

            better_products[
                "Rating of the product"
            ]

            > 0

        ]


        if better_products.empty:

            st.warning(
                "No products in your dataset have "
                "the same or higher rating."
            )

        else:

            # =============================================
            # CREATE URL PRODUCT PROFILE
            # =============================================

            url_profile = {}


            # =============================================
            # NUMERICAL FEATURES
            # =============================================

            for column in numeric_features:

                if column == "Rating of the product":

                    url_profile[column] = rating

                elif (
                    column
                    ==
                    "Customer review sentiment score (overall)"
                ):

                    url_profile[column] = (
                        average_sentiment
                    )

                else:

                    # For information not available
                    # from the URL, use dataset median.

                    url_profile[column] = (
                        df[column].median()
                    )


            # =============================================
            # CATEGORICAL FEATURES
            # =============================================

            for column in categorical_features:

                if (
                    column
                    ==
                    "Brand of the product"
                    and
                    brand
                ):

                    url_profile[column] = brand

                else:

                    url_profile[column] = (
                        df[column].mode().iloc[0]
                    )


            # =============================================
            # URL PRODUCT DATAFRAME
            # =============================================

            url_product_df = pd.DataFrame(
                [url_profile]
            )


            # =============================================
            # TRANSFORM URL PRODUCT
            # =============================================

            url_feature_vector = (
                preprocessor.transform(
                    url_product_df[
                        required_columns
                    ]
                )
            )


            # =============================================
            # CALCULATE SIMILARITY
            # =============================================

            url_similarities = (
                cosine_similarity(
                    url_feature_vector,
                    feature_matrix
                )[0]
            )


            recommendation_df[
                "URL Similarity"
            ] = url_similarities


            # =============================================
            # FILTER SAME/HIGHER RATING
            # =============================================

            better_products = recommendation_df[

                recommendation_df[
                    "Rating of the product"
                ]

                >=

                rating

            ].copy()


            # =============================================
            # RATING SCORE
            # =============================================

            better_products[
                "Rating Score"
            ] = (

                better_products[
                    "Rating of the product"
                ]

                /

                5

            )


            # =============================================
            # SENTIMENT SCORE
            # =============================================

            better_products[
                "Sentiment Score"
            ] = (

                better_products[
                    "Customer review sentiment score (overall)"
                ]

                +

                1

            ) / 2


            # =============================================
            # BRAND MATCH
            # =============================================

            if brand:

                better_products[
                    "Brand Match"
                ] = (

                    better_products[
                        "Brand of the product"
                    ]

                    .str.lower()

                    ==

                    brand.lower()

                ).astype(float)

            else:

                better_products[
                    "Brand Match"
                ] = 0


            # =============================================
            # RATING IMPROVEMENT
            # =============================================

            rating_difference = (

                better_products[
                    "Rating of the product"
                ]

                -

                rating

            )


            # Normalize rating improvement

            better_products[
                "Rating Advantage"
            ] = (

                rating_difference
                /
                5

            ).clip(
                lower=0,
                upper=1
            )


            # =============================================
            # FINAL ALTERNATIVE SCORE
            # =============================================

            better_products[
                "Alternative Score"
            ] = (

                0.50
                *
                better_products[
                    "URL Similarity"
                ]

                +

                0.20
                *
                better_products[
                    "Rating Score"
                ]

                +

                0.15
                *
                better_products[
                    "Sentiment Score"
                ]

                +

                0.10
                *
                better_products[
                    "Rating Advantage"
                ]

                +

                0.05
                *
                better_products[
                    "Brand Match"
                ]

            )


            # =============================================
            # SORT
            # =============================================

            better_products = (

                better_products

                .sort_values(

                    "Alternative Score",

                    ascending=False

                )

            )


            # =============================================
            # TOP 5
            # =============================================

            top_url_recommendations = (

                better_products

                .head(5)

                .copy()

            )


            # =============================================
            # DISPLAY TABLE
            # =============================================

            display = (
                top_url_recommendations[
                    [
                        "Product ID",
                        "Brand of the product",
                        "Rating of the product",
                        "Customer review sentiment score (overall)",
                        "URL Similarity",
                        "Alternative Score"
                    ]
                ]
                .copy()
            )


            display.columns = [

                "Product ID",
                "Brand",
                "Rating",
                "Review Sentiment",
                "Similarity",
                "Recommendation Score"

            ]


            display[
                "Rating"
            ] = (

                display[
                    "Rating"
                ]

                .round(1)

            )


            display[
                "Review Sentiment"
            ] = (

                display[
                    "Review Sentiment"
                ]

                .round(2)

            )


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


            # =============================================
            # BEST URL RECOMMENDATION
            # =============================================

            best_url_product = (
                top_url_recommendations.iloc[0]
            )


            best_rating = (
                best_url_product[
                    "Rating of the product"
                ]
            )


            best_brand = (
                best_url_product[
                    "Brand of the product"
                ]
            )


            best_id = int(
                best_url_product[
                    "Product ID"
                ]
            )


            best_similarity = (
                best_url_product[
                    "URL Similarity"
                ]
                *
                100
            )


            best_score = (
                best_url_product[
                    "Alternative Score"
                ]
                *
                100
            )


            rating_difference = (

                best_rating

                -

                rating

            )


            # =============================================
            # BEST ALTERNATIVE MESSAGE
            # =============================================

            st.success(

                f"🏆 Best Alternative: "

                f"{best_brand} "

                f"(Product ID {best_id}) "

                f"— ⭐ {best_rating:.1f}/5"

            )


            if rating_difference > 0:

                st.info(

                    f"⭐ This product has a "

                    f"{rating_difference:.1f} point "

                    f"higher rating than the product "

                    f"from your URL."

                )

            else:

                st.info(

                    "⭐ This product has the same rating "
                    "as the product from your URL."

                )


            # =============================================
            # RECOMMENDATION SCORE
            # =============================================

            st.caption(

                f"Similarity with URL product: "
                f"{best_similarity:.2f}% | "

                f"Recommendation Score: "
                f"{best_score:.2f}%"

            )


            # =============================================
            # CHART
            # =============================================

            st.subheader(
                "📊 Similarity with URL Product"
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
                )

                [

                    "Recommendation Score"

                ]

            )


            st.bar_chart(
                chart_data
            )


            st.caption(

                "Recommendation Score combines "
                "product similarity, rating, review "
                "sentiment, rating advantage and brand match."

            )


# =========================================================
# DIVIDER
# =========================================================

st.markdown("---")


# =========================================================
# DATASET PRODUCT RECOMMENDATION
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
        "the most similar products."
    )


    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    d1, d2, d3 = st.columns(3)


    with d1:

        st.metric(
            "📦 Total Products",
            len(df)
        )


    with d2:

        st.metric(
            "🔢 Features Used",
            len(required_columns)
        )


    with d3:

        st.metric(
            "⭐ Average Rating",
            f"{df['Rating of the product'].mean():.2f}/5"
        )


    # =====================================================
    # PRODUCT SELECTION
    # =====================================================

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


    p1, p2, p3 = st.columns(3)


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
            "💬 Review Sentiment",
            f"{selected_product['Customer review sentiment score (overall)']:.2f}"
        )


    # =====================================================
    # DATASET SIMILARITY
    # =====================================================

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

            !=

            selected_id

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


    # =====================================================
    # PRODUCT QUALITY
    # =====================================================

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

        0.5
        *
        rating_score

        +

        0.5
        *
        sentiment_score

    )


    # =====================================================
    # FINAL RECOMMENDATION SCORE
    # =====================================================

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


    # =====================================================
    # TOP 5 SIMILAR PRODUCTS
    # =====================================================

    st.subheader(
        "⭐ Top 5 Similar Products"
    )


    display = top_recommendations[

        [

            "Product ID",

            "Brand of the product",

            "Rating of the product",

            "Customer review sentiment score (overall)",

            "Similarity",

            "Recommendation Score"

        ]

    ].copy()


    display.columns = [

        "Product ID",

        "Brand",

        "Rating",

        "Review Sentiment",

        "Similarity",

        "Recommendation Score"

    ]


    display[
        "Rating"
    ] = (

        display[
            "Rating"
        ]

        .round(1)

    )


    display[
        "Review Sentiment"
    ] = (

        display[
            "Review Sentiment"
        ]

        .round(2)

    )


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


    # =====================================================
    # RECOMMENDATION CHART
    # =====================================================

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
        )

        [

            "Recommendation Score"

        ]

    )


    st.bar_chart(
        chart_data
    )


    # =====================================================
    # BEST RECOMMENDATION
    # =====================================================

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


    # =====================================================
    # SYSTEM EVALUATION
    # =====================================================

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

        "These values represent recommendation and "
        "similarity scores, not classification accuracy."

    )


    # =====================================================
    # HOW THE SYSTEM WORKS
    # =====================================================

    with st.expander(
        "💡 How does the recommendation system work?"
    ):

        st.write(
            """
            The system uses a content-based recommendation approach.

            For a pasted product URL, the system first attempts
            to extract the product name, rating, brand and
            customer reviews.

            Customer reviews are analyzed using sentiment analysis.

            The URL product is then represented using the
            information available from the webpage.

            Missing product attributes are represented using
            typical values from the uploaded dataset.

            The URL product profile is transformed using the
            same preprocessing pipeline as the dataset.

            Cosine similarity is then used to compare the
            URL product with products in the dataset.

            Products with the same or higher rating are
            prioritized.

            The final alternative score considers:

            50% Product Similarity

            20% Rating Quality

            15% Review Sentiment

            10% Rating Advantage

            5% Brand Match
            """
        )


    # =====================================================
    # COMPLETE DATASET
    # =====================================================

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
    "E-Commerce Product Recommendation Using Customer Reviews "
    "| Content-Based Recommendation | Sentiment Analysis"
)
