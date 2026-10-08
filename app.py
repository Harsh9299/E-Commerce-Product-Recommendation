import streamlit as st
import requests
import re
import json
from urllib.parse import urljoin, urlparse, quote_plus

from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="E-Commerce Product Recommendation",
    page_icon="🛒",
    layout="wide"
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
   MAIN HEADER
   ===================================================== */

.app-header {
    padding: 28px 30px;
    border-radius: 14px;
    background: #292e34;
    margin-bottom: 30px;
    border: 1px solid #41474e;
}

.app-header h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 900;
    color: #ffffff;
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
}


/* =====================================================
   METRICS
   ===================================================== */

[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 4px !important;
    box-shadow: none !important;
}

[data-testid="stMetricLabel"] {
    color: #aeb4ba !important;
}

[data-testid="stMetricValue"] {
    color: #f2f3f4 !important;
    font-weight: 800 !important;
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
   BUTTON
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
   TABLE
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
   DIVIDER
   ===================================================== */

hr {
    border: none;
    border-top: 1px solid #3a4046;
    margin: 32px 0;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# PROJECT HEADER
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


# =========================================================
# SENTIMENT ANALYZER
# =========================================================

analyzer = SentimentIntensityAnalyzer()


# =========================================================
# HTTP HEADERS
# =========================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/142.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    )
}


# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

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
# RATING EXTRACTION
# =========================================================

def extract_rating(text):

    if not text:
        return None

    patterns = [

        r"([0-5](?:\.[0-9])?)\s*(?:out of\s*5|/5)",

        r"([0-5](?:\.[0-9])?)\s*stars?",

        r"rating[^0-9]{0,20}([0-5](?:\.[0-9])?)"

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
# CLEAN TEXT
# =========================================================

def clean_text(text, max_length=300):

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text[:max_length]


# =========================================================
# PRODUCT INFORMATION EXTRACTION
# =========================================================

def extract_product_data(url):

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

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

        title = clean_text(
            h1.get_text(
                " ",
                strip=True
            ),
            250
        )


    if not title:

        meta = soup.find(
            "meta",
            property="og:title"
        )

        if meta:

            title = clean_text(
                meta.get("content"),
                250
            )


    if not title and soup.title:

        title = clean_text(
            soup.title.get_text(
                " ",
                strip=True
            ),
            250
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

    rating = None

    # JSON-LD
    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    for script in scripts:

        try:

            data = json.loads(
                script.string or script.get_text()
            )

            objects = (
                data
                if isinstance(data, list)
                else [data]
            )

            for obj in objects:

                if not isinstance(obj, dict):
                    continue

                aggregate = obj.get(
                    "aggregateRating"
                )

                if isinstance(
                    aggregate,
                    dict
                ):

                    value = aggregate.get(
                        "ratingValue"
                    )

                    if value:

                        try:

                            rating = float(
                                value
                            )

                            if 0 <= rating <= 5:
                                break

                        except:

                            pass

            if rating is not None:
                break

        except:

            pass


    # Regular page extraction

    if rating is None:

        rating = extract_rating(
            page_text
        )


    # =====================================================
    # REVIEWS
    # =====================================================

    reviews = []

    review_selectors = [

        '[itemprop="reviewBody"]',

        '[data-hook="review-body"]',

        '[class*="review-text"]',

        '[class*="reviewText"]',

        '[class*="review-content"]',

        '[class*="reviewContent"]',

        '[class*="review"]',

        '[id*="review"]'

    ]


    for selector in review_selectors:

        elements = soup.select(
            selector
        )

        for element in elements:

            text = clean_text(
                element.get_text(
                    " ",
                    strip=True
                ),
                1000
            )

            if len(text) >= 40:

                reviews.append(
                    text
                )


    # Remove duplicates

    reviews = list(
        dict.fromkeys(
            reviews
        )
    )


    # Maximum 30 reviews

    reviews = reviews[:30]


    # =====================================================
    # BRAND
    # =====================================================

    brand = None

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
        "Frontech",
        "Logitech",
        "Boat",
        "JBL",
        "Realme",
        "Oppo",
        "Vivo",
        "Levi's",
        "Pepe Jeans",
        "Flying Machine",
        "Roadster"

    ]


    if title:

        title_lower = title.lower()

        for candidate in known_brands:

            if candidate.lower() in title_lower:

                brand = candidate

                break


    return {

        "title": title,

        "rating": rating,

        "brand": brand,

        "reviews": reviews,

        "soup": soup,

        "page_url": url

    }


# =========================================================
# FIND RELATED PRODUCTS ON THE SAME PAGE
# =========================================================

def find_related_products(
    soup,
    base_url,
    original_rating
):

    candidates = []

    domain = urlparse(
        base_url
    ).netloc


    # =====================================================
    # RELATED/SIMILAR SECTION SELECTORS
    # =====================================================

    sections = soup.find_all(

        string=re.compile(

            r"(similar|related|recommended|"
            r"customers also|frequently bought|"
            r"you may also like|people also)",
            re.IGNORECASE

        )

    )


    possible_containers = []


    for section in sections:

        parent = section.parent

        if parent:

            possible_containers.append(
                parent
            )

            if parent.parent:

                possible_containers.append(
                    parent.parent
                )


    # =====================================================
    # SEARCH LINKS INSIDE RELATED AREAS
    # =====================================================

    for container in possible_containers:

        links = container.find_all(
            "a",
            href=True
        )

        for link in links:

            href = link.get(
                "href"
            )

            text = clean_text(
                link.get_text(
                    " ",
                    strip=True
                ),
                200
            )

            if not text or len(text) < 5:
                continue

            full_url = urljoin(
                base_url,
                href
            )

            if urlparse(
                full_url
            ).netloc != domain:

                continue

            if full_url == base_url:
                continue

            candidate_rating = extract_rating(
                link.get_text(
                    " ",
                    strip=True
                )
            )

            candidates.append({

                "Product":
                    text,

                "URL":
                    full_url,

                "Rating":
                    candidate_rating,

                "Source":
                    "Related products"

            })


    # =====================================================
    # GENERAL PRODUCT LINKS FALLBACK
    # =====================================================

    if len(candidates) < 5:

        links = soup.find_all(
            "a",
            href=True
        )

        for link in links:

            href = link.get(
                "href"
            )

            text = clean_text(
                link.get_text(
                    " ",
                    strip=True
                ),
                200
            )

            if not text or len(text) < 15:
                continue

            full_url = urljoin(
                base_url,
                href
            )

            if urlparse(
                full_url
            ).netloc != domain:

                continue

            # Product-like URLs

            href_lower = href.lower()

            if not any(
                word in href_lower
                for word in [
                    "/product",
                    "/dp/",
                    "/item",
                    "/p/",
                    "/pd/",
                    "/buy"
                ]
            ):

                continue

            if full_url == base_url:
                continue

            candidate_rating = extract_rating(
                link.get_text(
                    " ",
                    strip=True
                )
            )

            candidates.append({

                "Product":
                    text,

                "URL":
                    full_url,

                "Rating":
                    candidate_rating,

                "Source":
                    "Product page"

            })


    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    unique = {}

    for item in candidates:

        unique[
            item["URL"]
        ] = item


    candidates = list(
        unique.values()
    )


    return candidates[:20]


# =========================================================
# WEB SEARCH FALLBACK
# =========================================================

def web_search_similar_products(
    product_title,
    product_url
):

    if not product_title:

        return []


    domain = urlparse(
        product_url
    ).netloc


    # Remove common title noise

    search_title = re.sub(
        r"\s+",
        " ",
        product_title
    ).strip()


    query = (
        f'"{search_title}" '
        f'similar products'
    )


    search_url = (
        "https://www.google.com/search?q="
        +
        quote_plus(query)
    )


    try:

        response = requests.get(
            search_url,
            headers=HEADERS,
            timeout=15
        )

        if response.status_code != 200:

            return []


        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )


        results = []


        for link in soup.find_all(
            "a",
            href=True
        ):

            href = link.get(
                "href"
            )

            text = clean_text(
                link.get_text(
                    " ",
                    strip=True
                ),
                200
            )


            if not text or len(text) < 10:

                continue


            if href.startswith(
                "/url?q="
            ):

                href = href.split(
                    "/url?q=",
                    1
                )[1].split(
                    "&",
                    1
                )[0]


            if not href.startswith(
                "http"
            ):

                continue


            if urlparse(
                href
            ).netloc == domain:

                continue


            results.append({

                "Product":
                    text,

                "URL":
                    href,

                "Rating":
                    extract_rating(
                        text
                    ),

                "Source":
                    "Web search"

            })


        # Remove duplicates

        unique = {}

        for item in results:

            unique[
                item["URL"]
            ] = item


        return list(
            unique.values()
        )[:10]


    except:

        return []


# =========================================================
# FETCH PRODUCT RATING
# =========================================================

def fetch_product_rating(
    url
):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        if response.status_code != 200:

            return None


        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )


        # JSON-LD rating

        scripts = soup.find_all(
            "script",
            type="application/ld+json"
        )


        for script in scripts:

            try:

                data = json.loads(
                    script.string or script.get_text()
                )

                objects = (
                    data
                    if isinstance(data, list)
                    else [data]
                )


                for obj in objects:

                    if not isinstance(
                        obj,
                        dict
                    ):

                        continue


                    aggregate = obj.get(
                        "aggregateRating"
                    )


                    if isinstance(
                        aggregate,
                        dict
                    ):

                        value = aggregate.get(
                            "ratingValue"
                        )


                        if value:

                            value = float(
                                value
                            )


                            if 0 <= value <= 5:

                                return value

            except:

                pass


        text = soup.get_text(
            " ",
            strip=True
        )


        return extract_rating(
            text
        )


    except:

        return None


# =========================================================
# GET PRODUCT RECOMMENDATIONS
# =========================================================

def get_similar_products(
    product_data
):

    soup = product_data[
        "soup"
    ]

    product_url = product_data[
        "page_url"
    ]

    product_title = product_data[
        "title"
    ]

    original_rating = product_data[
        "rating"
    ]


    # First: related products from page

    candidates = find_related_products(
        soup,
        product_url,
        original_rating
    )


    # Second: web search fallback

    if len(candidates) < 5:

        search_results = (
            web_search_similar_products(
                product_title,
                product_url
            )
        )

        candidates.extend(
            search_results
        )


    # Remove duplicates

    unique = {}

    for item in candidates:

        unique[
            item["URL"]
        ] = item


    candidates = list(
        unique.values()
    )


    # =====================================================
    # FETCH RATINGS
    # =====================================================

    final_products = []


    for candidate in candidates[:15]:

        rating = candidate.get(
            "Rating"
        )


        if rating is None:

            rating = fetch_product_rating(
                candidate["URL"]
            )


        candidate["Rating"] = rating


        # Keep only products with known ratings

        if rating is None:

            continue


        # If original rating exists,
        # prioritize same or higher rating.

        if (
            original_rating is not None
            and
            rating < original_rating
        ):

            continue


        # Rating score

        rating_score = (
            rating / 5
        )


        # Simple similarity score based
        # on title word overlap

        original_words = set(
            re.findall(
                r"[a-zA-Z0-9]+",
                (
                    product_title
                    or ""
                ).lower()
            )
        )


        candidate_words = set(
            re.findall(
                r"[a-zA-Z0-9]+",
                candidate[
                    "Product"
                ].lower()
            )
        )


        if original_words and candidate_words:

            overlap = (
                len(
                    original_words
                    &
                    candidate_words
                )
                /
                len(
                    original_words
                    |
                    candidate_words
                )
            )

        else:

            overlap = 0


        candidate[
            "Similarity"
        ] = overlap


        candidate[
            "Recommendation Score"
        ] = (

            0.55 * overlap

            +

            0.45 * rating_score

        )


        final_products.append(
            candidate
        )


    # =====================================================
    # SORT
    # =====================================================

    final_products.sort(
        key=lambda x:
            x["Recommendation Score"],
        reverse=True
    )


    return final_products[:5]


# =========================================================
# URL INPUT
# =========================================================

st.markdown(
    '<div class="section-title">🔗 Analyze a Product URL</div>',
    unsafe_allow_html=True
)

st.write(
    "Paste a product URL. No CSV or dataset upload is required."
)


product_url = st.text_input(
    "Product URL",
    placeholder="https://www.amazon.in/product/..."
)


analyze_button = st.button(
    "🔍 Analyze Product & Find Similar Products",
    type="primary",
    use_container_width=True
)


# =========================================================
# MAIN ANALYSIS
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

    else:

        with st.spinner(
            "Analyzing the product..."
        ):

            try:

                product_data = (
                    extract_product_data(
                        product_url
                    )
                )

                st.session_state[
                    "product_data"
                ] = product_data


            except Exception as e:

                st.error(
                    "Unable to analyze this product."
                )

                st.info(
                    f"Reason: {e}"
                )


# =========================================================
# DISPLAY PRODUCT RESULTS
# =========================================================

if "product_data" in st.session_state:

    product_data = st.session_state[
        "product_data"
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


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "⭐ Product Rating",
            (
                f"{rating:.1f}/5"
                if rating is not None
                else "Not found"
            )
        )


    with c2:

        st.metric(
            "💬 Reviews Analyzed",
            len(reviews)
        )


    with c3:

        st.metric(
            "🏷️ Brand",
            brand if brand else "Not detected"
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


    if total_reviews:

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

        chart = pd.DataFrame({

            "Reviews": [

                positive,
                neutral,
                negative

            ]

        }, index=[

            "Positive",
            "Neutral",
            "Negative"

        ])


        st.bar_chart(
            chart
        )


    # =====================================================
    # PURCHASE SCORE
    # =====================================================

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
            "✅ BUY — Strong rating and "
            "customer feedback."
        )

    elif buy_percentage >= 55:

        st.warning(
            "⚠️ MAYBE — Customer feedback is "
            "mixed or moderate."
        )

    else:

        st.error(
            "❌ NOT RECOMMENDED — Rating or "
            "customer sentiment is relatively weak."
        )


    # =====================================================
    # SIMILAR PRODUCTS
    # =====================================================

    st.markdown(
        '<div class="section-title">🔄 Similar & Better-Rated Products</div>',
        unsafe_allow_html=True
    )


    with st.spinner(
        "Finding similar products..."
    ):

        similar_products = (
            get_similar_products(
                product_data
            )
        )


    if not similar_products:

        st.warning(
            "No related products with verifiable "
            "ratings could be found from the available "
            "webpage information."
        )

        st.info(
            "This can happen when the shopping website "
            "loads recommendations dynamically or "
            "blocks automated access."
        )


    else:

        display_rows = []


        for product in similar_products:

            display_rows.append({

                "Product":
                    product[
                        "Product"
                    ],

                "Rating":
                    f"{product['Rating']:.1f}/5",

                "Similarity":
                    f"{product['Similarity'] * 100:.1f}%",

                "Recommendation Score":
                    f"{product['Recommendation Score'] * 100:.1f}%"

            })


        result_df = pd.DataFrame(
            display_rows
        )


        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # BEST ALTERNATIVE
        # =================================================

        best = similar_products[0]


        st.success(
            f"🏆 Best Alternative: "
            f"{best['Product']} "
            f"— ⭐ {best['Rating']:.1f}/5"
        )


        if rating is not None:

            difference = (
                best["Rating"]
                -
                rating
            )


            if difference > 0:

                st.info(
                    f"⭐ This alternative has a "
                    f"{difference:.1f} point higher "
                    f"rating than the product you pasted."
                )

            elif difference == 0:

                st.info(
                    "⭐ This alternative has the same "
                    "rating as your selected product."
                )


        st.subheader(
            "📊 Recommendation Scores"
        )


        chart_data = pd.DataFrame({

            "Recommendation Score":
                [
                    p[
                        "Recommendation Score"
                    ] * 100
                    for p in similar_products
                ]

        }, index=[

            p["Product"][:45]
            for p in similar_products

        ])


        st.bar_chart(
            chart_data
        )


        st.caption(
            "Recommendation Score combines product-title "
            "similarity and verified product rating."
        )


        # =================================================
        # PRODUCT LINKS
        # =================================================

        with st.expander(
            "🔗 View Product Links"
        ):

            for index, product in enumerate(
                similar_products,
                1
            ):

                st.markdown(
                    f"**{index}. "
                    f"{product['Product']}**"
                )

                st.write(
                    product["URL"]
                )

                st.divider()


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("---")


st.markdown(
    '<div class="section-title">💡 How This System Works</div>',
    unsafe_allow_html=True
)


st.write(
    """
    This version works completely from the pasted product URL.

    1. The product webpage is analyzed to extract the product
       name, rating, brand and customer reviews.

    2. Customer reviews are analyzed using sentiment analysis.

    3. A Purchase Score is calculated from product rating,
       review sentiment, review confidence and negative-review risk.

    4. The system searches for related or similar products
       available from the product page and web search.

    5. Product ratings are verified when possible.

    6. Products with the same or higher rating are prioritized.

    7. The final recommendation score combines product-title
       similarity and product rating.

    No CSV dataset is required.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "E-Commerce Product Recommendation Using Customer Reviews "
    "| URL-Based Product Analysis | Sentiment Analysis"
)
