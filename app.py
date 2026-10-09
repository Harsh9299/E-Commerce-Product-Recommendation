import streamlit as st
import pandas as pd
import requests
import re
import json
import time

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
# DARK THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #1f2328;
}

.main {
    padding-top: 1rem;
}

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

.section-title {
    font-size: 30px;
    font-weight: 900;
    color: #f2f3f4;
    margin-top: 30px;
    margin-bottom: 18px;
}

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

.stMarkdown,
.stText {
    color: #d5d8db;
}

h1, h2, h3 {
    color: #f1f2f3 !important;
}

.stTextInput input {
    background: #292e34 !important;
    color: #f1f2f3 !important;
    border: 1px solid #555c63 !important;
    border-radius: 8px !important;
}

.stTextInput input::placeholder {
    color: #8e959c !important;
}

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

[data-testid="stDataFrame"] {
    border: 1px solid #3a4046;
    border-radius: 10px;
}

[data-testid="stExpander"] {
    border: 1px solid #3a4046;
    border-radius: 10px;
    background: #24282d;
}

hr {
    border: none;
    border-top: 1px solid #3a4046;
    margin: 32px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT HEADER
# =========================================================

st.markdown("""
<div class="app-header">
<h1>🛒 E-Commerce Product Recommendation Using Customer Reviews</h1>
<p>Analyze customer feedback and discover similar products with equal or higher ratings.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SENTIMENT ANALYZER
# =========================================================

analyzer = SentimentIntensityAnalyzer()


# =========================================================
# REQUEST HEADERS
# =========================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1"
}


# =========================================================
# CLEAN PRODUCT URL
# =========================================================

def clean_product_url(url):

    url = (url or "").strip()

    parsed = urlparse(url)
    host = parsed.netloc.lower()

    # Clean Amazon URL
    if "amazon." in host:

        match = re.search(
            r"/(?:dp|gp/product|product)/([A-Z0-9]{10})(?:/|$)",
            parsed.path,
            flags=re.IGNORECASE
        )

        if match:
            return (
                f"https://{parsed.netloc}/dp/"
                f"{match.group(1).upper()}"
            )

    return url


# =========================================================
# SAFE WEB REQUEST
# =========================================================

def request_webpage(url, timeout=25, retries=3):

    clean_url = clean_product_url(url)

    last_status = None

    for attempt in range(retries):

        try:

            response = requests.get(
                clean_url,
                headers=HEADERS,
                timeout=timeout,
                allow_redirects=True
            )

            last_status = response.status_code

            if response.status_code == 200:
                return response

            if response.status_code in (
                429, 500, 502, 503, 504
            ):

                if attempt < retries - 1:
                    time.sleep(
                        1.5 * (attempt + 1)
                    )
                    continue

            if response.status_code in (401, 403):

                raise RuntimeError(
                    f"The website blocked automated access "
                    f"(HTTP {response.status_code})."
                )

            raise RuntimeError(
                f"The website returned HTTP "
                f"{response.status_code}."
            )

        except requests.exceptions.RequestException as e:

            if attempt < retries - 1:
                time.sleep(
                    1.5 * (attempt + 1)
                )
                continue

            raise RuntimeError(
                f"Network error while opening the "
                f"product page: {e}"
            )

    if last_status == 503:

        raise RuntimeError(
            "The shopping website returned HTTP 503 "
            "(Service Unavailable). The website is "
            "temporarily refusing automated requests."
        )

    raise RuntimeError(
        f"The product page could not be accessed "
        f"(HTTP {last_status})."
    )


# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

def analyze_sentiment(text):

    score = analyzer.polarity_scores(
        text
    )["compound"]

    if score >= 0.05:
        return "Positive", score

    if score <= -0.05:
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
# TEXT CLEANING
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
# PRODUCT DATA EXTRACTION
# =========================================================

def extract_product_data(url):

    url = clean_product_url(url)

    response = request_webpage(
        url,
        timeout=25,
        retries=3
    )

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

    # -----------------------------------------------------
    # PAGE TEXT
    # -----------------------------------------------------

    page_text = soup.get_text(
        " ",
        strip=True
    )

    # -----------------------------------------------------
    # PRODUCT RATING
    # -----------------------------------------------------

    rating = None

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

                            value = float(value)

                            if 0 <= value <= 5:
                                rating = value
                                break

                        except:
                            pass

            if rating is not None:
                break

        except:
            pass

    if rating is None:

        rating = extract_rating(
            page_text
        )

    # -----------------------------------------------------
    # REVIEWS
    # -----------------------------------------------------

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
                reviews.append(text)

    reviews = list(
        dict.fromkeys(reviews)
    )

    reviews = reviews[:30]

    # -----------------------------------------------------
    # BRAND
    # -----------------------------------------------------

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
# FIND RELATED PRODUCTS
# =========================================================

def find_related_products(
    soup,
    base_url
):

    candidates = []

    domain = urlparse(
        base_url
    ).netloc

    sections = soup.find_all(
        string=re.compile(
            r"(similar|related|recommended|"
            r"customers also|frequently bought|"
            r"you may also like|people also)",
            re.IGNORECASE
        )
    )

    containers = []

    for section in sections:

        parent = section.parent

        if parent:

            containers.append(parent)

            if parent.parent:
                containers.append(
                    parent.parent
                )

    # -----------------------------------------------------
    # RELATED SECTIONS
    # -----------------------------------------------------

    for container in containers:

        links = container.find_all(
            "a",
            href=True
        )

        for link in links:

            href = link.get("href")

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

            rating = extract_rating(
                link.get_text(
                    " ",
                    strip=True
                )
            )

            candidates.append({
                "Product": text,
                "URL": full_url,
                "Rating": rating,
                "Source": "Related products"
            })

    # -----------------------------------------------------
    # GENERAL PRODUCT LINKS
    # -----------------------------------------------------

    if len(candidates) < 5:

        for link in soup.find_all(
            "a",
            href=True
        ):

            href = link.get("href")

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

            rating = extract_rating(
                link.get_text(
                    " ",
                    strip=True
                )
            )

            candidates.append({
                "Product": text,
                "URL": full_url,
                "Rating": rating,
                "Source": "Product page"
            })

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    unique = {}

    for item in candidates:
        unique[item["URL"]] = item

    return list(
        unique.values()
    )[:20]


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

    query = (
        f'"{product_title}" similar products'
    )

    search_url = (
        "https://www.google.com/search?q="
        + quote_plus(query)
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

            href = link.get("href")

            text = clean_text(
                link.get_text(
                    " ",
                    strip=True
                ),
                200
            )

            if not text or len(text) < 10:
                continue

            if href.startswith("/url?q="):

                href = href.split(
                    "/url?q=",
                    1
                )[1].split(
                    "&",
                    1
                )[0]

            if not href.startswith("http"):
                continue

            if urlparse(
                href
            ).netloc == domain:
                continue

            results.append({
                "Product": text,
                "URL": href,
                "Rating": extract_rating(text),
                "Source": "Web search"
            })

        unique = {}

        for item in results:
            unique[item["URL"]] = item

        return list(
            unique.values()
        )[:10]

    except:
        return []


# =========================================================
# FETCH PRODUCT RATING
# =========================================================

def fetch_product_rating(url):

    try:

        response = request_webpage(
            url,
            timeout=15,
            retries=2
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

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

                            value = float(value)

                            if 0 <= value <= 5:
                                return value

            except:
                pass

        page_text = soup.get_text(
            " ",
            strip=True
        )

        return extract_rating(
            page_text
        )

    except:
        return None


# =========================================================
# SIMILAR PRODUCT RECOMMENDATION
# =========================================================

def get_similar_products(
    product_data
):

    soup = product_data["soup"]
    product_url = product_data["page_url"]
    product_title = product_data["title"]
    original_rating = product_data["rating"]

    candidates = find_related_products(
        soup,
        product_url
    )

    # Web search fallback
    if len(candidates) < 5:

        candidates.extend(
            web_search_similar_products(
                product_title,
                product_url
            )
        )

    # Remove duplicates
    unique = {}

    for item in candidates:
        unique[item["URL"]] = item

    candidates = list(
        unique.values()
    )

    final_products = []

    original_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            (product_title or "").lower()
        )
    )

    # Check maximum 15 candidates
    for candidate in candidates[:15]:

        rating = candidate.get(
            "Rating"
        )

        if rating is None:

            rating = fetch_product_rating(
                candidate["URL"]
            )

        candidate["Rating"] = rating

        # No rating = ignore
        if rating is None:
            continue

        # Same or higher rating only
        if (
            original_rating is not None
            and rating < original_rating
        ):
            continue

        candidate_words = set(
            re.findall(
                r"[a-zA-Z0-9]+",
                candidate["Product"].lower()
            )
        )

        if original_words and candidate_words:

            similarity = (
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

            similarity = 0

        rating_score = rating / 5

        recommendation_score = (
            0.55 * similarity
            +
            0.45 * rating_score
        )

        candidate["Similarity"] = similarity

        candidate[
            "Recommendation Score"
        ] = recommendation_score

        final_products.append(
            candidate
        )

    # Sort highest recommendation score first
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
    placeholder="https://www.amazon.in/dp/XXXXXXXXXX"
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
        product_url.startswith("http://")
        or
        product_url.startswith("https://")
    ):

        st.error(
            "Please enter a valid URL beginning "
            "with http:// or https://"
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
# DISPLAY RESULTS
# =========================================================

if "product_data" in st.session_state:

    product_data = st.session_state[
        "product_data"
    ]

    title = product_data["title"]
    rating = product_data["rating"]
    brand = product_data["brand"]
    reviews = product_data["reviews"]

    # -----------------------------------------------------
    # PRODUCT INFORMATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📦 Product Information</div>',
        unsafe_allow_html=True
    )

    if title:
        st.subheader(title)
    else:
        st.info(
            "Product name could not be extracted."
        )

    c1, c2, c3 = st.columns(3)

    # -----------------------------------------------------
    # PRODUCT RATING
    # -----------------------------------------------------

    with c1:

        st.markdown(
            "⭐ **Product Rating**"
        )

        if rating is not None:

            if rating >= 4.0:

                st.markdown(
                    f"""
                    <div style="
                        background:#263d2d;
                        border-left:5px solid #4caf50;
                        padding:12px 16px;
                        border-radius:8px;
                        font-size:32px;
                        font-weight:800;
                        color:#6ee781;
                    ">
                        {rating:.1f}/5
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif rating >= 3.0:

                st.markdown(
                    f"""
                    <div style="
                        background:#403c24;
                        border-left:5px solid #d4b84c;
                        padding:12px 16px;
                        border-radius:8px;
                        font-size:32px;
                        font-weight:800;
                        color:#e6cf68;
                    ">
                        {rating:.1f}/5
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div style="
                        background:#40292b;
                        border-left:5px solid #d9534f;
                        padding:12px 16px;
                        border-radius:8px;
                        font-size:32px;
                        font-weight:800;
                        color:#ff7773;
                    ">
                        {rating:.1f}/5
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.markdown(
                """
                <div style="
                    color:#aeb4ba;
                    font-size:28px;
                    font-weight:700;
                ">
                    Not found
                </div>
                """,
                unsafe_allow_html=True
            )

    # -----------------------------------------------------
    # REVIEWS
    # -----------------------------------------------------

    with c2:

        st.metric(
            "💬 Reviews Analyzed",
            len(reviews)
        )

    # -----------------------------------------------------
    # BRAND
    # -----------------------------------------------------

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
            analyze_sentiment(review)
        )

        sentiment_scores.append(score)

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
            sum(sentiment_scores)
            /
            len(sentiment_scores)
        )

    else:

        average_sentiment = 0

    if total_reviews:

        positive_percentage = (
            positive / total_reviews * 100
        )

        neutral_percentage = (
            neutral / total_reviews * 100
        )

        negative_percentage = (
            negative / total_reviews * 100
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

        chart = pd.DataFrame(
            {
                "Reviews": [
                    positive,
                    neutral,
                    negative
                ]
            },
            index=[
                "Positive",
                "Neutral",
                "Negative"
            ]
        )

        st.bar_chart(chart)

    # =====================================================
    # BUY SCORE
    # =====================================================

    if rating is not None:
        rating_score = rating / 5
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

    buy_percentage = buy_score * 100

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

    # -----------------------------------------------------
    # FINAL DECISION BASED ON RATING
    # -----------------------------------------------------

    if rating is not None and rating >= 4.0:

        st.markdown(
            """
            <div style="
                background:#174d3b;
                border-left:6px solid #35d07f;
                padding:18px 20px;
                border-radius:10px;
                color:#35d07f;
                font-size:20px;
                font-weight:800;
                margin-top:10px;
            ">
                🟢 MUST BUY
            </div>
            """,
            unsafe_allow_html=True
        )

    elif rating is not None and rating >= 3.0:

        st.markdown(
            """
            <div style="
                background:#4a4320;
                border-left:6px solid #e6cf68;
                padding:18px 20px;
                border-radius:10px;
                color:#e6cf68;
                font-size:20px;
                font-weight:800;
                margin-top:10px;
            ">
                🟡 CAN BUY
            </div>
            """,
            unsafe_allow_html=True
        )

    elif rating is not None:

        st.markdown(
            """
            <div style="
                background:#4a2528;
                border-left:6px solid #ff7773;
                padding:18px 20px;
                border-radius:10px;
                color:#ff7773;
                font-size:20px;
                font-weight:800;
                margin-top:10px;
            ">
                🔴 DON'T BUY
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "Product rating could not be detected, "
            "so a purchase decision cannot be determined."
        )

    st.caption(
        "Purchase decision: "
        "4.0+ = MUST BUY | "
        "3.0–3.9 = CAN BUY | "
        "Below 3.0 = DON'T BUY"
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
            "ratings could be found."
        )

        st.info(
            "The shopping website may load recommendations "
            "dynamically or block automated access."
        )

    else:

        display_rows = []

        for product in similar_products:

            display_rows.append(
                {
                    "Product":
                        product["Product"],

                    "Rating":
                        f"{product['Rating']:.1f}/5",

                    "Similarity":
                        f"{product['Similarity'] * 100:.1f}%",

                    "Recommendation Score":
                        f"{product['Recommendation Score'] * 100:.1f}%"
                }
            )

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

        best_rating = best["Rating"]

        if best_rating >= 4.0:

            st.markdown(
                f"""
                <div style="
                    background:#174d3b;
                    border-left:6px solid #35d07f;
                    padding:18px 20px;
                    border-radius:10px;
                    color:#35d07f;
                    font-size:18px;
                    font-weight:700;
                    margin-top:20px;
                ">
                    🏆 Best Alternative:
                    {best['Product']}
                    — ⭐ {best_rating:.1f}/5
                    <br>
                    🟢 MUST BUY
                </div>
                """,
                unsafe_allow_html=True
            )

        elif best_rating >= 3.0:

            st.markdown(
                f"""
                <div style="
                    background:#4a4320;
                    border-left:6px solid #e6cf68;
                    padding:18px 20px;
                    border-radius:10px;
                    color:#e6cf68;
                    font-size:18px;
                    font-weight:700;
                    margin-top:20px;
                ">
                    🏆 Best Alternative:
                    {best['Product']}
                    — ⭐ {best_rating:.1f}/5
                    <br>
                    🟡 CAN BUY
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div style="
                    background:#4a2528;
                    border-left:6px solid #ff7773;
                    padding:18px 20px;
                    border-radius:10px;
                    color:#ff7773;
                    font-size:18px;
                    font-weight:700;
                    margin-top:20px;
                ">
                    🏆 Best Alternative:
                    {best['Product']}
                    — ⭐ {best_rating:.1f}/5
                    <br>
                    🔴 DON'T BUY
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # RATING COMPARISON
        # -------------------------------------------------

        if rating is not None:

            difference = (
                best_rating - rating
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

        # -------------------------------------------------
        # RECOMMENDATION SCORE CHART
        # -------------------------------------------------

        st.subheader(
            "📊 Recommendation Scores"
        )

        chart_data = pd.DataFrame(
            {
                "Recommendation Score": [
                    p["Recommendation Score"] * 100
                    for p in similar_products
                ]
            },
            index=[
                p["Product"][:45]
                for p in similar_products
            ]
        )

        st.bar_chart(chart_data)

        st.caption(
            "Recommendation Score = "
            "55% product-title similarity + "
            "45% verified product rating."
        )

        # -------------------------------------------------
        # PRODUCT LINKS
        # -------------------------------------------------

        with st.expander(
            "🔗 View Product Links"
        ):

            for index, product in enumerate(
                similar_products,
                1
            ):

                st.markdown(
                    f"**{index}. {product['Product']}**"
                )

                st.write(
                    product["URL"]
                )

                st.divider()


# =========================================================
# HOW THE SYSTEM WORKS
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">💡 How This System Works</div>',
    unsafe_allow_html=True
)

st.write(
    """
1. The user pastes an e-commerce product URL.

2. The system cleans the URL and removes unnecessary
   tracking parameters when possible.

3. The webpage is fetched and parsed using Requests
   and BeautifulSoup.

4. Product name, rating, brand and customer reviews
   are extracted.

5. Customer reviews are analyzed using VADER
   sentiment analysis.

6. Positive, neutral and negative review percentages
   are calculated.

7. A Buy Score is calculated using rating, sentiment,
   review confidence and negative-review risk.

8. The final purchase decision is based directly
   on the product rating.

9. Similar products are collected from the product
   page and web search.

10. Candidate product ratings are verified.

11. Products with ratings lower than the original
    product are removed.

12. Remaining products are ranked using:

    Recommendation Score =
    55% Similarity + 45% Rating

13. The best alternative is displayed to the user.

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
