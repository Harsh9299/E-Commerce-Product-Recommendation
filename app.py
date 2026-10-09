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
# GET AMAZON ASIN
# =========================================================

def get_amazon_asin(url):

    match = re.search(
        r"/(?:dp|gp/product|product)/([A-Z0-9]{10})",
        url,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return None


# =========================================================
# CHECK AMAZON PRODUCT PAGE
# =========================================================

def looks_like_amazon_product_page(text):

    if not text:
        return False

    lower = text.lower()

    indicators = [
        "add to cart",
        "buy now",
        "customer reviews",
        "customer review",
        "average customer review",
        "product details",
        "about this item",
        "delivery",
        "in stock"
    ]

    score = sum(
        1 for item in indicators
        if item in lower
    )

    return score >= 2


# =========================================================
# SAFE WEB REQUEST
# =========================================================

def request_webpage(url, timeout=30, retries=3):

    clean_url = clean_product_url(url)

    parsed = urlparse(clean_url)
    host = parsed.netloc.lower()

    # =====================================================
    # AMAZON
    # =====================================================

    if "amazon." in host:

        amazon_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/142.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,image/avif,image/webp,"
                "*/*;q=0.8"
            ),
            "Accept-Language": "en-IN,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        }

        session = requests.Session()
        session.headers.update(amazon_headers)

        for attempt in range(retries):

            try:

                response = session.get(
                    clean_url,
                    timeout=timeout,
                    allow_redirects=True
                )

                if response.status_code == 200:

                    if looks_like_amazon_product_page(
                        response.text
                    ):
                        return response

                if response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504
                ):

                    if attempt < retries - 1:

                        time.sleep(
                            2 * (attempt + 1)
                        )

                        continue

            except requests.exceptions.RequestException:

                if attempt < retries - 1:

                    time.sleep(
                        2 * (attempt + 1)
                    )

        # =================================================
        # JINA READER FALLBACK
        # =================================================

        jina_url = (
            "https://r.jina.ai/"
            + clean_url
        )

        try:

            jina_response = requests.get(
                jina_url,
                headers={
                    "User-Agent":
                        "Mozilla/5.0"
                },
                timeout=45
            )

            if (
                jina_response.status_code == 200
                and len(jina_response.text) > 1000
            ):

                return jina_response

        except requests.exceptions.RequestException:
            pass

        raise RuntimeError(
            "Amazon is blocking automated access to "
            "this product page. Please try again later "
            "or use another accessible product URL."
        )

    # =====================================================
    # OTHER WEBSITES
    # =====================================================

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
                429,
                500,
                502,
                503,
                504
            ):

                if attempt < retries - 1:

                    time.sleep(
                        1.5 * (attempt + 1)
                    )

                    continue

            if response.status_code in (
                401,
                403
            ):

                raise RuntimeError(
                    f"The website blocked automated access "
                    f"(HTTP {response.status_code})."
                )

        except requests.exceptions.RequestException as e:

            if attempt < retries - 1:

                time.sleep(
                    1.5 * (attempt + 1)
                )

                continue

            raise RuntimeError(
                f"Network error: {e}"
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

        r"([0-5](?:\.[0-9])?)\s*"
        r"(?:out\s*of\s*5|/5)",

        r"([0-5](?:\.[0-9])?)\s*stars?",

        r"rating[^0-9]{0,30}"
        r"([0-5](?:\.[0-9])?)"

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

            except (ValueError, TypeError):

                pass

    return None


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text, max_length=1000):

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()

    return text[:max_length]


# =========================================================
# EXTRACT JSON-LD DATA
# =========================================================

def get_json_ld(soup):

    data_items = []

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    for script in scripts:

        try:

            raw = (
                script.string
                or
                script.get_text()
            )

            if not raw:
                continue

            data = json.loads(raw)

            if isinstance(data, list):

                data_items.extend(data)

            else:

                data_items.append(data)

        except Exception:
            continue

    return data_items


# =========================================================
# EXTRACT PRODUCT DATA
# =========================================================

def extract_product_data(url):

    original_url = url.strip()

    clean_url = clean_product_url(
        original_url
    )

    asin = get_amazon_asin(
        clean_url
    )

    response = request_webpage(
        clean_url,
        timeout=45,
        retries=3
    )

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    page_text = soup.get_text(
        " ",
        strip=True
    )

    json_items = get_json_ld(
        soup
    )

    # =====================================================
    # PRODUCT TITLE
    # =====================================================

    title = None

    # JSON-LD
    for obj in json_items:

        if not isinstance(obj, dict):
            continue

        name = obj.get("name")

        if (
            name
            and isinstance(name, str)
            and len(name.strip()) > 5
        ):

            title = clean_text(
                name,
                300
            )

            break

    # H1
    if not title:

        h1 = soup.find("h1")

        if h1:

            candidate = clean_text(
                h1.get_text(
                    " ",
                    strip=True
                ),
                300
            )

            if (
                candidate
                and candidate.lower()
                not in (
                    "amazon",
                    "amazon.in"
                )
            ):

                title = candidate

    # OpenGraph
    if not title:

        meta = soup.find(
            "meta",
            property="og:title"
        )

        if meta:

            candidate = clean_text(
                meta.get("content"),
                300
            )

            if (
                candidate
                and candidate.lower()
                not in (
                    "amazon",
                    "amazon.in"
                )
            ):

                title = candidate

    # HTML title
    if not title and soup.title:

        candidate = clean_text(
            soup.title.get_text(
                " ",
                strip=True
            ),
            300
        )

        if (
            candidate
            and candidate.lower()
            not in (
                "amazon",
                "amazon.in"
            )
        ):

            title = candidate

    # Jina markdown title fallback
    if not title:

        for line in response.text.splitlines():

            line = line.strip()

            if line.startswith("# "):

                candidate = clean_text(
                    line[2:],
                    300
                )

                if (
                    candidate
                    and candidate.lower()
                    not in (
                        "amazon",
                        "amazon.in"
                    )
                ):

                    title = candidate
                    break

    # Amazon URL fallback
    if not title and asin:

        path = urlparse(
            clean_url
        ).path

        parts = path.strip(
            "/"
        ).split("/")

        if parts:

            slug = parts[0]

            if slug.lower() not in (
                "dp",
                "gp",
                "product"
            ):

                title = clean_text(
                    slug.replace(
                        "-",
                        " "
                    ).title(),
                    300
                )

    # =====================================================
    # RATING
    # =====================================================

    rating = None

    for obj in json_items:

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

            try:

                if value is not None:

                    value = float(value)

                    if 0 <= value <= 5:

                        rating = value
                        break

            except (ValueError, TypeError):

                pass

    # Rating from page
    if rating is None:

        rating = extract_rating(
            page_text
        )

    # =====================================================
    # REVIEWS
    # =====================================================

    reviews = []

    selectors = [

        '[itemprop="reviewBody"]',

        '[data-hook="review-body"]',

        '[data-hook="review-collapsed"]',

        '[class*="review-text"]',

        '[class*="reviewText"]',

        '[class*="review-content"]',

        '[class*="reviewContent"]'

    ]

    for selector in selectors:

        for element in soup.select(
            selector
        ):

            text = clean_text(
                element.get_text(
                    " ",
                    strip=True
                ),
                1200
            )

            if len(text) >= 25:

                reviews.append(
                    text
                )

    # Remove duplicates
    reviews = list(
        dict.fromkeys(
            reviews
        )
    )

    # =====================================================
    # JINA / PLAIN TEXT REVIEW FALLBACK
    # =====================================================

    if not reviews:

        lines = response.text.splitlines()

        review_keywords = [
            "verified purchase",
            "reviewed in india",
            "customer review",
            "customer reviews"
        ]

        for line in lines:

            line = clean_text(
                line,
                1200
            )

            lower = line.lower()

            if (
                len(line) >= 50
                and any(
                    keyword in lower
                    for keyword in review_keywords
                )
            ):

                reviews.append(
                    line
                )

    reviews = list(
        dict.fromkeys(
            reviews
        )
    )

    reviews = reviews[:30]

    # =====================================================
    # BRAND
    # =====================================================

    brand = None

    for obj in json_items:

        if not isinstance(obj, dict):
            continue

        brand_data = obj.get(
            "brand"
        )

        brand_name = None

        if isinstance(
            brand_data,
            dict
        ):

            brand_name = brand_data.get(
                "name"
            )

        elif isinstance(
            brand_data,
            str
        ):

            brand_name = brand_data

        if brand_name:

            brand = clean_text(
                brand_name,
                100
            )

            break

    # Known brand fallback
    if not brand and title:

        known_brands = [

            "Dell",
            "HP",
            "Lenovo",
            "ASUS",
            "Acer",
            "Logitech",
            "Zebronics",
            "Ant Esports",
            "Redragon",
            "Frontech",
            "Boat",
            "JBL",
            "Sony",
            "Samsung",
            "Apple",
            "OnePlus",
            "Realme",
            "Oppo",
            "Vivo",
            "Cosmic Byte",
            "Portronics",
            "Amazon Basics"

        ]

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
        "page_url": clean_url,
        "asin": asin,
        "raw_response": response.text
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

            containers.append(
                parent
            )

            if parent.parent:

                containers.append(
                    parent.parent
                )

    # Related sections
    for container in containers:

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
                250
            )

            if (
                not text
                or len(text) < 5
            ):
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

            candidates.append(
                {
                    "Product": text,
                    "URL": full_url,
                    "Rating": extract_rating(
                        link.get_text(
                            " ",
                            strip=True
                        )
                    ),
                    "Source":
                        "Related products"
                }
            )

    # General product links
    if len(candidates) < 5:

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
                250
            )

            if (
                not text
                or len(text) < 15
            ):
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

            candidates.append(
                {
                    "Product": text,
                    "URL": full_url,
                    "Rating": extract_rating(
                        link.get_text(
                            " ",
                            strip=True
                        )
                    ),
                    "Source":
                        "Product page"
                }
            )

    # Remove duplicates
    unique = {}

    for item in candidates:

        unique[
            item["URL"]
        ] = item

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
        f'"{product_title}" '
        f'similar products'
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

            href = link.get(
                "href"
            )

            text = clean_text(
                link.get_text(
                    " ",
                    strip=True
                ),
                250
            )

            if (
                not text
                or len(text) < 10
            ):
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

            results.append(
                {
                    "Product": text,
                    "URL": href,
                    "Rating":
                        extract_rating(text),
                    "Source":
                        "Web search"
                }
            )

        unique = {}

        for item in results:

            unique[
                item["URL"]
            ] = item

        return list(
            unique.values()
        )[:10]

    except Exception:

        return []


# =========================================================
# FETCH PRODUCT RATING
# =========================================================

def fetch_product_rating(url):

    try:

        clean_url = clean_product_url(
            url
        )

        response = request_webpage(
            clean_url,
            timeout=30,
            retries=2
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        json_items = get_json_ld(
            soup
        )

        # JSON-LD rating
        for obj in json_items:

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

                try:

                    value = float(value)

                    if 0 <= value <= 5:

                        return value

                except (
                    ValueError,
                    TypeError
                ):

                    pass

        # Text fallback
        page_text = soup.get_text(
            " ",
            strip=True
        )

        return extract_rating(
            page_text
        )

    except Exception:

        return None


# =========================================================
# SIMILAR PRODUCT RECOMMENDATION
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

        unique[
            item["URL"]
        ] = item

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

    for candidate in candidates[:15]:

        rating = candidate.get(
            "Rating"
        )

        if rating is None:

            rating = fetch_product_rating(
                candidate["URL"]
            )

        candidate["Rating"] = rating

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
                candidate[
                    "Product"
                ].lower()
            )
        )

        if (
            original_words
            and candidate_words
        ):

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

        rating_score = (
            rating / 5
        )

        recommendation_score = (
            0.55 * similarity
            +
            0.45 * rating_score
        )

        candidate[
            "Similarity"
        ] = similarity

        candidate[
            "Recommendation Score"
        ] = recommendation_score

        final_products.append(
            candidate
        )

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
    '<div class="section-title">'
    '🔗 Analyze a Product URL'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Paste a product URL. "
    "No CSV or dataset upload is required."
)

product_url = st.text_input(
    "Product URL",
    placeholder=(
        "https://www.amazon.in/dp/XXXXXXXXXX"
    )
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

    # Clear previous result
    st.session_state.pop(
        "product_data",
        None
    )

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

                # Do not accept an empty product
                if (
                    not product_data["title"]
                    and product_data["rating"]
                    is None
                    and not product_data["reviews"]
                ):

                    raise RuntimeError(
                        "No usable product information "
                        "could be extracted from this page."
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

    product_data = (
        st.session_state[
            "product_data"
        ]
    )

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
        '<div class="section-title">'
        '📦 Product Information'
        '</div>',
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

    # -----------------------------------------------------
    # RATING
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
            brand
            if brand
            else "Not detected"
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
            sum(sentiment_scores)
            /
            len(sentiment_scores)
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
        '<div class="section-title">'
        '💬 Customer Review Insights'
        '</div>',
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

        st.bar_chart(
            chart
        )

    # =====================================================
    # BUY SCORE
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
        '<div class="section-title">'
        '🛍️ Purchase Decision'
        '</div>',
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

    # =====================================================
    # FINAL RATING DECISION
    # =====================================================

    if (
        rating is not None
        and rating >= 4.0
    ):

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

    elif (
        rating is not None
        and rating >= 3.0
    ):

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
        '<div class="section-title">'
        '🔄 Similar & Better-Rated Products'
        '</div>',
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
            "No related products with "
            "verifiable ratings could be found."
        )

        st.info(
            "The shopping website may load "
            "recommendations dynamically or "
            "block automated access."
        )

    else:

        display_rows = []

        for product in similar_products:

            display_rows.append(
                {
                    "Product":
                        product[
                            "Product"
                        ],

                    "Rating":
                        f"{product['Rating']:.1f}/5",

                    "Similarity":
                        (
                            f"{product['Similarity'] * 100:.1f}%"
                        ),

                    "Recommendation Score":
                        (
                            f"{product['Recommendation Score'] * 100:.1f}%"
                        )
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

        best_rating = best[
            "Rating"
        ]

        if best_rating >= 4.0:

            decision_text = (
                "🟢 MUST BUY"
            )

            background = "#174d3b"
            border = "#35d07f"
            text_color = "#35d07f"

        elif best_rating >= 3.0:

            decision_text = (
                "🟡 CAN BUY"
            )

            background = "#4a4320"
            border = "#e6cf68"
            text_color = "#e6cf68"

        else:

            decision_text = (
                "🔴 DON'T BUY"
            )

            background = "#4a2528"
            border = "#ff7773"
            text_color = "#ff7773"

        st.markdown(
            f"""
            <div style="
                background:{background};
                border-left:6px solid {border};
                padding:18px 20px;
                border-radius:10px;
                color:{text_color};
                font-size:18px;
                font-weight:700;
                margin-top:20px;
            ">
                🏆 Best Alternative:
                {best['Product']}
                — ⭐ {best_rating:.1f}/5
                <br>
                {decision_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # RATING COMPARISON
        # =================================================

        if rating is not None:

            difference = (
                best_rating
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

        # =================================================
        # RECOMMENDATION SCORE CHART
        # =================================================

        st.subheader(
            "📊 Recommendation Scores"
        )

        chart_data = pd.DataFrame(
            {
                "Recommendation Score": [
                    p[
                        "Recommendation Score"
                    ] * 100
                    for p in similar_products
                ]
            },
            index=[
                p[
                    "Product"
                ][:45]
                for p in similar_products
            ]
        )

        st.bar_chart(
            chart_data
        )

        st.caption(
            "Recommendation Score = "
            "55% product-title similarity + "
            "45% verified product rating."
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
# HOW THE SYSTEM WORKS
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '💡 How This System Works'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    """
1. The user pastes an e-commerce product URL.

2. The system cleans the URL and removes unnecessary
   Amazon tracking parameters when possible.

3. The webpage is fetched using Requests. If Amazon
   blocks the normal request, a fallback reader is tried.

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
