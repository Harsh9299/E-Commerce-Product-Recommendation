import streamlit as st
import pandas as pd
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
    "A content-based recommendation system using product ratings, "
    "customer review sentiment, price, brand, and customer behavior."
)

# =========================================================
# DATASET UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload the CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload the e-commerce CSV dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# =========================================================
# PRODUCT ID
# =========================================================

df = df.reset_index(drop=True)
df["Product ID"] = df.index + 1

# =========================================================
# REQUIRED FEATURES
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

required_columns = numeric_features + categorical_features

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("The following columns are missing from the dataset:")
    st.write(missing_columns)
    st.stop()

# =========================================================
# DATA PREPROCESSING
# =========================================================

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

# =========================================================
# FEATURE ENCODING
# =========================================================

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

feature_matrix = preprocessor.fit_transform(
    df[required_columns]
)

# =========================================================
# COSINE SIMILARITY
# =========================================================

similarity_matrix = cosine_similarity(
    feature_matrix
)

# =========================================================
# PRODUCT SELECTION
# =========================================================

st.header("🔍 Select a Product")

selected_id = st.selectbox(
    "Choose Product ID",
    df["Product ID"].tolist()
)

selected_index = df.index[
    df["Product ID"] == selected_id
][0]

selected_product = df.loc[selected_index]

# =========================================================
# SELECTED PRODUCT DETAILS
# =========================================================

st.subheader("Selected Product")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Brand",
        selected_product["Brand of the product"]
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

# =========================================================
# CALCULATE RECOMMENDATION SCORES
# =========================================================

similarities = similarity_matrix[selected_index]

recommendations = pd.DataFrame({
    "Product ID": df["Product ID"],
    "Similarity": similarities
})

# Remove the selected product
recommendations = recommendations[
    recommendations["Product ID"] != selected_id
].copy()

# Add product information
recommendations = recommendations.merge(
    df,
    on="Product ID",
    how="left"
)

# ---------------------------------------------------------
# PRODUCT QUALITY SCORE
# ---------------------------------------------------------

# Rating converted to 0-1
rating_score = (
    recommendations["Rating of the product"] / 5
)

# Sentiment is between -1 and +1
sentiment_score = (
    recommendations[
        "Customer review sentiment score (overall)"
    ] + 1
) / 2

# Combined product quality
recommendations["Product Quality Score"] = (
    0.5 * rating_score +
    0.5 * sentiment_score
)

# ---------------------------------------------------------
# FINAL RECOMMENDATION SCORE
# ---------------------------------------------------------

recommendations["Recommendation Score"] = (
    0.60 * recommendations["Similarity"] +
    0.40 * recommendations["Product Quality Score"]
)

# Sort by final recommendation score
recommendations = recommendations.sort_values(
    "Recommendation Score",
    ascending=False
)

# Select top 5
top_recommendations = recommendations.head(5).copy()

# =========================================================
# RECOMMENDED PRODUCTS
# =========================================================

st.header("⭐ Top 5 Recommended Products")

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

display["Recommendation Score"] = (
    display["Recommendation Score"] * 100
).round(2)

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# RECOMMENDATION SCORE CHART
# =========================================================

st.header("📊 Recommendation Score")

chart_data = display.copy()

# Unique label prevents duplicate brand names
chart_data["Product"] = (
    chart_data["Brand"]
    + " - ID "
    + chart_data["Product ID"].astype(str)
)

chart_data = chart_data.set_index(
    "Product"
)["Recommendation Score"]

st.bar_chart(chart_data)

# =========================================================
# PRODUCT COMPARISON
# =========================================================

st.header("📈 Product Comparison")

comparison = top_recommendations[
    [
        "Product ID",
        "Brand of the product",
        "Rating of the product",
        "Customer review sentiment score (overall)",
        "Price of the product"
    ]
].copy()

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

# =========================================================
# RECOMMENDATION EXPLANATION
# =========================================================

st.header("💡 Why Are These Products Recommended?")

st.write(
    """
    The system uses a content-based recommendation approach.
    Each product is represented using numerical and categorical
    characteristics such as rating, price, review sentiment,
    brand, customer behavior, season, holiday and location.

    Cosine similarity is used to identify products with similar
    characteristics. The final recommendation score combines
    product similarity with product quality based on rating and
    customer review sentiment.
    """
)

# =========================================================
# TOP RECOMMENDATION
# =========================================================

best = top_recommendations.iloc[0]

st.success(
    f"🏆 Best Recommendation: "
    f"{best['Brand of the product']} "
    f"(Product ID {int(best['Product ID'])}) "
    f"with a recommendation score of "
    f"{best['Recommendation Score'] * 100:.2f}%"
)

# =========================================================
# SYSTEM EVALUATION
# =========================================================

st.header("📊 System Evaluation")

average_score = (
    top_recommendations["Recommendation Score"]
    .mean() * 100
)

average_similarity = (
    top_recommendations["Similarity"]
    .mean() * 100
)

best_score = (
    top_recommendations["Recommendation Score"]
    .max() * 100
)

col1, col2, col3 = st.columns(3)

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

# =========================================================
# DATASET INFORMATION
# =========================================================

st.header("📁 Dataset Information")

col1, col2 = st.columns(2)

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

with st.expander("View Complete Dataset"):
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
    "Content-Based Recommendation"
)
