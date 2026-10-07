import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="E-Commerce Recommendation",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Product Recommendation System")
st.write(
    "Recommendation based on product ratings, review sentiment, "
    "price, customer behavior and product attributes."
)

# --------------------------------------------------
# DATASET
# --------------------------------------------------

file = st.file_uploader(
    "Upload the CSV dataset",
    type="csv"
)

if file is None:
    st.info("Upload the dataset to continue.")
    st.stop()

df = pd.read_csv(file)

# Product ID
df["Product ID"] = range(1, len(df) + 1)

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

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

df[numeric_features] = df[numeric_features].fillna(0)
df[categorical_features] = df[categorical_features].fillna("Unknown")

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

features = preprocessor.fit_transform(
    df[numeric_features + categorical_features]
)

# --------------------------------------------------
# SIMILARITY
# --------------------------------------------------

similarity_matrix = cosine_similarity(features)

# --------------------------------------------------
# PRODUCT SELECTION
# --------------------------------------------------

st.header("🔍 Select a Product")

selected_id = st.selectbox(
    "Choose Product ID",
    df["Product ID"]
)

selected_index = df.index[
    df["Product ID"] == selected_id
][0]

product = df.loc[selected_index]

# --------------------------------------------------
# PRODUCT DETAILS
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Brand",
    product["Brand of the product"]
)

c2.metric(
    "Rating",
    f"{product['Rating of the product']:.1f}/5"
)

c3.metric(
    "Price",
    f"₹{product['Price of the product']}"
)

c4.metric(
    "Review Sentiment",
    f"{product['Customer review sentiment score (overall)']:.2f}"
)

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

similarities = similarity_matrix[selected_index]

recommendations = pd.DataFrame({
    "Product ID": df["Product ID"],
    "Similarity": similarities
})

recommendations = recommendations[
    recommendations["Product ID"] != selected_id
]

recommendations = recommendations.sort_values(
    "Similarity",
    ascending=False
).head(5)

recommendations = recommendations.merge(
    df,
    on="Product ID"
)

# Product quality score
recommendations["Product Score"] = (
    (recommendations["Rating of the product"] / 5) * 0.5
    +
    (
        (recommendations[
            "Customer review sentiment score (overall)"
        ] + 1) / 2
    ) * 0.5
)

# Final recommendation score
recommendations["Recommendation Score"] = (
    recommendations["Similarity"] * 0.6
    +
    recommendations["Product Score"] * 0.4
)

recommendations = recommendations.sort_values(
    "Recommendation Score",
    ascending=False
)

# --------------------------------------------------
# RECOMMENDATION TABLE
# --------------------------------------------------

st.header("⭐ Recommended Products")

display = recommendations[[
    "Product ID",
    "Brand of the product",
    "Rating of the product",
    "Price of the product",
    "Customer review sentiment score (overall)",
    "Recommendation Score"
]].copy()

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

# --------------------------------------------------
# CHART
# --------------------------------------------------

st.header("📊 Recommendation Score")

chart = display.set_index("Brand")[
    "Recommendation Score"
]

st.bar_chart(chart)

# --------------------------------------------------
# PRODUCT COMPARISON
# --------------------------------------------------

st.header("📈 Product Comparison")

comparison = recommendations[[
    "Brand of the product",
    "Rating of the product",
    "Customer review sentiment score (overall)",
    "Price of the product"
]].copy()

comparison.columns = [
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

# --------------------------------------------------
# RECOMMENDATION EXPLANATION
# --------------------------------------------------

st.header("💡 Why are these products recommended?")

st.write(
    """
    The system uses content-based recommendation. Each product is
    represented using its numerical and categorical characteristics.
    Cosine similarity is then used to identify products with similar
    characteristics.

    The final recommendation score combines product similarity
    with product quality based on rating and customer review
    sentiment.
    """
)

# --------------------------------------------------
# SIMPLE EVALUATION
# --------------------------------------------------

st.header("📊 System Evaluation")

average_score = (
    recommendations["Recommendation Score"].mean() * 100
)

best_score = (
    recommendations["Recommendation Score"].max() * 100
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Average Top-5 Score",
    f"{average_score:.2f}%"
)

c2.metric(
    "Best Recommendation",
    f"{best_score:.2f}%"
)

c3.metric(
    "Products Recommended",
    "5"
)

# --------------------------------------------------
# DATASET
# --------------------------------------------------

with st.expander("View Dataset"):

    st.write(
        f"Dataset contains {len(df)} products."
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
