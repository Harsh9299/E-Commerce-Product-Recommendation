import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="E-Commerce Product Recommendation",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Product Recommendation System")
st.write("Recommendation based on product ratings, review sentiment, price, customer behavior and product attributes.")

# ---------------- LOAD DATA ----------------

file = st.file_uploader("Upload the CSV dataset", type="csv")

if file is None:
    st.info("Upload content_based_recommendation_dataset.csv")
    st.stop()

df = pd.read_csv(file)

# Give every row a product ID
df["Product ID"] = range(1, len(df) + 1)

# ---------------- FEATURES ----------------

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

# Remove missing values
df[numeric_features] = df[numeric_features].fillna(0)
df[categorical_features] = df[categorical_features].fillna("Unknown")

# ---------------- PREPROCESSING ----------------

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

features = preprocessor.fit_transform(
    df[numeric_features + categorical_features]
)

# ---------------- SIMILARITY ----------------

similarity_matrix = cosine_similarity(features)

# ---------------- PRODUCT SELECTION ----------------

st.subheader("Select a Product")

selected_id = st.selectbox(
    "Choose Product ID",
    df["Product ID"]
)

selected_index = df.index[
    df["Product ID"] == selected_id
][0]

# ---------------- PRODUCT DETAILS ----------------

product = df.loc[selected_index]

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

# ---------------- RECOMMENDATIONS ----------------

similarities = similarity_matrix[selected_index]

recommendations = pd.DataFrame({
    "Product ID": df["Product ID"],
    "Similarity": similarities
})

# Remove selected product
recommendations = recommendations[
    recommendations["Product ID"] != selected_id
]

# Top 5
recommendations = recommendations.sort_values(
    "Similarity",
    ascending=False
).head(5)

# Add product information
recommendations = recommendations.merge(
    df,
    on="Product ID"
)

recommendations["Recommendation Score"] = (
    recommendations["Similarity"] * 100
).round(2)

# ---------------- DISPLAY ----------------

st.subheader("⭐ Recommended Products")

display = recommendations[[
    "Product ID",
    "Brand of the product",
    "Rating of the product",
    "Price of the product",
    "Customer review sentiment score (overall)",
    "Recommendation Score"
]]

display.columns = [
    "Product ID",
    "Brand",
    "Rating",
    "Price",
    "Review Sentiment",
    "Recommendation Score (%)"
]

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True
)

# ---------------- REASON ----------------

st.subheader("💡 Recommendation Explanation")

st.write(
    """
    Products are recommended using content-based similarity.
    The system compares product rating, review sentiment,
    price, brand, customer purchasing behavior, season,
    holiday and geographical information.
    """
)

# ---------------- DATASET ----------------

with st.expander("View Dataset"):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
