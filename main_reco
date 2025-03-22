import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Expanded Credit Card Dataset
credit_cards = pd.DataFrame({
    "Card Name": [
        "SBI Cashback Credit Card", "HDFC Millennia Credit Card", "Amex Platinum Travel Credit Card",
        "Axis Bank Vistara Credit Card", "ICICI Amazon Pay Credit Card", "HDFC Infinia Credit Card",
        "Citi PremierMiles Credit Card", "Standard Chartered Ultimate Card", "RBL Bank Shoprite Credit Card",
        "Kotak PVR Gold Credit Card", "IDFC FIRST Select Credit Card", "AU Small Finance LIT Card",
        "Bank of Baroda Easy Credit Card", "IndianOil Citi Credit Card", "BPCL SBI Card Octane",
        "HDFC Bharat Cashback Card", "HDFC Regalia Credit Card", "SBI Card ELITE",
        "Axis Bank Ace Credit Card", "Flipkart Axis Bank Credit Card", "YES FIRST Preferred Credit Card",
        "ICICI Bank Coral Credit Card", "HDFC Diners Club Black Credit Card", "SBI SimplyCLICK Credit Card",
        "ICICI Bank Platinum Chip Credit Card", "Union Bank of India VISA Signature Credit Card",
        "Union Bank of India VISA Platinum Credit Card", "Union Bank of India VISA Gold Credit Card"
    ],
    "Features": [
        "cashback rewards online shopping dining",
        "cashback on online shopping dining travel benefits",
        "premium travel benefits lounge access international rewards",
        "frequent flyer miles travel benefits lounge access",
        "cashback on Amazon purchases online shopping benefits",
        "luxury lifestyle benefits unlimited lounge access premium rewards",
        "air miles travel benefits lounge access",
        "luxury lifestyle benefits high rewards on shopping and dining",
        "rewards on grocery and retail shopping shopping benefits",
        "benefits on movie ticket purchases entertainment perks",
        "no annual fee multiple benefits lifestyle rewards",
        "customizable benefits zero annual charges flexible rewards",
        "rewards without annual fees low-interest rates",
        "benefits on fuel purchases at IndianOil outlets fuel rewards",
        "accelerated rewards on fuel spends fuel surcharge waiver",
        "cashback on fuel and other categories fuel benefits",
        "high reward points on various spends premium benefits",
        "accelerated rewards premium lifestyle benefits",
        "cashback on bill payments online food delivery ride-hailing services",
        "cashback on Flipkart purchases online shopping benefits",
        "reward points on all purchases lifestyle benefits",
        "rewards on shopping and dining lifestyle privileges",
        "premium travel benefits unlimited lounge access high reward points",
        "rewards on online shopping e-commerce benefits",
        "low-interest rates basic rewards EMV chip security",
        "premium lifestyle benefits global acceptance higher credit limits",
        "lifestyle benefits global acceptance moderate credit limits",
        "basic benefits global acceptance suitable for beginners"
    ]
})

# Streamlit Title
st.title("💳 AI-Powered Credit Card Recommender")

# Sidebar for User Inputs
st.sidebar.header("🔍 Your Preferences")
spending_category = st.sidebar.selectbox(
    "Select Your Primary Spending Category",
    ["Cashback", "Travel", "Fuel", "Shopping", "Lifestyle", "Entertainment", "Low Interest", "Premium/Luxury"]
)
additional_preferences = st.sidebar.text_area(
    "Enter additional preferences (e.g., dining, lounge access, online shopping)", ""
)

# Combine User Input as Query
user_query = spending_category.lower() + " " + additional_preferences.lower()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(credit_cards["Features"])
query_vector = vectorizer.transform([user_query])

# Compute Similarity Scores
similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Get Top 3 Recommendations
top_indices = similarity_scores.argsort()[-3:][::-1]
top_recommendations = credit_cards.iloc[top_indices][["Card Name"]]

# Display Top Recommendations
st.subheader("🎯 Recommended Credit Cards")
for idx, row in top_recommendations.iterrows():
    st.success(f"🔹 {row['Card Name']} (Match Score: {round(similarity_scores[idx] * 100, 2)}%)")

# Show All Scores (Optional)
if st.checkbox("Show All Scores"):
    credit_cards["Similarity Score"] = similarity_scores
    st.dataframe(credit_cards.sort_values("Similarity Score", ascending=False))

# Additional Tip
st.markdown("💡 **Tip:** Choose a credit card that aligns with your spending habits to maximize benefits!")

# Disclaimer
st.markdown("🔍 **Disclaimer:** The recommendations are based on the provided data and user inputs. "
            "Please verify the latest card features and terms on the official bank websites before applying.")
