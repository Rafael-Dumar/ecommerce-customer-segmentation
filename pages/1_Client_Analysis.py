import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import joblib

st.set_page_config(page_title="Client Analysis", layout="wide")
@st.cache_data
def load_data():
    """
    Load customer data from the database.
    """
    load_dotenv()
    db_url = os.getenv("DB_URL")
    try:
        scaler = joblib.load("models/rfm_scaler.pkl")
        kmeans = joblib.load("models/kmeans_model.pkl")
        engine = create_engine(db_url)
        df = pd.read_sql("SELECT * FROM customer_segments", engine)
        return df, scaler, kmeans
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None
    
df, scaler, kmeans = load_data()

cluster_analysis = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2).sort_values(by='Monetary', ascending=False).reset_index()
cluster_map = {
    3: 'Whales',
    2: 'High-value',
    4: 'loyal',
    0: 'Regular',
    1: 'at-risk'
    }

df['Persona'] = df['Cluster'].map(cluster_map)

def get_recommendation(persona):
        recommendations = {
            'Whales': 'Exclusive offers and premium services',
            'High-value': 'Loyalty programs and personalized discounts',
            'loyal' : 'Engagement campaigns and rewards',
            'Regular' : 'Send targeted promotions to increase engagement',
            'at-risk' : 'Re-engagement campaigns and special offers'
        }
        return recommendations.get(persona, 'No recommendation available.')

st.title("👤Client Analysis")
st.header("Customer Segments Overview")

col1, col2, col3 = st.columns(3)
with col1:
     recency_input = st.number_input("Enter Recency (days since last purchase):", min_value=0, value=5)
with col2:
     frequency_input = st.number_input("Enter Frequency (number of purchases):", min_value=0, value=10)
with col3:
     monetary_input = st.number_input("Enter Monetary value (total spent):", min_value=0.0, value=500.0)

if st.button("Analyze Client"):
    input_data = pd.DataFrame({
        'Recency': [recency_input],
        'Frequency': [frequency_input],
        'Monetary': [monetary_input]
    })
    input_scaled = scaler.transform(input_data)
    # Predict cluster
    cluster_prediction = kmeans.predict(input_scaled)[0]
    # Map to persona
    persona_prediction = cluster_map.get(cluster_prediction, "Unknown")
    # Get recommendation
    recommendation = get_recommendation(persona_prediction)

    # Display results
    if persona_prediction in ["Whales", "High-value"]:
        st.success(f"The client is classified as:  {persona_prediction}")
    elif persona_prediction in ["loyal", "Regular"]:
        st.info(f"The client is classified as:  {persona_prediction}")
    else:
        st.error(f"The client is classified as:  {persona_prediction}")

    st.markdown(f"Recommended Action: {recommendation}")

    st.markdown("### Segment Characteristics:")
    segment_info = cluster_analysis[cluster_analysis['Cluster'] == cluster_prediction]
    st.dataframe(segment_info, width='stretch')
    
