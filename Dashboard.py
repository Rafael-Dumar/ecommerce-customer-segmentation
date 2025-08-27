import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    """
    Load customer data from the database.
    """
    load_dotenv()
    db_url = os.getenv("DB_URL")
    try:
        engine = create_engine(db_url)
        df = pd.read_sql("SELECT * FROM customer_segments", engine)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()


st.title("📊Customer Segmentation Dashboard")
st.header("Customer Segments Overview")

# bar chart
st.subheader("Client Distribution by Segment")
persona_counts = df['Persona'].value_counts().sort_values(ascending=False)
fig_bar = px.bar(persona_counts, x=persona_counts.index, y=persona_counts.values, 
                labels={'x':'Persona', 'y':'Number of Clients'}, text=persona_counts.values)
st.plotly_chart(fig_bar, width='Stretch')
    
# table
st.subheader("Average RFM Metrics by Segment")
st.markdown("The table below shows the average Recency, Frequency, and Monetary values for each customer segment.")
cluster_analysis = df.groupby('Persona')[['Recency', 'Frequency', 'Monetary']].mean().round(2).sort_values(by='Monetary', ascending=False).reset_index()
st.dataframe(cluster_analysis, width='Stretch')
    
#bubble chart
st.header("RFM Metrics Distribution")
st.markdown("Below, we compared the profile of each segment based on the average RFM metrics.")


# Create three columns for the three metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Recency Mean")
    # bar chart for Recency, ordered from best (lowest) to worst (highest)
    fig_recency = px.bar(
        cluster_analysis.sort_values('Recency'), 
        x='Persona', 
        y='Recency',
        labels={'Persona':'', 'Recency':'Recency Mean (Days)'},
        text_auto=True
    ).update_traces(textposition='outside')
    st.plotly_chart(fig_recency, width='stretch')

with col2:
    st.subheader("Frequency Mean")
    # bar chart for Frequency, ordered from best (highest) to worst (lowest)
    fig_freq = px.bar(
        cluster_analysis.sort_values('Frequency', ascending=False),
        x='Persona',
        y='Frequency',
        text_auto=True,
        labels={'Persona':'', 'Frequency':'Nº of Purchases'}
    ).update_traces(textposition='outside')
    st.plotly_chart(fig_freq, width='stretch')

with col3:
    st.subheader("Monetary Mean")
    # bar chart for Monetary, ordered from best (highest) to worst (lowest)
    fig_monetary = px.bar(
        cluster_analysis.sort_values('Monetary', ascending=False),
        x='Persona',
        y='Monetary',
        text_auto='.2s', 
        labels={'Persona':'', 'Monetary':'Average Spend ($)'}
    ).update_traces(textposition='outside')
    st.plotly_chart(fig_monetary, width='stretch')

st.subheader("3D Visualization of Customer Segments")
st.markdown("The 3D scatter plot below visualizes customer segments based on their RFM metrics. Each point represents a customer, colored by their assigned persona.")
fig_3d = px.scatter_3d(
    df,
    x='Recency',
    y='Frequency',
    z='Monetary',
    color='Persona',
    opacity=0.5,
    hover_data=['Customer ID'] 
)

fig_3d.update_layout(
    scene=dict(
        xaxis_title='Recency',
        yaxis_title='Frequency',
        zaxis_title='Monetary value'
    )
)
st.plotly_chart(fig_3d, width='stretch')


