import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv


st.set_page_config(page_title="Segment Explorer", layout="wide")
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

st.title("🔍Segment Explorer")
st.header("Explore Customer Segments")

# create a list of unique personas
persona_count = df['Persona'].unique()
# widget to select persona
selected_persona = st.selectbox("Select a Persona to Explore:", persona_count)
# filter dataframe based on selected persona
filtered_df = df[df['Persona'] == selected_persona]
# display filtered dataframe
st.write(f'Found {len(filtered_df)} customers in the {selected_persona}, segment')
# display dataframe
st.dataframe(filtered_df.copy(), width='Stretch')

# download button to download filtered dataframe as csv
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(filtered_df)

st.download_button(
    label="📥Download data as CSV",
    data=csv,
    file_name=f'list_{selected_persona}_customers.csv',
    mime='text/csv',
)



