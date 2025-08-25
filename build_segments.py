import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib



def build_customers():
    """
    Build a customer-level dataset from the online retail data.
    """
    print("loading and preprocessing data...")
    try:
        df = pd.read_csv('data/raw/online_retail_II.csv', encoding='unicode_escape', dtype={'CustomerID': str})
    except FileNotFoundError:
        print("Error: File 'online_retail_II.csv' not found.")
        return
    

    # Data Cleaning
    df_clean = df.copy()
    # Convert 'InvoiceDate' to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
    # remove lines without 'Customer ID'
    df_clean = df_clean.dropna(subset=['Customer ID'])
    # remove transactions with negative or zero quantity
    df_clean = df_clean[df_clean['Quantity'] > 0]
    #transform 'Customer ID' to int
    df_clean['Customer ID'] = df_clean['Customer ID'].astype(float).astype(int)
    # calculate 'TotalPrice'
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['Price']

    # RFM Metrics
    print("calculating RFM metrics...")
    # snapshot date is one day after the last invoice date
    snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)
    # calculate Recency, Frequency, Monetary
    rfm_df = df_clean.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda date: (snapshot_date - date.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    # K-Means Clustering
    print("performing K-Means clustering...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    
    
    
    # using k=4 for segmentation
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Defining Personas and Recommended Actions
    print("defining personas and recommended actions...")
    cluster_analysis = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    # Sort clusters by Monetary value to assign personas
    cluster_analysis = cluster_analysis.sort_values(by='Monetary', ascending=False)
    
    
    cluster_map = {
    3: 'Whales',
    2: 'High-value',
    4: 'loyal',
    0: 'Regular',
    1: 'at-risk'
    }
    
    rfm_df['Persona'] = rfm_df['Cluster'].map(cluster_map)

    def get_recommendation(persona):
        recommendations = {
            'Whales': 'Exclusive offers and premium services',
            'High-value': 'Loyalty programs and personalized discounts',
            'loyal' : 'Engagement campaigns and rewards',
            'Regular' : 'Send targeted promotions to increase engagement',
            'at-risk' : 'Re-engagement campaigns and special offers'
        }
        return recommendations.get(persona, 'No recommendation available.')

    rfm_df['Recommended_Action'] = rfm_df['Persona'].apply(get_recommendation)

    # Saving Artifacts
    print("saving artifacts...")
    
    joblib.dump(scaler, 'models/rfm_scaler.pkl')
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    rfm_df.to_csv('data/processed/segmented_customers.csv', index=False)
    
    print("\n Process completed successfully.")
    print("artifacts saved in 'models/' and 'data/processed/' directories.")
    print("\nSample of segmented customers:")
    print(rfm_df[['Customer ID', 'Persona', 'Recommended_Action']].head())


if __name__ == "__main__":
    build_customers()


