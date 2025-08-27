# E-commerce Customer Segmentation Platform

## link to see:
https://ecommerce-customer-segmentation-dumar.streamlit.app
## 📖 Project Description
This project develops an end-to-end Machine Learning pipeline to segment customers of an online retail store based on their purchasing behavior. The goal is to move beyond simple analytics and create actionable customer personas using Unsupervised Learning, culminating in an interactive dashboard for strategic business insights.

The methodology is centered around **RFM Analysis (Recency, Frequency, Monetary)** to engineer features, followed by **K-Means Clustering** to identify distinct customer segments. The final deliverable is an interactive **Streamlit Dashboard** that visualizes these segments and allows for real-time customer profile simulation.

## ✨ Key Features
* **Exploratory Data Analysis (EDA):** In-depth visual analysis of a large transactional dataset with over 500,000 records.
* **RFM Feature Engineering:** Calculated Recency, Frequency, and Monetary metrics to quantify customer behavior.
* **Unsupervised Learning:** Applied K-Means clustering to segment customers, using the Elbow Method to determine the optimal number of clusters.
* **Persona Creation:** Translated numerical cluster data into actionable business personas
* **Interactive Dashboard:** Built a multi-page Streamlit application to visualize the segments and simulate the persona for new customer profiles.
* **Data Persistence:** The final segmented data is stored in a **PostgreSQL** database, separating the analytical pipeline from the final application.

## 🚀 Technologies Used
- **Python 3**
- **Pandas:** For data manipulation
- **Scikit-learn:** For preprocessing (`StandardScaler`) and modeling (`KMeans`).
- **PostgreSQL & SQLAlchemy:** For database storage and connection.
- **Streamlit:** For building the interactive web application/dashboard.
- **Plotly:** For creating interactive data visualizations.
- **Joblib:** For serializing model and scaler artifacts.
- **Git & GitHub:** For version control and project management.

## 📂 Project Structure
The repository is organized to separate the analytical notebook, production scripts, and the final dashboard application.

ecommerce-segmentation-project/
├── build_segments.py                 # Script to run the full pipeline (RFM + Clustering) and populate the database
├── dashboard.py                      # The main Streamlit dashboard application file
├── pages/
│   └── 1_Client_Analysis.py          # Page for simulating new customer profiles
│   └── 2_Segment_Explorer.py         # Page for exploring customers within a segment
├── models/
│   ├── kmeans_model.pkl              # Saved final K-Means model artifact
│   └── rfm_scaler.pkl                # Saved scaler artifact
├── notebooks/
│   └── exploratory_analysis.ipynb    # Notebook with the full research and analysis
├── data/
│   └── raw/
│       └── online_retail_II.csv      # The original raw dataset
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation




📊 Analysis Conclusion
The K-Means algorithm successfully identified 5 distinct customer personas from the RFM data, ranging from high-value "Whales" (likely B2B accounts) and "Champions" to a large group of "At-Risk" customers who have not purchased in a long time.

This segmentation provides a clear framework for the business to develop targeted marketing strategies, such as loyalty programs for top-tier customers and re-engagement campaigns for those at risk, thereby optimizing marketing spend and improving customer retention.

👨‍💻 Author
Rafael Dumar Batista

LinkedIn: https://www.linkedin.com/in/rafaeldumar/

Email: rafaeldumar15@gmail.com
