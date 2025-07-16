# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(
    page_title="🎧 Spotify Churn Prediction System",
    layout="wide",
    page_icon="🎯"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_model():
    return pickle.load(open("xgboost_model.pkl", "rb"))

@st.cache_resource
def load_encoders():
    return pickle.load(open("label_encoders.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("spotify_churn_dataset.csv")

model = load_model()
encoders = load_encoders()
df = load_data()

# --- SIDEBAR ---
st.sidebar.image("https://avatars.githubusercontent.com/u/139329865?v=4", width=100)
st.sidebar.title("Claude Tomoh")
st.sidebar.markdown("🎓 Future Interns – ML Track")
st.sidebar.markdown("[📂 GitHub](https://github.com/20215o) | [🌐 LinkedIn](https://linkedin.com)")
st.sidebar.markdown("---")
st.sidebar.markdown("Use the options below to explore:")

# --- HEADER ---
st.title("🎧 Spotify Churn Prediction System")
st.markdown("##### Built by Claude Tomoh – Future Interns ML Internship (Task 2)")
st.markdown("An intelligent app to predict user churn, segment risk levels, and drive business retention strategies.")

# --- DATA PREVIEW ---
with st.expander("📊 Preview Dataset"):
    st.dataframe(df.head())

# --- ENCODING ---
input_df = df.copy()
for col, encoder in encoders.items():
    input_df[col] = encoder.transform(input_df[col])

X = input_df.drop(columns=['user_id', 'churned'])
df["churn_probability"] = model.predict_proba(X)[:, 1]
df["predicted_churn"] = model.predict(X)

# --- SEGMENT RISK ---
def risk_group(p): return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"
df["churn_risk"] = df["churn_probability"].apply(risk_group)

# --- METRICS ROW ---
st.markdown("### 📈 Summary Insights")
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", f"{len(df):,}")
col2.metric("Churn Rate", f"{df['churned'].mean()*100:.1f} %")
col3.metric("High Risk Users", df[df['churn_risk'] == "High"].shape[0])

# --- PIE CHART ---
with st.container():
    st.markdown("### 🎯 Churn Distribution")
    fig1 = px.pie(df, names='churned', color='churned',
                  color_discrete_map={0: "lightgreen", 1: "red"},
                  hole=0.4,
                  labels={0: "Active", 1: "Churned"})
    st.plotly_chart(fig1, use_container_width=True)

# --- FEATURE IMPORTANCE ---
st.markdown("### 📌 What Drives Churn?")
importance = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importance)
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(features[sorted_idx], importance[sorted_idx], color="skyblue")
ax2.set_title("XGBoost Feature Importance")
st.pyplot(fig2)

# --- ADVANCED VISUALIZATION ---
with st.container():
    st.markdown("### 📊 Listening Time vs Churn")
    fig3 = px.violin(df, y='avg_daily_minutes', x='churned', color='churned',
                     box=True, points='all', labels={'churned': 'Churned'})
    st.plotly_chart(fig3, use_container_width=True)

with st.container():
    st.markdown("### 🔍 Support Tickets by Churn Risk")
    fig4 = px.box(df, x="churn_risk", y="support_tickets", color="churn_risk")
    st.plotly_chart(fig4, use_container_width=True)

# --- LOOKUP USER ---
st.markdown("### 👤 User Risk Checker")
user_id = st.selectbox("Select a User ID", df["user_id"].unique())
user = df[df["user_id"] == user_id]
st.success(f"Churn Probability: {user['churn_probability'].values[0]:.2%}")
st.warning(f"Risk Segment: {user['churn_risk'].values[0]}")
st.dataframe(user.drop(columns=['churn_probability', 'predicted_churn', 'churn_risk']))

# --- INSIGHT HIGHLIGHTS ---
with st.expander("📌 Claude's AI Insights"):
    st.markdown("""
    - 🎵 Users with lower listening time churn more often.
    - 🧾 Support tickets increase the likelihood of churn.
    - 🏷️ Free and lower-tier subscription users are at higher risk.
    - 🌍 Certain countries have higher churn rates – consider geo-targeted campaigns.
    - 🛑 Many high-risk users haven't logged in recently — reactivation strategies could help.
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("✅ _Built by Claude Tomoh as part of Future Interns ML Internship (Task 2)._")
