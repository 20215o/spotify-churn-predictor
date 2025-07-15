
import streamlit as st

# --- PAGE SETUP ---
st.set_page_config(page_title="Churn Prediction System", layout="centered")

# --- MAIN TITLE ---
st.title("🎧 Spotify Churn Prediction System")
st.subheader("By Claude Tomoh – Future Interns Machine Learning Task 2")

# --- INTRODUCTION ---
st.markdown("""
Welcome to the Churn Prediction System for a simulated Spotify-like platform.

This app helps you:
- 🔍 Understand which users are likely to stop using the service.
- 📉 Analyze key risk factors driving churn.
- 📊 Segment users into High, Medium, or Low Risk.
- 🚀 Take proactive business decisions to retain customers.

---
""")
