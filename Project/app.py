
import streamlit as st
import pandas as pd
import joblib

# ------------------------
# Load model
# ------------------------
model = joblib.load("model.pkl")  # pipeline includes encoder

# ------------------------
# Page setup
# ------------------------
st.set_page_config(page_title="Hotel Price Predictor", page_icon="🏨", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align:center;color:#2F4F4F;'>🏨 Hotel Price Predictor 💰</h1>",
    unsafe_allow_html=True
)
st.write("---")

# ------------------------
# Input fields
# ------------------------
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("Select City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"])
    rating = st.slider("Hotel Rating", 1.0, 5.0, 4.0)
    rooms = st.number_input("Number of Rooms", min_value=1, value=50)

with col2:
    distance_from_center_km = st.number_input("Distance from City Center (km)", min_value=0.0, value=5.0)
    amenities_count = st.number_input("Number of Amenities", min_value=1, value=10)

# ------------------------
# Predict button
# ------------------------
if st.button("💵 Predict Price"):
    try:
        # Create DataFrame with **exact column names**
        sample_df = pd.DataFrame([{
            "city": city,  # keep as string
            "rating": float(rating),
            "rooms": int(rooms),
            "distance_from_center_km": float(distance_from_center_km),
            "amenities_count": int(amenities_count)
        }])
        
        # Predict
        predicted_price = model.predict(sample_df)[0]

        st.markdown(
            f"<div style='padding:20px;background-color:#f0f8ff;border-radius:10px;text-align:center;'>"
            f"<h2 style='color:#008080;'>Predicted Hotel Price</h2>"
            f"<p style='font-size:28px;font-weight:bold;color:#2F4F4F;'>₹ {predicted_price:.2f}</p></div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error during prediction: {e}")
