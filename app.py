import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("C:\\Users\\Hp\\Downloads\\breast_cancer_model.pkl")
scaler = joblib.load("C:\\Users\\Hp\\Downloads\\scaler.pkl")

def predict_cancer(input_data):
    expected_features = 30  # Updated to match numerical features
    if len(input_data) != expected_features:
        raise ValueError(f"Expected {expected_features} features, but got {len(input_data)}.")

    input_data = np.array(input_data).reshape(1, -1)  # Convert input to 2D array
    input_data_scaled = scaler.transform(input_data)  # Apply the same scaling
    prediction = model.predict(input_data_scaled)

    return prediction[0]  # Return prediction result


# Streamlit UI setup
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("🔬 Breast Cancer Prediction System")
st.write("An AI-powered tool to detect breast cancer based on medical data.")

menu = st.sidebar.radio("Navigation", ["Home", "About Us", "Contact"])

if menu == "Home":
    st.header("Enter Patient Data for Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, format="%.4f")
        texture_mean = st.number_input("Texture Mean", min_value=0.0, format="%.4f")
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, format="%.4f")
        area_mean = st.number_input("Area Mean", min_value=0.0, format="%.4f")
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, format="%.4f")
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, format="%.4f")
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, format="%.4f")
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, format="%.4f")
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, format="%.4f")
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, format="%.4f")
    
    with col2:
        radius_se = st.number_input("Radius SE", min_value=0.0, format="%.4f")
        texture_se = st.number_input("Texture SE", min_value=0.0, format="%.4f")
        perimeter_se = st.number_input("Perimeter SE", min_value=0.0, format="%.4f")
        area_se = st.number_input("Area SE", min_value=0.0, format="%.4f")
        smoothness_se = st.number_input("Smoothness SE", min_value=0.0, format="%.4f")
        compactness_se = st.number_input("Compactness SE", min_value=0.0, format="%.4f")
        concavity_se = st.number_input("Concavity SE", min_value=0.0, format="%.4f")
        concave_points_se = st.number_input("Concave Points SE", min_value=0.0, format="%.4f")
        symmetry_se = st.number_input("Symmetry SE", min_value=0.0, format="%.4f")
        fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, format="%.4f")
    
    with col3:
        radius_worst = st.number_input("Radius Worst", min_value=0.0, format="%.4f")
        texture_worst = st.number_input("Texture Worst", min_value=0.0, format="%.4f")
        perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, format="%.4f")
        area_worst = st.number_input("Area Worst", min_value=0.0, format="%.4f")
        smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, format="%.4f")
        compactness_worst = st.number_input("Compactness Worst", min_value=0.0, format="%.4f")
        concavity_worst = st.number_input("Concavity Worst", min_value=0.0, format="%.4f")
        concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, format="%.4f")
        symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, format="%.4f")
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, format="%.4f")
    
    input_data = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]
    
    if st.button("🔍 Predict Cancer"):
        prediction = predict_cancer(input_data)
        if prediction == 1:
            st.error("🔴 The model predicts this case as **Malignant (Cancerous)**. Immediate medical attention is advised.")
        else:
            st.success("🟢 The model predicts this case as **Benign (Non-Cancerous)**. No immediate concern detected.")
    
elif menu == "About Us":
    st.header("About Us")
    st.write("We are a team of AI researchers dedicated to leveraging technology for early detection of life-threatening diseases.")
    st.image("https://via.placeholder.com/800x400", caption="Our mission: Saving lives through AI-powered diagnostics.")
    
elif menu == "Contact":
    st.header("Contact Us")
    st.write("📩 Email: support@cancer-detection.ai")
    st.write("📍 Address: AI Health Tech, Silicon Valley, CA")
