import streamlit as st
import pickle
import numpy as np

# Load model
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="ML Prediction App")

st.title("Machine Learning Prediction App")

st.write("Enter the values below")

# Example inputs (change these)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):

    input_data = np.array([[feature1, feature2, feature3]])

    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")
