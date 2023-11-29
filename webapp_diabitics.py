#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    # Title and description
    st.title("Diabetes Prediction App")
    st.write(
        "This app predicts whether a person has diabetes or not based on input features."
    )

    # Input features
    st.sidebar.header("User Input Features")
    st.sidebar.markdown(
        
    )

    # Collect user input features
    pregnancy = st.sidebar.slider("Number of Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Plasma Glucose Concentration", 0, 199, 117)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 72)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 23)
    insulin = st.sidebar.slider("Insulin Level", 0, 846, 30)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    age = st.sidebar.slider("Age", 21, 81, 29)

    # Features as a DataFrame
    user_input = {
        "Pregnancies": pregnancy,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age,
    }

    features = pd.DataFrame([user_input])

    # Show user input features
    st.subheader("User Input Features")
    st.write(features)

    # Make predictions
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    # Display prediction
    st.subheader("Prediction")
    outcome_mapping = {0: "No Diabetes", 1: "Diabetes"}
    st.write(outcome_mapping[prediction[0]])

    # Display prediction probabilities
    st.subheader("Prediction Probability")
    st.write(f"Probability of No Diabetes: {prediction_proba[0, 0]:.2f}")
    st.write(f"Probability of Diabetes: {prediction_proba[0, 1]:.2f}")


# In[ ]:




