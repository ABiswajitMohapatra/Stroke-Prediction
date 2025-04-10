import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("stroke_prediction_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🧠 Stroke Prediction App")

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", 1, 120)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence = st.selectbox("Residence Type", ['Urban', 'Rural'])
avg_glucose_level = st.number_input("Average Glucose Level")
bmi = st.number_input("BMI")
smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Preprocess input
gender = 1 if gender == 'Male' else 0
residence = 1 if residence == 'Urban' else 0

# One-hot encoding
work_cols = ['work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children']
smoke_cols = ['smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']

work_encoded = [1 if f'work_type_{work_type}' == col else 0 for col in work_cols]
smoke_encoded = [1 if f'smoking_status_{smoking_status}' == col else 0 for col in smoke_cols]

# Scale input
scaled_values = scaler.transform([[age, bmi, avg_glucose_level]])[0]

# Final feature vector
input_data = [gender, hypertension, heart_disease, residence] + list(scaled_values) + work_encoded + smoke_encoded
input_array = np.array(input_data).reshape(1, -1)

# Predict
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_array)
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Stroke!")
    else:
        st.success("✅ Low Risk of Stroke")
