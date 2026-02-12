import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🎓 Student Performance Prediction")
st.write("Enter student details below:")

# Inputs
study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=50)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)
previous_grade = st.number_input("Previous Grade (%)", min_value=0, max_value=100)

# Prediction
if st.button("Predict Performance"):
    # Prepare input for prediction (3 features)
    input_data = np.array([study_hours, attendance, previous_grade]).reshape(1, -1)
    
    prediction = model.predict(input_data)[0]

    # If regression model predicting 0/1
    if prediction >= 0.5:
        st.success("🎉 Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")

