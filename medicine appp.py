import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the pre-trained models using pickle
try:
    with open('disease_model.pkl', 'rb') as file:
        disease_model = pickle.load(file)

    with open('medication_model.pkl', 'rb') as file:
        medication_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Load the data
try:
    df_final2 = pd.read_csv("trained.csv", low_memory=False)  # Setting low_memory=False to avoid the warning
    df_medications = df_final2.filter(like='[')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Prepare the data
X = df_final2.filter(like='Symptom').copy()  # Make a copy of the DataFrame
y_disease = df_final2['Disease']  # Target variable for disease
y_medications = df_medications  # Target variable for medications

# Label encoding
label_encoder = LabelEncoder()
for column in X.columns:
    if X[column].dtype == 'object' or X[column].dtype == 'float64':  # Check if the column is categorical or float
        try:
            X.loc[:, column] = label_encoder.fit_transform(X[column].astype(str))  # Ensure consistent data type
        except Exception as e:
            st.error(f"Error encoding column {column}: {e}")
            st.stop()

# Streamlit app
st.title("Disease Prediction and Medication Recommendation System")

# Description
st.markdown("""
This application allows users to input their symptoms and receive predictions on possible diseases and recommended medications.
Please input the symptoms you are experiencing (comma-separated) and click the "Predict Disease" button to see the results.
""")

# User input for symptoms
user_input = st.text_input("Enter your symptoms (comma-separated):")

# Process user input
if user_input:
    input_symptoms = [symptom.strip() for symptom in user_input.split(',')]  # List of input symptoms

    # Create an input vector based on the symptoms present in the dataset
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in X.columns]
    input_df = pd.DataFrame([input_vector], columns=X.columns)

    # Predict disease
    if st.button("Predict Disease"):
        try:
            predicted_disease = disease_model.predict(input_df)
            st.subheader("Predicted Disease")
            st.write(f"The predicted disease is: **{predicted_disease[0]}**")

            # Predict medications based on the disease
            predicted_medications = medication_model.predict(input_df)

            st.subheader("Recommended Medications")
            recommended_meds = df_medications.columns[np.where(predicted_medications[0] == 1)]
            if recommended_meds.size > 0:
                for med in recommended_meds:
                    st.write(f"- {med}")
            else:
                st.write("No specific medications recommended.")

        except Exception as e:
            st.error(f"Error making predictions: {e}")

# Display the accuracy of the model
try:
    disease_accuracy = accuracy_score(y_disease, disease_model.predict(X))
    medications_accuracy = accuracy_score(y_medications, medication_model.predict(X))

    st.subheader("Model Accuracy")
    st.write(f"Disease Prediction Accuracy: **{disease_accuracy:.2f}**")
    st.write(f"Medication Prediction Accuracy: **{medications_accuracy:.2f}**")
except Exception as e:
    st.error(f"Error calculating accuracy: {e}")









 
