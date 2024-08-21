# Medicine-Recommendation-system
A medicine recommendation system based on the diseases and symptoms the patient may have.
<img src = "https://github.com/William-Itotia/Medicine-Recommendation-system/blob/main/static/StockCake-Digital%20Health%20Concept_1723969232.jpg?raw=true">

## Overview
The healthcare industry is undergoing significant transformation with the integration of technology, particularly in areas like accessibility, efficiency, and personalized care. This project focuses on creating a recommendation system that can predict diseases based on user-input symptoms, provide detailed descriptions of these diseases, suggest preventive measures, and recommend appropriate medications. The aim is to offer timely and accurate medical advice, thereby reducing the need for immediate hospital visits and improving overall health outcomes.

## Problem Statement
Accessing timely and accurate medical advice remains a challenge for many due to geographical barriers, time constraints, and overcrowded healthcare facilities. This often results in delayed diagnoses and treatment, worsening health conditions. There is a clear need for a solution that offers immediate, reliable, and personalized medical recommendations based on symptoms, improving healthcare accessibility and easing the burden on medical facilities.

## Objectives
* To gather a repository of detailed descriptions for a wide range of diseases, including causes, symptoms, and treatment options.
* To develop and train a machine learning model on extensive medical data to predict possible diseases based on the input symptoms.
* To provide suggestions for precautions and preventive measures tailored to the predicted diseases.
* To recommend appropriate medications based on the predicted disease.
* To integrate a user-friendly interface for individuals to input their symptoms.

## Data Understanding 
Five datasets were used: 
* symptoms_df: Contains diseases and associated symptoms.
* precautions_df: Lists diseases with their corresponding precautions.
* descriptions_df: Provides detailed descriptions of diseases.
* medications_df: Contains recommended medications for each disease.
* training_df: Used for training the machine learning model, linking symptoms to diseases.

Missing values are present in the symptoms dataset, which are not dropped due to varying symptom occurrences in different diseases. Missing precautions are filled with 'None'.

## Data Preparation
One-Hot Encoding
The symptom data is transformed using one-hot encoding to convert categorical variables into numerical format, which is essential for machine learning algorithms.
Train-Test Split
The dataset is split into training and testing sets, with symptoms as the predictor variables and diseases as the target variables.

## Modeling 
Several machine learning models are tested, including:

* Support Vector Classifier (SVC)
* Random Forest Classifier
* K-Neighbors Classifier
* Multinomial Naive Bayes
* Logistic Regression

The SVC model was used in this project.

## Deployment
Steamlit was used to deploy this application.You can access it here:https://medicine-recommendation-system-42luxseofvhqy7ad97zvsk.streamlit.app/

## Conclusion
The recommendation system effectively predicts diseases based on symptoms and provides comprehensive information including precautions and medications. The next steps involve refining the user interface, integrating the system into healthcare platforms, and continually updating the model with new medical data.

## Recomendations
It is vital to remember that this system does not replace a doctor.It is merely a guide to help understand one’s symptoms and take appropriate actions.It is designed to be used by individuals who cannot access a medical facility despite being in urgent need of medical care.Seeking advice from medical professionals is highly encouraged.

