# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:08:59 2024

@author: divya
"""

import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load (open(r"C:\Users\divya\OneDrive\Desktop\MultipleDisease\trained_model.sav","rb"))

#creating a function for prediction

def diabetes_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    print(prediction)
    
    if (prediction[0] == 0):
         return 'The person is diabetic'
    else:
         return 'The person is not diabetic'
         
    return 'The person is diabetic'  

        
def main():
    
    # giving title 
    st.title("Diabetes Prediction Web App")
    
    
    # getting input data from user
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("BloodPressure Value")
    SkinThickness = st.text_input("SkinThickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
    Age = st.text_input("Age of Person")
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagnosis)
    
    
    
    
if __name__== '__main__':
    main()
    
    
    
