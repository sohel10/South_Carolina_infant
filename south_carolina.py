#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pickle
import streamlit as st
from PIL import Image

# Loading the saved model

loaded_model = pickle.load(open('C:/Users/sohel/Dropbox/Ariana/Interview_Data/Python_model/South_Carolina/trained_model.sav','rb'))
with open("gbc_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Creating a function for Prediction
def mortality_prediction(input_data):
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        st.image('high.png')
        return 'Zip Code Infant Mortality is High'
    else:
        st.image('low.png')
        return 'Zip Code Infant Mortality is low'

def main():
    # Giving a title
    st.title('Welcome ealy infant mortality Prediction Web App based on Zip Code in South Carolina')
    st.markdown("<span style='font-size: 20px;'><strong>By Sohel Ahmed</strong></span>", unsafe_allow_html=True)
    image = Image.open("south_infant.png")
    st.image(image, use_column_width=True)
    # Getting the input data from the user
    PCPs_per_1000 = st.text_input('PCPs per 1000(0-911) ')
    Percent_30_min_from_OBGYN= st.text_input('Percentage 30 Minutes from OBGYN(0-100)')
    Percent_30_min_from_Hospital = st.text_input('Percentage 30 Minutes from Hospital(0-100)')
    Percent_30_min_from_OB_Hospital_Unit = st.text_input('Percentage 30 Minutes from OB Hospital Unit(0-100)')
    Number_of_Hospitals = st.text_input('Number of Hospital(0-6)')
    Number_of_Midwives = st.text_input('Number of Midwives(0-3)')
    Percent_Labor_Force_Participation = st.text_input('Percentage Labor Force Participation(0-89)')
    Percent_Unemployed = st.text_input('Percentage Unemplyed(0-39)')
    Percent_Uninsured = st.text_input('Percentage Uninsured(0-37)')
    Percent_in_Poverty= st.text_input('Percentage in Poverty(0-100)')
    Percent_with_Kids_in_Poverty = st.text_input('Percentage With Kids in Poverty(0-100)')
    Percent_with_SNAP_Benefits = st.text_input('Percentage With SNAP Benefits(0-100)')
 
    # Code for Prediction
    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Infant Mortality'):
        diagnosis = mortality_prediction([PCPs_per_1000, Percent_30_min_from_OBGYN, Percent_30_min_from_Hospital, 
                                           Percent_30_min_from_OB_Hospital_Unit,Number_of_Hospitals, Number_of_Midwives,
                                           Percent_Labor_Force_Participation,  Percent_Unemployed,  Percent_Uninsured,
                                          Percent_in_Poverty, Percent_with_Kids_in_Poverty, Percent_with_SNAP_Benefits ])

    st.success(diagnosis)

if __name__ == '__main__':
    main()


# In[ ]:




