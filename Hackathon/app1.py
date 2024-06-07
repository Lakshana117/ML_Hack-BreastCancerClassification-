import numpy as np
import pickle
import streamlit as st

# loading the saved model
with open('C:\\Users\\subas\\Documents\\Hackathon\\LinearRegression.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# creating a function for Prediction
def breast_cancer_detector(input_data):
    # Convert input data to float
    input_data = [float(value) for value in input_data]

    # changing the input_data to numpy array
    
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The breast cancer is Malignant'
    else:
        return 'The breast cancer is Benign'


def main():
    # giving a title
    st.title('Breast Cancer Prediction Web App')

    # getting the input data from the user
    radius_mean = st.text_input('Radius')
    
    texture_mean = st.text_input('texture_mean')
    area_mean = st.text_input('area_mean')
    perimeter_mean = st.text_input('perimeter_mean')
    smoothness_mean = st.text_input('smoothness_mean')
    compactness_mean = st.text_input('compactness_mean')
    concavity_mean = st.text_input('concavity_mean')
    concavepoints_mean = st.text_input('concavepoints_mean')
    symmetry_mean = st.text_input('symmetry mean')
    fractal_dimension_mean = st.text_input('fractal_dimension_mean')
    radius_se = st.text_input('radius se')
    texture_se = st.text_input('texture se')
    area_se = st.text_input('area se')
    perimeter_se = st.text_input('perimeter_se')
    smoothness_se = st.text_input('smoothness_se')
    compactness_se = st.text_input('compactness_se')
    concavity_se = st.text_input('concavity_se')
    concavepoints_se = st.text_input('concavepoints se')
    symmetry_se = st.text_input('Symmetry se')
    fractal_dimension_se = st.text_input('fractal dimension se')
    radius_worst = st.text_input('radius worst')
    texture_worst = st.text_input('texture worst')
    area_worst = st.text_input('area worst')
    perimeter_worst = st.text_input('perimeter worst')
    smoothness_worst = st.text_input('smoothness worst')
    compactness_worst = st.text_input('compactness worst')
    concavity_worst = st.text_input('concavity worst')
    concavepoints_worst = st.text_input('concavepoints worst')
    symmetry_worst = st.text_input('symmetry worst')
    fractal_dimension_worst = st.text_input('fractal dimension worst')
    
    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Breast Cancer Test Result'):
        input_data = [radius_mean, texture_mean, area_mean, perimeter_mean, smoothness_mean, compactness_mean,concavity_mean, concavepoints_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, area_se, perimeter_se, smoothness_se, compactness_se,concavity_se, concavepoints_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, area_worst, perimeter_worst, smoothness_worst, compactness_worst,concavity_worst, concavepoints_worst, symmetry_worst, fractal_dimension_worst]
        diagnosis = breast_cancer_detector(input_data)

    st.success(diagnosis)

if __name__ == '__main__':
    main()