import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf

## loading the trained model
model = tf.keras.models.load_model('regression.h5')

## loading the encoders
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

##loading the scaler
with open('sscaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Estimated Salary Prediction using Regression')

## user input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', ['No','Yes'])
is_active_member = st.selectbox('Is Active Member', ['No','Yes'])
exited = st.selectbox('Exited', ['No', 'Yes'])

## prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    #'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [0 if has_cr_card=='No' else 1],
    'IsActiveMember': [0 if is_active_member=='No' else 1],
    'Exited': [0 if exited=='No' else 1]
})

#onehot encoding geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
#combining encoded geography with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
#scaling the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)[0][0]
st.write(f'Predicted Salary: {prediction}')