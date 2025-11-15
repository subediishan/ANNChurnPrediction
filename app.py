## this is end to end webapp
import streamlit as st 
import tensorflow as tf 
import pandas as pd
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler
import pickle 
import numpy as np

model = tf.keras.models.load_model('model.h5')

with open ('onehot_encoder_geo.pkl','rb') as file : 
  onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file : 
  label_encoder_gender = pickle.load(file)

with open('scalar.pkl','rb') as file : 
  scalar = pickle.load(file)


st.title(" Churn Prediction")
geography = st.selectbox("Geography",onehot_encoder_geo.categories_[0]) ##categories return list of categories
age = st.slider("Age",18,92,20)
gender = st.selectbox("Gender",["Male","Female"])
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
no_of_products = st.slider("No of products",1, 4)
has_cr_card = st.selectbox("Had credit card 1 - yes 0 - No",[0, 1 ])
is_active_member = st.selectbox("Is active Member ", [0,1])
tenure = st.slider("Tenure",0,10)
data_x = {
'CreditScore' : [credit_score], 
'Geography' : [geography],
'Gender': [label_encoder_gender.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure], 
'Balance' : [balance],
'NumOfProducts': [no_of_products], 
'HasCrCard':[has_cr_card],
'IsActiveMember': [is_active_member], 
'EstimatedSalary':[estimated_salary]
}
input_dat  = pd.DataFrame(data_x)

geo_encoded =onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography'])) 


##concat and drop Geography --> scale 

input_dat = pd.concat([input_dat.drop(["Geography"],axis = 1), geo_encoded_df], axis = 1 )
input_data_scaled = scalar.transform(input_dat)

prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]
st.write(f"churn probability : {prediction_probability}")
if prediction_probability > 0.5:
  st.write("The person is likely to churn ") 

else: 
  st.write("The person is not likely to churn ") 
print(data_x)
print(prediction_probability)
print(input_data_scaled)