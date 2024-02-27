import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib as plotly

data = pd.read_csv('Loan_prediction_dataset for Numpy checkpoint.csv')

df = data.copy()
df.drop('Loan_ID', axis = 1, inplace = True)

df = df[['ApplicantIncome', 'LoanAmount', 'CoapplicantIncome', 'Dependents', 'Property_Area', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status']]
df.dropna(inplace = True)

st.markdown("<h1 style='text-align: center; color: #514BFF;'>LOAN PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; margin-top: 0rem; color: #514BFF;'>BUILT BY SEYI OLORUNHUNDO</h4>", unsafe_allow_html=True)

st.image('pngwing.com (12).png', width=250, use_column_width=True)
st.markdown("<h4 style='color: #1F4172; text-align: center; font-family: Arial, sans-serif;'>Project Overview</h4>", unsafe_allow_html=True)
st.markdown("<p> This Financial domain's predictive modeling project seeks to harness state-of-the-art machine learning techniques, focusing on building a robust and highly accurate model for predicting loan approval to customers. Through in-depth analyses of historical data from bank account holders, the project identifies key features encapsulating account holder  personal and bank transaction detail information and patterns among approximately 614 individuals. The primary objective is to predict individuals most likely to granted a bank loan based on predetermined variables. At its core, the project aims to establish a reliable machine learning model capable of effectively predicting individuals fit for granting loans based on their personal information and bank transaction history. It considers essential features such as applicant_income, loan_amount, credit_history, property_area, loan_amount_term and other influential factors. The overarching goal is to create a versatile model adaptable to diverse loan application scenarios, providing meaningful and actionable predictions for a broad spectrum of enterprises. This initiative aims to empower financial institutions with a potent tool that not only anticipates individuals' financial behaviors but also contributes to strategic loan application decision-making, ultimately assisting human descision making in the granting of loans that will support customers and wom't be a bad debt to the borrowing institution. </p>", unsafe_allow_html=True)

scaler = StandardScaler()


data['Dependents'] = data['Dependents'].str.replace('+', '')
df['Dependents'] = pd.to_numeric(data['Dependents'], errors = 'coerce')


encoders = {}

for i in df.select_dtypes(exclude = 'number').columns:
    encoder = LabelEncoder()
    df[i] = encoder.fit_transform(df[i])
    encoders[i + '_encoder'] = encoder


st.markdown("<h4 style='color: #F0F6F5; text-align: center; font-family: Arial, sans-serif;'>DATA</h4>", unsafe_allow_html=True)
st.dataframe(data)

x = df.drop('Loan_Status', axis = 1)
y = df.Loan_Status

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 4, stratify = y)
model = LogisticRegression()

model.fit(xtrain, ytrain)

st.markdown("<h4 style='color: #1F4172; text-align: center; font-family: Arial, sans-serif;'>PREDICTOR MODEL</h4>", unsafe_allow_html=True)

st.sidebar.image('pngwing.com (13).png', width=100, use_column_width=True, caption='Welcome User')
st.markdown("<br>", unsafe_allow_html=True)

applicant_income = st.sidebar.number_input('PRIMARY APPLICANT INCOME', data['ApplicantIncome'].min(), data['ApplicantIncome'].max())
amount = st.sidebar.number_input('LOAN AMOUNT', data['LoanAmount'].min(), data['LoanAmount'].max())
coapplicant_income = st.sidebar.number_input('COAPPLICANT INCOME', data['CoapplicantIncome'].min(), data['CoapplicantIncome'].max())
dependents = st.sidebar.number_input('NUMBER OF DEPENDANTS', df['Dependents'].min(), df['Dependents'].max())
property_area = st.sidebar.selectbox('AREA CATEGORY OF APPLICANTS PROPERTY', data['Property_Area'].unique())
loan_term = st.sidebar.number_input('LOAN TENURE', data['Loan_Amount_Term'].min(), data['Loan_Amount_Term'].max())
credit_history = st.sidebar.number_input('APPLICANT CREDIT HISTORY', data['Credit_History'].min(), data['Credit_History'].max())



input_var = pd.DataFrame({
    'ApplicantIncome': [applicant_income],
    'LoanAmount': [amount],
    'CoapplicantIncome': [coapplicant_income],
    'Dependents': [dependents],
    'Property_Area': [property_area],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
})

input_var['Property_Area'] = encoders['Property_Area_encoder'].transform(input_var['Property_Area'])

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h5 style='margin: -30px; color: olive; font:sans-serif' >", unsafe_allow_html=True)
st.dataframe(input_var)



prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred:
        # Include the prediction step here
        predicted = model.predict(input_var)
        output = 'LOAN NOT APPROVED' if predicted[0] == 0 else 'LOAN APPROVED'
        st.success(f'The individual is predicted to {output}')
        st.balloons()


