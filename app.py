#import key libraries and packages

import numpy as np 
import pandas as pd 
import pickle
import gradio as gr
import  os

expected_inputs = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']
numerics = ['tanure','PaymentMethod','TotalCharges']
categoricals = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod']


    
#Execution
ml_components_dict = load_ml_components(fp = ml_core_fp)
encoder= ml_components_dict["enc"]
num_imputer = ml_components_dict["num_imputer"]
cat_imputer = ml_components_dict["cat_imputer"]
scaler = ml_components_dict["stds"]


# Specify the path to your saved model
customer_churn_model = 'compare_models[0]'
customer_churn_path = os.path.join(DIRPATH, customer_churn_model)


