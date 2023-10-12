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

#Define  the function
#function to load the encoder
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "model", "ml.pkl")

#useful functions
gr.cache_resource()
def  load_ml_components(fp):
    "load the ml components to re-use in app"
    with open(fp, 'rb') as file:
        obj = pickle.load(file)
        return obj
    
#Execution
ml_components_dict = load_ml_components(fp = ml_core_fp)
encoder= ml_components_dict["enc"]
num_imputer = ml_components_dict["num_imputer"]
cat_imputer = ml_components_dict["cat_imputer"]
scaler = ml_components_dict["stds"]


# Specify the path to your saved model
customer_churn_model = 'compare_models[0]'
customer_churn_path = os.path.join(DIRPATH, customer_churn_model)

# Load the model
model = ml_components_dict["models"]

#functionto process input and return prediction
def customer(*args,encoder = encoder,model= model,scaler=scaler):
# convert input into a dataframe
    input_data = pd.DataFrame([args],columns=expected_inputs)
    #make the prediction
    model_output =model.predict(input_data)

    # return the prediction 
    return{"Prediction: The customer is likely to churn":(model_output[0]),
           "Prediction: The customer will not churn":1-float(model_output[0])}