import yaml
import os
import json
import numpy as np
import joblib
import pandas as pd
params_path = "params.yaml"
schema_path = os.path.join("prediction_service","schema_in.json")

class NotInRange(Exception):
    def __init__(self, message = "VALUES NOt IN RANGE"):
        self.message = message
        super().__init__(self.message)
        

class NotInCols(Exception):
    def __init__(self, message="Not in columns"):
        self.message = message
        super().__init__(self.message)
        

def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config  = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    try:
        if 3 <= prediction <= 8:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"

def get_schema(schema_path = schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

def validate_input(dict_request):
    def validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCols
    def validate_values(col,val):
        schema = get_schema()
        if not (schema[col]["min"] <= float(val) <= schema[col]["max"]):
            raise NotInRange
    for col, val in dict_request.items():
        validate_cols(col)
        validate_values(col,val)
        
    return True
def form_response(dict_request):
    if validate_input(dict_request):
        data = pd.DataFrame([dict_request]) 
        response = predict(data)
        return response


def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = pd.DataFrame([dict_request]) 
            response = predict(data)
            response = {"response":response} 
            return response
       
    except Exception as e:
        error = {"The expected Range":get_schema(),"response":str(e)}
        return error