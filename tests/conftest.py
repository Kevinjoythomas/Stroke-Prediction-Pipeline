import pytest
import os
import json



@pytest.fixture
def schema_in(schema_path="schema_in.json"):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

@pytest.fixture
def input_data():
    return {
        "incorrect_range": 
        {
        "gender": 7897897, 
        "age": 99, 
        "hypertension": 2, 
        "heart_disease": 2, 
        "work_type": 555, 
        "Residence_type": 2, 
        "avg_glucose_level": 789, 
        "bmi": 99, 
        "smoking_status": 12, 
    },
    
    "correct_range": {
        "gender": 1, 
        "age": 80, 
        "hypertension": 1, 
        "heart_disease": 1, 
        "work_type": 1, 
        "Residence_type": 1, 
        "avg_glucose_level": 271.74, 
        "bmi": 35.0, 
        "smoking_status": 3, 
    },
    
    "incorrect_col": {
        "gender": 1, 
        "agE": 80, 
        "hypertension": 1, 
        "heart_disease": 1, 
        "work_type": 1, 
        "Residence_type": 1, 
        "avg_glucose_level": 271.74, 
        "bmi": 35.0, 
        "smoking_status": 3, 
    }

    }
@pytest.fixture
def target_features():
    return {
        "min": 0.0,
        "max": 1.0
    }
    