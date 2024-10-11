import pytest
import yaml
import os
import json

@pytest.fixture
def config(config_path = "params.yaml"):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

@pytest.fixture
def schema_in(schema_path="schema_in.json"):
    with open(schema_in) as json_file:
        schema = json.load(json_file)
    return schema

@pytest.fixture
def input_data():
    return {
        "incorrect_range": 
        {"fixed_acidity": 7897897, 
        "volatile_acidity": 555, 
        "citric_acid": 99, 
        "residual_sugar": 99, 
        "chlorides": 12, 
        "free_sulfur_dioxide": 789, 
        "total_sulfur_dioxide": 75, 
        "density": 2, 
        "pH": 33, 
        "sulphates": 9, 
        "alcohol": 9
        },

        "correct_range":
        {"fixed_acidity": 5, 
        "volatile_acidity": 1, 
        "citric_acid": 0.5, 
        "residual_sugar": 10, 
        "chlorides": 0.5, 
        "free_sulfur_dioxide": 3, 
        "total_sulfur_dioxide": 75, 
        "density": 1, 
        "pH": 3, 
        "sulphates": 1, 
        "alcohol": 9
        },

        "incorrect_col":
        {"fixed acidity": 5, 
        "volatile acidity": 1, 
        "citric acid": 0.5, 
        "residual sugar": 10, 
        "chlorides": 0.5, 
        "free sulfur dioxide": 3, 
        "total sulfur dioxide": 75, 
        "density": 1, 
        "pH": 3, 
        "sulphates": 1, 
        "alcohol": 9
        }
    }
@pytest.fixture
def target_features():
    return {
        "min": 3.0,
        "max": 8.0
    }
    