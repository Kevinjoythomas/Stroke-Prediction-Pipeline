import json
import logging
import os
import joblib
import pytest
from prediction_service.prediction import form_response, api_response
import prediction_service

# Assuming your fixtures are defined in the same module or imported from another module



def test_form_for_correct_range(input_data, target_features):
    data = input_data["correct_range"]
    res = form_response(data)
    assert target_features["min"] <= res <= target_features["max"]

def test_api_response(input_data, target_features):
    data = input_data["correct_range"]
    res = api_response(data)
    print("THE RESULTS IS \n \n ",res)
    assert target_features["min"] <= res['response'] <= target_features["max"]


def test_form_for_incorrect_range(input_data):
    data = input_data["incorrect_range"]
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)

def test_api_for_incorrect_range(input_data):
    data = input_data["incorrect_range"]
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_for_incorrect_col(input_data):
    data = input_data["incorrect_col"]
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message

def test_form_for_incorrect_col(input_data):
    data = input_data["incorrect_col"]
    with pytest.raises(prediction_service.prediction.NotInCols):
        res = form_response(data)
