# load data
# train model
# save the metrics

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from get_data import read_params
import argparse
import joblib
import json 
import mlflow
from urllib.parse import urlparse

def eval(pred, actual):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return (rmse, mae, r2)

def train(config_path):
    config = read_params(config_path=config_path)
    test_data_path = config['split_data']['test_path']
    train_data_path = config['split_data']['train_path']
    random_state =  config['base']['random_state']
    model_dir = config['model_dir']
    
    alpha = config['estimators']['ElasticNet']['params']['alpha']
    l1_ratio = config['estimators']['ElasticNet']['params']['l1_ratio']
    
    target_col = config['base']['target_col']
    train = pd.read_csv(train_data_path,sep = ",")
    test = pd.read_csv(test_data_path,sep = ",")
    
    train_y = train[target_col] 
    test_y = test[target_col]
    
    train_x = train.drop(target_col,axis = 1)
    test_x = test.drop(target_col,axis = 1)
    
    ######### mlflow
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config['remote_server_url']
    print(remote_server_uri)

    # Set the MLflow tracking URI to the remote server
    mlflow.set_tracking_uri(remote_server_uri)
    print("setting experiment")

    # Start the MLflow run
    with mlflow.start_run(run_name=mlflow_config['run_name']) as mlops_run:
        
        # Train the model
        lr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state
        )
        
        lr.fit(train_x, train_y)
        
        # Make predictions and evaluate
        elastic_predicted = lr.predict(test_x)
        (rmse, mae, r2) = eval(elastic_predicted, test_y)

        # Log parameters and metrics to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # Get the tracking URI scheme (file or http)
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        print("tracking url")

        # Log the model based on the artifact storage type
        if tracking_url_type_store != "file":
            # Log the model to the remote tracking server
            mlflow.sklearn.log_model(
                lr,
                "model",
                registered_model_name=mlflow_config["registered_model_name"]
            )
        else:
            # Log the model locally (in case of file storage)
            mlflow.sklearn.log_model(lr, "model")


   
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train(config_path=parsed_args.config)