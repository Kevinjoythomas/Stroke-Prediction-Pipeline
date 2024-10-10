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
    
    lr = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state
    )
    
    lr.fit(train_x,train_y)
    
    elastic_predicted = lr.predict(test_x)
    (rmse, mae, r2) = eval(elastic_predicted,test_y)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir,"model.joblib")
    joblib.dump(lr,model_path)
    
    scores_file = config['reports']['scores']
    params_file = config['reports']['params']
    with open(scores_file,"w") as f:
        scores = {
            "rmse":rmse,
            "mae":mae,
            "r2":r2
        }
        json.dump(scores,f, indent=4)

    with open(params_file,"w") as f:
        params = {
            "L1_ratio":l1_ratio,
            "alpha":alpha,
            "random_state":random_state
        }
        json.dump(params,f, indent=4)
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train(config_path=parsed_args.config)