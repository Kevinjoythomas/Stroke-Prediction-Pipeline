# read
# process
# return 

import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
        
def get_data(config_path):
    config = read_params(config_path)
    
    data_path = config["base"]["data_source"]["s3_source"]
    df = pd.read_csv(data_path,sep = ",",encoding='utf-8')
    print(df.head())
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    get_data(config_path=parsed_args.config)