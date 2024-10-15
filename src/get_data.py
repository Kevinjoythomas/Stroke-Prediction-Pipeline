import os
import yaml
import pandas as pd
import argparse
import boto3
from io import BytesIO  # Change this line

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    bucket_name = config["data_source"]["s3_bucket"]
    file_key = config["data_source"]["s3_file_key"]
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    s3 = boto3.client('s3', 
                  aws_access_key_id=aws_access_key_id, 
                  aws_secret_access_key=aws_secret_access_key)
    
    csv_buffer = BytesIO() 
    s3.download_fileobj(bucket_name, file_key, csv_buffer)
    
    csv_buffer.seek(0)  
    df = pd.read_csv(csv_buffer, sep=",", encoding='utf-8')
    return df

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)

    # Optionally, print or use the DataFrame
    print(data.head())
