#  read the data from source
# save it in the data/raw for further process
import os 
from get_data import read_params, get_data
import argparse
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def fillnull(df):
    features = ['age', 'avg_glucose_level','heart_disease','Residence_type','work_type', 'gender','stroke','smoking_status']
    df_with_bmi = df[df['bmi'].notnull()]
    X = pd.get_dummies(df_with_bmi[features], drop_first=True)
    y = df_with_bmi['bmi']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    bmi_predictor = XGBRegressor(learning_rate= 0.01, max_depth= 5, n_estimators= 200, random_state=42)
    bmi_predictor.fit(X_train, y_train)
    dummies = pd.get_dummies(df[features], drop_first=True)

    X_missing = dummies.loc[df['bmi'].isnull()]

    # Now, you can make predictions
    df.loc[df['bmi'].isnull(), 'bmi'] = bmi_predictor.predict(X_missing)
    return df
    
def load_and_save(config_path):
    config = read_params(config_path)
    df =get_data(config_path)
    raw_data_path = config['load_data']['raw_dataset_csv']
    df.drop(['ever_married',"id"],inplace=True,axis=1)
    df = fillnull(df)
    pd.set_option('future.no_silent_downcasting', True)
    df['gender'] = df['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
    df['smoking_status'] = df['smoking_status'].replace({'Unknown':0,'never smoked':1,'formerly smoked':2,"smokes":3}).astype(np.uint8)
    df['Residence_type'] = df['Residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)

    df['work_type'] = df['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)
    new_cols = [col.replace(" ","_") for col in df.columns]
    df.to_csv(raw_data_path,sep=",",index=False, header=new_cols)
    # print(df.columns)
    # print(df.isnull().count())
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)