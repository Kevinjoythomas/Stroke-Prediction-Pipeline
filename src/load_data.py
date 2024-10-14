import os
import argparse
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from get_data import read_params, get_data

def fillnull(df):
    features = ['age', 'avg_glucose_level', 'heart_disease', 'Residence_type', 
                'work_type', 'gender', 'stroke', 'smoking_status']
    df_with_bmi = df[df['bmi'].notnull()]
    X = pd.get_dummies(df_with_bmi[features], drop_first=True)
    y = df_with_bmi['bmi']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    bmi_predictor = XGBRegressor(learning_rate=0.01, max_depth=5, 
                                 n_estimators=200, random_state=42)
    bmi_predictor.fit(X_train, y_train)
    
    dummies = pd.get_dummies(df[features], drop_first=True)
    X_missing = dummies.loc[df['bmi'].isnull()]
    df.loc[df['bmi'].isnull(), 'bmi'] = bmi_predictor.predict(X_missing)
    
    return df

def apply_smote(X, y):
    """
    Apply SMOTE to oversample the minority class.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Original shape: {X.shape}, Resampled shape: {X_res.shape}")
    return X_res, y_res

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    
    raw_data_path = config['load_data']['raw_dataset_csv']
    df.drop(['ever_married', "id"], inplace=True, axis=1)
    df = fillnull(df)
    
    pd.set_option('future.no_silent_downcasting', True)
    
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': -1}).astype(np.int8)

    df['smoking_status'] = df['smoking_status'].map({'Unknown': 0, 'never smoked': 1, 'formerly smoked': 2, 'smokes': 3}).astype(np.int8)

    df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1}).astype(np.int8)

    df['work_type'] = df['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': -1, 'Never_worked': -2}).astype(np.int8)


    new_cols = [col.replace(" ", "_") for col in df.columns]
    
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    X_res, y_res = apply_smote(X, y)
    
    # Save the resampled data
    resampled_data = pd.concat([X_res, y_res], axis=1)
    resampled_data.to_csv(raw_data_path, sep=",", index=False, header=new_cols)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
