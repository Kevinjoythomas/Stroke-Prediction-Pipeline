# 🧠 Stroke Prediction Pipeline 🚀 | MLOps at Scale

Welcome to the **Stroke Prediction Pipeline** project! This project covers all stages of the ML lifecycle – from data ingestion to model serving in a production-ready pipeline. With CI/CD automation, MLFlow tracking, and DVC pipeline orchestration, this repository demonstrates an end-to-end MLOps framework designed to tackle real-world machine learning challenges efficiently.

## 🔍 Overview
This project builds a **logistic regression model** to predict the likelihood of stroke based on a dataset. With a modular pipeline architecture and MLFlow tracking, it captures the complete process of building, training, evaluating, and deploying the model. Data is pulled directly from **AWS S3**, and the model is served on a **Flask server** hosted on **AWS EC2**.

### 🌐 Live Model Endpoint:
**Stroke Prediction Flask App**

---

## 🚀 Key Features

### 🛠️ End-to-End Machine Learning Pipeline
- **DVC (Data Version Control)** to manage the ML pipeline  
- **AWS S3** used for storage of datasets and intermediate artifacts  
- **MLFlow** to track experiments, metrics, and parameters  

### ✅ CI/CD with GitHub Actions
- Automated testing using **Pytest** and **Tox**  
- Deployment pipeline ensures continuous integration and delivery with every code change  

### 📊 Comprehensive Evaluation Metrics
**Model:** Logistic Regression

| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Class 0      | 0.83      | 0.78   | 0.80     | 2939    |
| Class 1      | 0.79      | 0.84   | 0.81     | 2895    |
| **Accuracy** |           |        | **0.81** | 5834    |
| **Macro Avg**| 0.81      | 0.81   | 0.81     | 5834    |
| **Weighted Avg** | 0.81 | 0.81   | 0.81     | 5834    |

**F1 Score:** 0.8105

---

## 📂 Project Structure

Stroke-Prediction-Pipeline/ │ ├── data/ # Raw and processed data ├── models/ # Model artifacts (stored in AWS S3) ├── notebooks/ # Jupyter notebooks for EDA and prototyping ├── src/ # Source code for the pipeline │ ├── data_preprocessing.py │ ├── feature_engineering.py │ └── train_model.py ├── tests/ # Unit and integration tests (Tox & Pytest) ├── .github/workflows/ # CI/CD pipeline definition (GitHub Actions) ├── requirements.txt # Python dependencies ├── dvc.yaml # DVC pipeline configuration ├── app.py # Flask server for model serving └── README.md # Project documentation

# 🧪 Model Development Stages

### Exploratory Data Analysis (EDA)
- Extensive EDA to understand data distributions, patterns, and outliers  
- Feature selection to retain only the most relevant predictors  

### Preprocessing & Pipeline
- Handled missing values and data imbalances  
- Created pipelines for data ingestion, preprocessing, feature extraction, and model training  

### Model Training & Evaluation
- **Model:** Logistic Regression  
- Tracked hyperparameters and metrics using **MLFlow**  
- Stored datasets and models with **DVC** for seamless version control  

### Deployment on AWS EC2
- Hosted the Flask app on **AWS EC2** with the model endpoint ready for predictions  

---

## ⚙️ CI/CD Pipeline
🚀 This project uses **GitHub Actions** to automate testing, linting, and deployment.  
Every push triggers the following:

- **Code Quality Checks:** Run Tox for testing across environments  
- **Automated Testing:** Use Pytest to verify functionality  
- **Deployment to EC2:** If all tests pass, the app is redeployed to the AWS server  

---

## 🌐 Flask App Usage
Once the Flask server is running, predictions can be made by sending a POST request with a CSV file to:

http://ec2-13-60-233-133.eu-north-1.compute.amazonaws.com:5000/predict

# 🧑‍🔬 Testing  
The project uses **Pytest** for unit tests and **Tox** for cross-environment testing.

## 🎯 Key Technologies Used
- **Machine Learning**: Logistic Regression
- **MLOps**: DVC, MLFlow
- **Cloud**: AWS EC2, S3
- **CI/CD**: GitHub Actions
- **Testing**: Pytest, Tox
- **Server**: Flask + Gunicorn

## 🛡️ Results
The model achieved **81% accuracy**, with balanced precision and recall for both classes, making it a reliable predictor for stroke detection.
