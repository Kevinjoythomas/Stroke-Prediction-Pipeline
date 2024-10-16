# 🧠 Stroke Prediction Pipeline 🚀 | MLOps at Scale

Welcome To The Stroke Prediction Pipeline Project! This Project Covers All Stages Of The ML Lifecycle—From Data Ingestion To Model Serving In a Production-Ready Pipeline. With CI/CD Automation using **Github Actions**, **MLFlow** Tracking, and **DVC** Pipeline Orchestration, This Repository Demonstrates an End-To-End **MLOps** Framework Designed To Tackle Real-World Machine Learning Challenges Efficiently

## 🔍 Overview
This Project Builds a **Logistic Regression** Model To Predict The Likelihood Of Stroke Based On A Dataset. With a Modular Pipeline Architecture and MLFlow Tracking, It Captures The Complete Process Of Building, **Training, Evaluating, and Deploying** The Model. Data is Pulled Directly from **AWS S3**, and The Model Is Served On a Flask Server Hosted on **AWS EC2**.

## 🌐 Live Model Endpoint:
A Website Developed With **HTML** And **CSS**, Hosted On A **Flask** Server, Ensures That All Inputs Fall Within The Recommended Range.
[Stroke Prediction Site](http://ec2-13-60-233-133.eu-north-1.compute.amazonaws.com:5000/)

![Website](https://github.com/Kevinjoythomas/Stroke-Prediction-Pipeline/blob/main/website.png)

---

## Implementation of Google Pipeline Diagram for Project Workflow
![Project Structure](https://github.com/Kevinjoythomas/Stroke-Prediction-Pipeline/blob/main/pipeline.jpeg)

## 🚀 Key Features

## 🛠️ End-to-End Machine Learning Pipeline
- **DVC (Data Version Control)** to manage the ML pipeline  
- **AWS S3** used for storage of datasets and intermediate artifacts  
- **MLFlow** to track experiments, metrics, and parameters  

## ✅ CI/CD with GitHub Actions
- Automated testing using **Pytest** and **Tox**  
- Deployment pipeline ensures continuous integration and delivery with every code change  

## 📊 Comprehensive Evaluation Metrics
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

# 🧪 Model Development Stages

## Exploratory Data Analysis (EDA)
- Extensive EDA to understand data distributions, patterns, and outliers  
- Feature selection to retain only the most relevant predictors  

## Preprocessing & Pipeline
- Handled missing values and data imbalances  
- Created pipelines for data ingestion, preprocessing, feature extraction, and model training  

## Model Training & Evaluation
- **Model:** Logistic Regression  
- Tracked hyperparameters and metrics using **MLFlow**  
- Stored datasets and models with **DVC** for seamless version control  

## Deployment on AWS EC2
- Hosted the Flask app on **AWS EC2** with the model endpoint ready for predictions  

---

## ⚙️ CI/CD Pipeline
🚀 This project uses **GitHub Actions** to automate testing, linting, and deployment.  
Every push triggers the following:

- **Code Quality Checks:** Run Tox for testing across environments  
- **Automated Testing:** Use Pytest to verify functionality  
- **Deployment to EC2:** If all tests pass, the app is redeployed to the AWS server  

---

## 🧑‍🔬 Testing  
The project uses **Pytest** for unit tests and **Tox** for cross-environment testing.

## 🎯 Key Technologies Used
- **Machine Learning**: Logistic Regression
- **MLOps**: DVC, MLFlow
- **Cloud**: AWS EC2, S3
- **CI/CD**: GitHub Actions
- **Testing**: Pytest, Tox
- **Website**: HTML, CSS
- **Server**: Flask + Gunicorn

## 🛡️ Results
The model achieved **81% accuracy**, with balanced precision and recall for both classes, making it a reliable predictor for stroke detection.
