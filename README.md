ENERGY EFFICIENCY PREDICTIONS IN BUILDINGS - WEEK 1


📌 Overview
This project is about predicting the energy efficiency of buildings.
In Week 1, the main goal is to explore the dataset and understand how building features (like wall area, roof area, glazing area, etc.) affect heating and cooling loads.


📂 Files in this Repository
Energy Efficiency Prediction in Buildings - Week 1 Task.ipynb → Jupyter Notebook with code and analysis
README.md → Project details


🛠️ Tools Used
Python
Jupyter Notebook
Pandas & NumPy (data handling)
Matplotlib & Seaborn (visualization)


📊 Dataset Information

The dataset includes building parameters such as:
Surface Area
Wall Area
Roof Area
Height
Orientation
Glazing Area
Target values:
Heating Load
Cooling Load


🚀 Week 1 Work Done
Loaded and explored the dataset
Checked for missing values
Did basic data visualization
Found relationships between features and energy loads



ENERGY EFFICIENCY PREDICTIONS IN BUILDINGS - WEEK 2

This project analyzes building architectural features to predict energy efficiency (heating and cooling loads) using classic machine learning regression models.

Project Overview

Predicts energy efficiency (heating and cooling loads) for buildings
Uses Python with pandas, scikit-learn, matplotlib, and seaborn
Implements Linear Regression, Decision Tree Regressor, and Random Forest Regressor
Evaluates models using RMSE and R² metrics

📊 Dataset

Features: Wall area, roof area, surface area, glazing area, orientation, etc.

Target variables: Heating load (Y1), Cooling load (Y2)

The dataset is loaded and briefly explored within the notebook

Getting Started
Requirements

Python 3.6+
pandas
numpy
matplotlib
seaborn
scikit-learn

Project Structure

Energy-Efficiency-Prediction-in-Buildings-Week-2-Task.ipynb: Main notebook with data loading, preprocessing, modeling, and evaluation

dataset.xlsx: Raw data (make sure to provide or download as needed)

Results

Provides exploratory data analysis and feature importances
Compares regression model performances
Best model (Random Forest) gives the highest accuracy for energy predictions

Acknowledgements

Dataset and framework inspired by classic energy efficiency research and similar projects on energy use in buildings


ENERGY EFFICIENCY PREDICTIONS IN BUILDINGS - WEEK 3

# 🏠 Energy Efficiency Prediction in Buildings (Heating & Cooling Load)

## 📌 Project Overview
This project focuses on predicting **Heating Load** and **Cooling Load** of buildings using **Machine Learning**.  
By analyzing building design parameters (e.g., wall area, roof area, orientation, glazing area), we can estimate energy efficiency.  

The solution is deployed as an **interactive Streamlit web app**, where users can:
- Upload their own dataset (Excel/CSV).
- Merge it with the UCI ENB2012 dataset.
- Train a **RandomForest Regressor** model.
- Evaluate accuracy (R² score, RMSE).
- Make custom predictions using sliders.
- Download the trained ML model.

---

## 📂 Dataset
- **Primary Dataset:** [UCI Energy Efficiency Dataset (ENB2012)](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)  
- Contains building characteristics like:
  - Relative Compactness
  - Surface Area
  - Wall Area
  - Roof Area
  - Overall Height
  - Orientation
  - Glazing Area
  - Glazing Area Distribution
- **Target Variables:**
  - Heating Load
  - Cooling Load

---

## ⚙️ Technologies Used
- Python 3.10+  
- Pandas, NumPy (Data Processing)  
- Scikit-Learn (Model Training)  
- Streamlit (Deployment)  
- Joblib (Model Saving)  

---
