ENERGY EFFICIENCY PREDICTIONS IN BUILDINGS 

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
