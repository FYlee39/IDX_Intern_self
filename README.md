# 🏠 Housing Price Prediction Web App

A multipage Streamlit application that predicts housing prices using a trained machine learning model.  
Users can upload a CSV dataset, validate inputs, generate predictions, and explore the model structure interactively.

This app demonstrates a production-style machine learning inference workflow, including data validation, model serving, and user-friendly visualization.

---

## Features

### 1. Upload and Predict
- Upload a CSV file containing housing features
- Automatically validate required columns
- Generate predictions using a trained model
- Download prediction results as a CSV file
- Persist uploaded data across pages using session state

### 2. Model Information
- Display the trained model structure
- Render the model visualization similar to Jupyter Notebook
- Show estimator parameters interactively

### 3. Multipage Navigation
- Home page with instructions
- Upload & Predict page for inference
- Model Info page for transparency
- About page for documentation

---

##  Project Structure

```{text}
web_app/
│
├── main.py
├── utils.py
├── state_helpers.py
├── model.pkl
│
├── home.py
├── upload_predict.py
├── model_info.py
├── about.py
│
└── requirements.txt
```

---

##  Model Overview

The application loads a trained machine learning model and performs batch prediction on uploaded datasets.

Typical pipeline:
```{text}
Input CSV  
→ Feature Validation  
→ Preprocessing  
→ Machine Learning Model  
→ Predicted Price  
```

---

## Input Requirements

The uploaded CSV file must contain the required feature columns.

Example:

```{text}

```

If required columns are missing, the app will display a validation error.

---


---

## ⚙️ Key Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Error Handling

The app automatically detects:

- Missing columns
- Empty datasets
- Invalid file formats
- Missing model files

---

## 📌 Notes

This application is intended for demonstration and evaluation purposes.  
Predictions are estimates and should not be used as official property valuations.

---
