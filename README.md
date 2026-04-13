


#  README.md 


# ❤️ Heart Disease Detection System

## 📌 Overview
This project is an AI-based Heart Disease Detection System that combines two approaches:

- 🧠 Rule-Based Expert System using Experta
- 🤖 Machine Learning Model using Decision Tree Classifier

The system analyzes patient health data to predict the risk of heart disease and compares both approaches in terms of accuracy and interpretability.

---

## 📁 Project Structure

```

Heart_Disease_Detection/
│
├── data/                  # Raw and cleaned datasets
├── notebooks/             # Data analysis and model development
├── ml_model/             # Trained machine learning model (model.pkl)
├── rule_based_system/    # Expert system rules and engine
├── ui/                   # Streamlit web application
├── reports/              # Model evaluation and comparison
├── utils/                # Data preprocessing scripts
│
├── README.md
├── requirements.txt

````

---

## ⚙️ Workflow

### 1. Data Preprocessing
- Handling missing values
- Removing duplicates
- Encoding categorical variables
- Feature scaling using MinMaxScaler
- Saving cleaned dataset

---

### 2. Data Visualization
- Histograms for feature distribution
- Boxplots for outlier detection
- Correlation heatmap
- Feature importance analysis

---

### 3. Machine Learning Model
- Algorithm: Decision Tree Classifier
- Train/Test split: 80/20
- Hyperparameter tuning using GridSearchCV
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Model saved as `model.pkl`

---

### 4. Rule-Based Expert System
- Built using Experta library
- Contains 10+ medical rules
- Based on IF-THEN logic
- Outputs risk levels:
  - Low Risk
  - Medium Risk
  - High Risk

---

### 5. Model Comparison
Both systems are compared based on:
- Accuracy
- Explainability
- Practical usability

---

## 🚀 How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Run the Streamlit App

```bash
streamlit run ui/app.py
```

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Experta
* Streamlit
* Joblib

---

## 📊 Results Summary

* Machine Learning model provides higher predictive accuracy
* Expert System provides better interpretability and rule transparency
* Combining both gives a balanced intelligent system

---



# 📦 requirements.txt 

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
experta
````

---


