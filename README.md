# 🏦 Bank Customer Churn: MLOps Drift Monitoring System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-Evidently_AI-orange)
![Status](https://img.shields.io/badge/Status-Drift_Detected-red)

## 📌 Project Overview
This project goes beyond standard model training by implementing a robust **MLOps Monitoring System**. It simulates a real-world production environment where customer data changes over time ("Data Drift"), causing model degradation.

Using **Evidently AI**, this pipeline detects when input features deviate from the training distribution, alerting data scientists to retrain the model before performance crashes.

**Key Objective:** Detect "Data Drift" and "Target Drift" in a Bank Churn Prediction model.

## 🛠️ Tech Stack
* **Language:** Python (Jupyter Notebook)
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **MLOps & Monitoring:** Evidently AI (v0.7+)
* **Data Manipulation:** Pandas, NumPy

## 📖 The "Chaos" Methodology
To demonstrate the monitoring capabilities, this project uses a **Synthetic Drift Injection** strategy:

1.  **Phase 1: Reference Data (Normal)**
    * Simulated historical bank data (Normal Age, Normal Salaries).
    * Trained a Random Forest model on this baseline.
2.  **Phase 2: Production Data (Drifted)**
    * Simulated new incoming data but **intentionally introduced drift**.
    * **Drift Scenario:** A sudden influx of younger customers (Student Campaign).
    * **Injection:** Manually shifted `Age` distribution by -20 years and reduced `EstimatedSalary` by 40%.

## 📊 Results & Analysis
The monitoring system successfully flagged the corrupted data streams.

### 1. Dashboard Output
*(Place your screenshot here: `![Dashboard](<img width="1854" height="895" alt="image" src="https://github.com/user-attachments/assets/9c4824bb-0e8c-4b90-8a1a-d931a9b7b8b4" />
)`)*

### 2. Drift Report Findings
* **Data Drift:** **DETECTED** (2 out of 10 features failed statistical tests).
* **Critical Alerts:**
    * `Age`: **High Drift** (Wasserstein Distance: >1.1). The distribution shifted significantly to the left.
    * `EstimatedSalary`: **High Drift**. The distribution narrowed and shifted down.
* **Model Performance:** The model's predictions (`prediction` column) also showed drift, indicating that the model is now behaving differently in production than in training.

## 🚀 How to Run This Project
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/churn-drift-monitoring.git](https://github.com/your-username/churn-drift-monitoring.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas scikit-learn evidently jupyter
    ```
3.  **Launch the Notebook:**
    ```bash
    jupyter notebook
    ```
4.  **Execute the Project:**
    * Open `model_training.ipynb` in the Jupyter interface.
    * Run all cells to generate the data, train the model, and detect drift.
    * The final cell will generate the `drift_report_churn.html` file.

## 🧠 What I Learned
* How to implement **Post-Deployment Monitoring** for ML models.
* The difference between **Reference** (Training) and **Current** (Production) data.
* How to interpret **Wasserstein Distance** and statistical tests for distribution shifts.
* Using **Evidently AI** to generate automated health reports.

---
*Created by [Atharv Tolambiya]*
