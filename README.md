# Predictive Forecasting of UAC Care Load & Placement Demand 📈

This project introduces predictive modeling to establish proactive, forward-looking intelligence for the Department of Health and Human Services (HHS) Unaccompanied Alien Children (UAC) Program. It shifts operations from reactive reporting to proactive capacity management.

## Project Objectives
* **Short-term Forecasting:** Daily prediction array establishing active UAC counts in HHS capacities.
* **Flow Imbalance Insights:** Modeling variations between Customs and Border Protection (CBP) intake transfers vs HHS successful discharges (Placement demand).
* **Early-Warning Indicators:** Intelligent automated alert trigger anticipating infrastructure overload via a multi-model validation architecture.

## Methodology & Engineering
This pipeline dynamically compares traditional Baseline persistence logic directly against rigorous Statistical models (SARIMA / Exponential Smoothing) and deep Machine Learning Regression (Random Forests, Gradient Boosting) utilizing Walk-Forward Validation without data leakage.

* **Engineered Features Include:** Net Flow (Transfers-Discharges), Rolling 7/14-day Windows, and customized Lags (t-1, t-7, t-14).
* **Model Evaluation Metrics:** Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE), RMSE, and precisely segmented short term Horizon Error mapping.

## Deliverables Include
1. `/docs/Research_Paper.md` - Complete Data Analytics & Methodology documentation.
2. `/docs/Executive_Summary.md` - Non-technical high-level findings and strategic guidance.
3. `app.py` - Production-ready Streamlit frontend delivering interactive diagnostic dashboard features to end-users natively.  

## Installation & Deployment

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Analytical Dashboard:**
   ```bash
   streamlit run app.py
   ```
*(Available locally at `http://localhost:8501`)*

### Stack Requirements
* Python 3.9+
* Pandas, Numpy, Scikit-learn, Statsmodels
* Streamlit & Plotly
