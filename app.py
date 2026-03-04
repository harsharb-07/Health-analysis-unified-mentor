import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="UAC Predictive Forecasting Dashboard", layout="wide", page_icon="📈")

# 1. Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/uac_data.csv')
    except:
        # Fallback if file not exactly placed
        df = pd.read_csv('uac_data.csv')
    # Time-series Preparation
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Ensure continuity / Handle missing days via interpolation or masking
    df = df.resample('D').interpolate(method='linear')
    return df

df = load_data()

# 2. Engineered Features for ML Models
@st.cache_data
def create_features(df):
    data = df.copy()
    
    # Net flow (Transfers out of CBP - Discharges out of HHS)
    data['Net_Flow'] = data['Children transferred out of CBP custody'] - data['Children discharged from HHS Care']
    
    # Target variable to forecast
    target = 'Children in HHS Care'
    
    # Lag features
    for lag in [1, 7, 14]:
        data[f'HHS_Lag_{lag}'] = data[target].shift(lag)
        
    # Rolling averages and variances
    data['HHS_Roll_Mean_7'] = data[target].rolling(window=7).mean()
    data['HHS_Roll_Var_7'] = data[target].rolling(window=7).var()
    data['HHS_Roll_Mean_14'] = data[target].rolling(window=14).mean()
    
    # Calendar features
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    
    data.dropna(inplace=True)
    return data

ml_df = create_features(df)

# Dashboard Sidebar
st.sidebar.title("Configuration parameters")
horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=14)
selected_model = st.sidebar.selectbox("Select Model", ["Gradient Boosting", "Random Forest", "SARIMA", "Exponential Smoothing", "Moving Average", "Naive Persistence"])

threshold = st.sidebar.number_input("Shelter Capacity Limit (HHS)", value=8000, step=500)

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.markdown("Detailed predictive intelligence for managing Unaccompanied Alien Children (UAC) capacity, analyzing flow imbalance, and modeling placement demand.")

# ==================== KPIs ====================
st.header("Executive KPIs")

def train_eval_metrics(df, model_type, horizon):
    # Train-test strategy: Strict time-based split (no random sampling)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    target = 'Children in HHS Care'
    features = [c for c in df.columns if c != target and c != 'Children_discharged_from_HHS_Care']
    
    # Walk-forward validation arrays
    y_test_pred = np.zeros(len(test))
    
    if model_type in ["Random Forest", "Gradient Boosting"]:
        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]
        model = RandomForestRegressor(n_estimators=50, random_state=42) if model_type == "Random Forest" else GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_test_true = y_test.values
    elif model_type == "Naive Persistence":
        y_test_true = test[target].values
        preds = []
        history = list(train[target].values)
        for i in range(len(test)):
            preds.append(history[-1])
            history.append(y_test_true[i])
        y_test_pred = np.array(preds)
    elif model_type == "Exponential Smoothing":
        y_test_true = test[target].values
        preds = []
        history = list(train[target].values)
        for i in range(len(test)):
            model_fit = ExponentialSmoothing(history[-90:], trend='add', seasonal=None).fit(optimized=True)
            preds.append(model_fit.forecast(1)[0])
            history.append(y_test_true[i])
        y_test_pred = np.array(preds)
    elif model_type == "Moving Average":
        y_test_true = test[target].values
        preds = []
        history = list(train[target].values)
        for i in range(len(test)):
            yhat = np.mean(history[-7:])
            preds.append(yhat)
            history.append(y_test_true[i]) 
        y_test_pred = np.array(preds)
    else: # SARIMA
        y_test_true = test[target].values
        preds = []
        history = list(train[target].values)
        for i in range(len(test)):
            model = SARIMAX(history[-60:], order=(1, 1, 0))
            model_fit = model.fit(disp=False)
            preds.append(model_fit.forecast(1)[0])
            history.append(y_test_true[i])
        y_test_pred = np.array(preds)
        
    mae = mean_absolute_error(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mape = mean_absolute_percentage_error(y_test_true, y_test_pred)
    
    # Horizon Error: Short vs medium-term reliability based on the selected horizon
    horizon_error = mean_absolute_error(y_test_true[:horizon], y_test_pred[:horizon]) if len(y_test_true) >= horizon else mae
    
    return mae, rmse, mape, horizon_error, y_test_pred, y_test_true

mae, rmse, mape, horizon_error, y_test_pred, y_test_true = train_eval_metrics(ml_df, selected_model, horizon)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast Accuracy (1-MAPE)", f"{(1 - mape)*100:.1f}%")
col2.metric("Mean Abs Error (MAE)", f"{mae:.0f} Children")
col3.metric("Current Active HHS Load", f"{int(df['Children in HHS Care'].iloc[-1])}")

# Simulate lead time and capacity breach
breach_risk = "Low"
forecast_max = pd.Series(y_test_pred[:horizon]).max()
if forecast_max > threshold:
    breach_risk = "High"
    lead_time = pd.Series(y_test_pred[:horizon])[pd.Series(y_test_pred[:horizon]) > threshold].index[0]
else:
    lead_time = "> 30 Days"

col4.metric("Capacity Breach Probability", breach_risk)

if breach_risk == "High":
    st.error(f"⚠️ EARLY WARNING: Projected capacity breach in {lead_time} days. Immediate resource scaling required.")
else:
    st.success("✅ Capacity levels look stable for the forecast horizon.")

st.markdown("---")

# ==================== Core Modules ====================
st.header("Future Care Load Forecast - HHS Setup")

# Generate live future predictions
def generate_future_forecast(df, selected_model, horizon):
    target = 'Children in HHS Care'
    
    if selected_model in ["Random Forest", "Gradient Boosting"]:
        features = [c for c in df.columns if c != target and c != 'Children_discharged_from_HHS_Care']
        X = df[features]
        y = df[target]
        
        model = RandomForestRegressor(n_estimators=50, random_state=42) if selected_model == "Random Forest" else GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Iterative forecasting
        last_row = df.iloc[-1:].copy()
        forecasts = []
        for _ in range(horizon):
            pred = model.predict(last_row[features])[0]
            forecasts.append(pred)
            # Update lag features
            last_row['HHS_Lag_14'] = last_row['HHS_Lag_7']
            last_row['HHS_Lag_7'] = last_row['HHS_Lag_1']
            last_row['HHS_Lag_1'] = pred
            # Roll means statically simplified
            last_row['HHS_Roll_Mean_7'] = (last_row['HHS_Roll_Mean_7'] * 6 + pred) / 7
        return np.array(forecasts)
    elif selected_model == "Naive Persistence":
        last_val = df[target].iloc[-1]
        return np.array([last_val] * horizon)
    elif selected_model == "Exponential Smoothing":
        model_fit = ExponentialSmoothing(df[target].values[-90:], trend='add', seasonal=None).fit(optimized=True)
        return model_fit.forecast(steps=horizon)
    elif selected_model == "Moving Average":
        history = list(df[target].values[-7:])
        forecasts = []
        for _ in range(horizon):
            pred = np.mean(history[-7:])
            forecasts.append(pred)
            history.append(pred)
        return np.array(forecasts)
    else: # SARIMA
        model = SARIMAX(df[target].values, order=(1, 1, 1))
        model_fit = model.fit(disp=False)
        return model_fit.forecast(steps=horizon)

future_preds = generate_future_forecast(ml_df, selected_model, horizon)
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon)

# Plotting historical + forecast
fig_careload = go.Figure()
# Historical
fig_careload.add_trace(go.Scatter(x=df.index[-90:], y=df['Children in HHS Care'].iloc[-90:], mode='lines', name='Historical Active Load', line=dict(color='blue')))
# Forecast
fig_careload.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Forecasted Load', line=dict(color='red', dash='dash')))
# Threshold line
fig_careload.add_trace(go.Scatter(x=[df.index[-90], future_dates[-1]], y=[threshold, threshold], mode='lines', name='Capacity Limit', line=dict(color='grey', dash='dot')))

# Confidence intervals (Simplistic approximation for visual requirement)
if selected_model in ["Gradient Boosting", "SARIMA"]:   
    uncertainty = future_preds * 0.05
    fig_careload.add_trace(go.Scatter(
        name='Upper Bound',
        x=future_dates, y=future_preds + uncertainty,
        mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False
    ))
    fig_careload.add_trace(go.Scatter(
        name='Lower Bound',
        x=future_dates, y=future_preds - uncertainty,
        marker=dict(color="#444"), line=dict(width=0), mode='lines', fillcolor='rgba(68, 68, 68, 0.3)', fill='tonexty', showlegend=False
    ))

fig_careload.update_layout(title="HHS Active Care Load Forecast (Next {} Days)".format(horizon), xaxis_title="Date", yaxis_title="Children in Care")
st.plotly_chart(fig_careload, use_container_width=True)

# ==================== Discharge Demand Panel ====================
st.header("Discharge Demand vs Imbalance")
st.markdown("Monitor the net flow pressure. If transfer intakes exceed successful discharges, the net flow goes positive, building capacity stress.")

fig_flow = go.Figure()
fig_flow.add_trace(go.Scatter(x=df.index[-90:], y=df['Children transferred out of CBP custody'].iloc[-90:], mode='lines', name='Inflow (Transfers in)', line=dict(color='orange')))
fig_flow.add_trace(go.Scatter(x=df.index[-90:], y=df['Children discharged from HHS Care'].iloc[-90:], mode='lines', name='Outflow (Discharges out)', line=dict(color='green')))
fig_flow.add_trace(go.Bar(x=df.index[-90:], y=(df['Children transferred out of CBP custody'] - df['Children discharged from HHS Care']).iloc[-90:], name='Net Flow Pressure', marker_color='red', opacity=0.5))

fig_flow.update_layout(title="Flow Dynamics (Last 90 Days)", xaxis_title="Date", yaxis_title="Number of Children")
st.plotly_chart(fig_flow, use_container_width=True)

# ==================== Model Selection & Assessment ====================
st.header("Model Performance & Diagnostics")
st.markdown("We compare Statistical models against Machine Learning models on Holdout sets.")

@st.cache_data
def get_all_metrics(df, hz):
    metrics = {}
    for mod in ["Naive Persistence", "Moving Average", "Exponential Smoothing", "SARIMA", "Random Forest", "Gradient Boosting"]:
        m_mae, m_rmse, m_mape, m_herr, _, _ = train_eval_metrics(df, mod, hz)
        metrics[mod] = (m_mae, m_rmse, m_mape, m_herr)
        
    return {
        "Model": list(metrics.keys()),
        "MAE": [int(v[0]) for v in metrics.values()],
        "RMSE": [int(v[1]) for v in metrics.values()],
        "MAPE": [f"{v[2]*100:.2f}%" for v in metrics.values()],
        "Horizon Error": [int(v[3]) for v in metrics.values()]
    }

perf_data = get_all_metrics(ml_df, horizon)

st.table(pd.DataFrame(perf_data).set_index("Model"))

with st.expander("View Residuals & Decomposition Analysis (EDA)"):
    st.markdown("Time Series Decomposition isolates Trend from Seasonality.")
    # Assuming standard frequency 7 for daily data representing week
    decomposition = seasonal_decompose(df['Children in HHS Care'], model='additive', period=7)
    
    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Scatter(x=df.index[-100:], y=decomposition.trend[-100:], name="Trend"))
    fig_decomp.add_trace(go.Scatter(x=df.index[-100:], y=decomposition.seasonal[-100:], name="Seasonality"))
    fig_decomp.update_layout(title="Trend & Weekly Seasonality (Last 100 days)")
    st.plotly_chart(fig_decomp, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Unified Mentor | U.S Dept of HHS Project")
