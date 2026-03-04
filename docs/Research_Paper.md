# Research Paper: Predictive Forecasting of Care Load & Placement Demand

## 1. Introduction
The Unaccompanied Alien Children (UAC) Program manages the custody, care, and placement of minors who enter the United States without a legal guardian. Because intake numbers exhibit high variance driven by external geopolitical and socioeconomic factors, managing safe shelter capacity has historically been responsive rather than proactive. This study transitions the UAC program from descriptive reporting to predictive forecasting.

## 2. Dataset and Preprocessing
The dataset consists of daily aggregations describing operations across the intake ecosystem:
- **Intake Flow**: `Children apprehended and placed in CBP custody`
- **Holding**: `Children in CBP custody`
- **Transfers**: `Children transferred out of CBP custody`
- **Active Load**: `Children in HHS Care`
- **Outflow**: `Children discharged from HHS Care`

### 2.1 Preprocessing Steps
1. **Time-Series Continuity**: The `Date` column was set as a continuous datetime index to guarantee no missing calendar days exist.
2. **Imputation**: Missing values across minor reporting gaps were linearly interpolated to maintain sequence integrity.
3. **Decomposition**: Analysis revealed strong multi-level seasonality (day-of-week intake spikes, macro annual fluctuations).

## 3. Exploratory Data Analysis (EDA) Insight
1. **CBP to HHS Transfer Delaying**: The dataset highlights a buffering effect. High spikes in CBP apprehensions do not immediately impact HHS care load daily but do so in persistent waves over the subsequent 3-5 days.
2. **System Flow Imbalance**: Computing a `Net Flow` metric (`Children transferred out of CBP custody` - `Children discharged from HHS Care`) acts as a highly sensitive leading indicator for incoming capacity stress in HHS. Continuous positive net flow leads to an exponential increase in Active HHS Care load.

## 4. Methodology and Feature Engineering
To predict medium-term facility stress, we generated the following features:
- **Lag Features**: Observation values at `t-1`, `t-7`, and `t-14`.
- **Rolling Aggregations**: 7-day and 14-day rolling means and variances smooth out temporary intake volatility and surface persistent operational pressure trends.
- **Calendar Proxies**: Day of week and monthly indicators to account for seasonal variations in migration behavior.

## 5. Forecasting Models & Approach
A strict time-based split (walk-forward validation) was executed blocking 80% of data for training and 20% for testing. Random sampling was strictly avoided to prevent temporal data leakage. 

We compared:
- **Baseline Models**: Moving average and persistence models.
- **Statistical Models**: SARIMA, effective at capturing the structured day-of-week and seasonal periodicities.
- **Machine Learning**: Random Forest and Gradient Boosting Regressors. These were better suited for recognizing complex interaction thresholds (e.g., specific capacity limits interacting with specific seasonal patterns).

## 6. Results and Evaluation
- **Mean Absolute Percentage Error (MAPE)**: Gradient Boosting significantly outperformed the ARIMA and Moving Average baselines when forecasting the active `Children in HHS Care` metric.
- **Horizon Decay**: As the forecast horizon expanded past 14 days, all models began reverting to long-term mean expectations, indicating that optimal operational trust should be placed in the 1 to 14-day window.

## 7. Operational Recommendations
- **Early-Warning Alerts**: Stakeholders should configure automated alerts when predictions show a 14-day cumulative imbalance exceeding 10% of total system discharge capacity.
- **Staffing Scale-up**: Medical and caseworker personnel deployments should be initiated precisely when the forecasted Active Care load curve accelerates, roughly 7 days before physical capacity limits are hit.
- **Sponsor Vetting Prioritization**: Short-term discharge demand predictions should act as quotas for caseworker velocity.

## 8. Conclusion
Applying predictive modeling to intake and discharge metrics provides crucial foresight. This capability enables operations directors to seamlessly shift resources to pressure points days before physical capacity limits are breached, securing continuity of quality care.
