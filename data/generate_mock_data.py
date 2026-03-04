import pandas as pd
import numpy as np

def generate_mock_data(start_date='2022-01-01', end_date='2024-01-01', output_file='data/uac_data.csv'):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    np.random.seed(42)
    
    # Base trends
    trend = np.linspace(50, 200, n)
    # Seasonal components
    seasonality = 50 * np.sin(2 * np.pi * np.arange(n) / 365) + 20 * np.sin(2 * np.pi * np.arange(n) / 7)
    
    # Children apprehended (Intake volume)
    apprehended = np.clip(trend + seasonality + np.random.normal(0, 20, n), 0, None)
    
    # Children in CBP custody (Active load) - usually a moving sum/buffer of apprehended
    cbp_custody = np.zeros(n)
    transferred_out = np.zeros(n)
    
    current_cbp = 500
    for i in range(n):
        intake = apprehended[i]
        # About 30-50% get transferred out daily from CBP to HHS
        transfer = current_cbp * np.random.uniform(0.3, 0.5) 
        current_cbp = current_cbp + intake - transfer
        cbp_custody[i] = current_cbp
        transferred_out[i] = transfer
        
    # Children in HHS Care
    hhs_care = np.zeros(n)
    discharged = np.zeros(n)
    
    current_hhs = 5000
    for i in range(n):
        intake_hhs = transferred_out[i]
        # About 1-3% get discharged daily from HHS
        discharge = current_hhs * np.random.uniform(0.01, 0.03)
        current_hhs = current_hhs + intake_hhs - discharge
        hhs_care[i] = current_hhs
        discharged[i] = discharge
        
    df = pd.DataFrame({
        'Date': dates,
        'Children apprehended and placed in CBP custody': apprehended.astype(int),
        'Children in CBP custody': cbp_custody.astype(int),
        'Children transferred out of CBP custody': transferred_out.astype(int),
        'Children in HHS Care': hhs_care.astype(int),
        'Children discharged from HHS Care': discharged.astype(int)
    })
    
    df.to_csv(output_file, index=False)
    print(f"Mock data successfully generated and saved to {output_file}")

if __name__ == "__main__":
    generate_mock_data()
