import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def process_vehicle_dataset(filename):
    print(f"--- Processing {filename} ---")
    
    # 1. SET UP RELATIVE DIRECTORY PATHS
    # Get the directory where this script is located (balanced_scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up two levels ('../../') to reach the root, then into the specific dataset folders
    input_dir = os.path.join(script_dir, '../../datasets/balanced')
    geom_out_dir = os.path.join(script_dir, '../../datasets/balanced_normalised')
    prob_out_dir = os.path.join(script_dir, '../../datasets/balanced_distribution')
    
    # Create the output directories if they don't already exist
    os.makedirs(geom_out_dir, exist_ok=True)
    os.makedirs(prob_out_dir, exist_ok=True)
    
    # Build the full file path for the input CSV
    input_filepath = os.path.join(input_dir, filename)
    
    # 2. LOAD DATA
    df = pd.read_csv(input_filepath)
    
    # 3. CLEANING (Drop NaNs)
    initial_rows = len(df)
    df = df.dropna()
    print(f"Cleaned data: {len(df)} rows kept out of {initial_rows}.")
    
    # 4. SEPARATE FEATURES (X) AND TARGET (y)
    target_col = 'class'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 5. GEOMETRIC DISTANCE PREPROCESSING
    scaler = MinMaxScaler()
    X_geom = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    df_geom = pd.concat([X_geom, y], axis=1)
    
    # Save Geometric to 'balanced_normalised'
    geom_filename = filename.replace('.csv', '_Geometric.csv')
    geom_filepath = os.path.join(geom_out_dir, geom_filename)
    df_geom.to_csv(geom_filepath, index=False)
    print(f"Saved Geometric Dataset: {geom_filepath}")
    
    # 6. PROBABILISTIC DISTANCE PREPROCESSING
    epsilon = 1e-9
    X_prob = X_geom + epsilon
    X_prob = X_prob.div(X_prob.sum(axis=1), axis=0)
    df_prob = pd.concat([X_prob, y], axis=1)
    
    # Save Probabilistic to 'balanced_distribution'
    prob_filename = filename.replace('.csv', '_Probabilistic.csv')
    prob_filepath = os.path.join(prob_out_dir, prob_filename)
    df_prob.to_csv(prob_filepath, index=False)
    print(f"Saved Probabilistic Dataset: {prob_filepath}\n")

if __name__ == "__main__":
    # Now you just pass the filename, and the script handles the routing!
    process_vehicle_dataset('vehicle_lr_hc.csv')