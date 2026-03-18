import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# 1. THE IMBALANCED CONFIGURATION DICTIONARY (All 9 Datasets)
DATASET_CONFIG = {
    "climate_model_crashes_lr_hc.csv": {
        "target": "outcome",
        "drop_cols": ["Study", "Run"], 
        "encode_cols": []
    },
    "ecoli_lr_lc.csv": {
        "target": "class",
        "drop_cols": ["sequence_name"], 
        "encode_cols": []
    },
    "Indian_Liver_Patient_lr_mc.csv": {
        "target": "is_patient",
        "drop_cols": [],
        "encode_cols": ["gender"] 
    },
    "german_credit_data_mr_hc.csv": {
        "target": "kredit",
        "drop_cols": [],
        "encode_cols": []
    },
    "solar_flare_mr_mc.csv": {
        "target": "severe flares",
        "drop_cols": [],
        "encode_cols": ["modified Zurich class", "largest spot size", "spot distribution"] 
    },
    "yeast_mr_lc.csv": {
        "target": "name",
        "drop_cols": [], 
        "encode_cols": []
    },
    "HTRU_2_hr_lc.csv": {
        "target": "class", # Updated to use your new column header!
        "drop_cols": [],
        "encode_cols": []
    },
    "online_shoppers_intention_hr_hc.csv": {
        "target": "Revenue",
        "drop_cols": [],
        "encode_cols": ["Month", "VisitorType", "Weekend"] 
    },
    "page_blocks_classification_hr_mc.csv": {
        "target": "class",
        "drop_cols": [],
        "encode_cols": []
    }
}

def process_imbalanced_datasets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, '../../datasets/imbalanced')
    geom_out_dir = os.path.join(script_dir, '../../datasets/imbalanced_normalised')
    prob_out_dir = os.path.join(script_dir, '../../datasets/imbalanced_distributions')
    
    os.makedirs(geom_out_dir, exist_ok=True)
    os.makedirs(prob_out_dir, exist_ok=True)
    
    print(f"Starting batch process for {len(DATASET_CONFIG)} imbalanced datasets...\n")
    
    for filename, config in DATASET_CONFIG.items():
        print(f"--- Processing {filename} ---")
        input_filepath = os.path.join(input_dir, filename)
        
        if not os.path.exists(input_filepath):
            print(f"  [!] File not found: {input_filepath}. Skipping.\n")
            continue
            
        df = pd.read_csv(input_filepath)
        
        initial_rows = len(df)
        df = df.dropna()
        if len(df) < initial_rows:
            print(f"  Dropped {initial_rows - len(df)} rows containing NaNs.")
            
        for col in config.get("drop_cols", []):
            if col in df.columns:
                df = df.drop(columns=[col])
                
        target_col = config["target"]
        if target_col not in df.columns:
            print(f"  [!] Error: Target column '{target_col}' not found in {filename}. Skipping.\n")
            continue
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # ONE-HOT ENCODING
        encode_cols = config.get("encode_cols", [])
        if encode_cols:
            X = pd.get_dummies(X, columns=encode_cols, dtype=float)
            print(f"  Encoded columns: {encode_cols}.")
        
        # GEOMETRIC DISTANCE PREPROCESSING
        scaler = MinMaxScaler()
        X_geom = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        df_geom = pd.concat([X_geom, y], axis=1)
        
        geom_filename = filename.replace('.csv', '_Geometric.csv')
        geom_filepath = os.path.join(geom_out_dir, geom_filename)
        df_geom.to_csv(geom_filepath, index=False)
        
        # PROBABILISTIC DISTANCE PREPROCESSING
        epsilon = 1e-9
        X_prob = X_geom + epsilon
        X_prob = X_prob.div(X_prob.sum(axis=1), axis=0)
        df_prob = pd.concat([X_prob, y], axis=1)
        
        prob_filename = filename.replace('.csv', '_Probabilistic.csv')
        prob_filepath = os.path.join(prob_out_dir, prob_filename)
        df_prob.to_csv(prob_filepath, index=False)
        
        print(f"  Saved: {geom_filename} & {prob_filename}\n")

if __name__ == "__main__":
    process_imbalanced_datasets()
    print("Batch processing complete! All 18 datasets for the experimental matrix are now preprocessed.")