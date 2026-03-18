import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

DATASET_CONFIG = {
    "bupa_lr_lc.csv": {
        "target": "selector",
        "drop_cols": []},
    "Heart_disease_lr_mc.csv": {"target": "target", "drop_cols": []},
    "vehicle_lr_hc.csv": {"target": "class", "drop_cols": []},
    "contaceptive_mr_lc.csv": {"target": "Contraceptive_Method", "drop_cols": []},
    "Retinopathy_Debrecen_mr_hc.csv": {"target": "class", "drop_cols": []},
    "Wine_mr_mc.csv": {"target": "quality", "drop_cols": ["Id"]},
    "abalone_hr_lc.csv": {
        "target": "Rings", 
        "drop_cols": [],
        "encode_cols": ["Sex"] 
    },
    "EEG_Eye_State_hr_mc.csv": {"target": "eyeDetection", "drop_cols": []},
    "letter-recognition_hr_hc.csv": {"target": "letter", "drop_cols": []}
}

def process_all_datasets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, '../../datasets/balanced')
    geom_out_dir = os.path.join(script_dir, '../../datasets/balanced_normalised')
    prob_out_dir = os.path.join(script_dir, '../../datasets/balanced_distribution')
    
    os.makedirs(geom_out_dir, exist_ok=True)
    os.makedirs(prob_out_dir, exist_ok=True)
    
    print(f"Starting batch process for {len(DATASET_CONFIG)} datasets...\n")
    
    for filename, config in DATASET_CONFIG.items():
        print(f"--- Processing {filename} ---")
        input_filepath = os.path.join(input_dir, filename)
        
        if not os.path.exists(input_filepath):
            print(f"  [!] File not found: {input_filepath}. Skipping.\n")
            continue
            
        df = pd.read_csv(input_filepath)
        
        # Clean NaNs
        initial_rows = len(df)
        df = df.dropna()
        if len(df) < initial_rows:
            print(f"  Dropped {initial_rows - len(df)} rows containing NaNs.")
            
        # Drop dataset-specific garbage columns
        for col in config.get("drop_cols", []):
            if col in df.columns:
                df = df.drop(columns=[col])
                
        # Separate Features (X) and Target (y)
        target_col = config["target"]
        if target_col not in df.columns:
            print(f"  [!] Error: Target column '{target_col}' not found. Skipping.\n")
            continue
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # --- NEW: ONE-HOT ENCODING ---
        encode_cols = config.get("encode_cols", [])
        if encode_cols:
            # Convert categories to separate numerical columns (1.0 or 0.0)
            X = pd.get_dummies(X, columns=encode_cols, dtype=float)
            print(f"  Encoded columns: {encode_cols}. Total features now: {len(X.columns)}")
        # -----------------------------
        
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
    process_all_datasets()
    print("Batch processing complete!")