import pandas as pd
import os
from preprocessing import preprocess_for_pairwise

# UNIFIED CONFIGURATION DICTIONARY
MASTER_CONFIG = {
    "balanced": {
        "bupa_lr_lc.csv": {"target": "selector", "drop_cols": []},
        "Heart_disease_lr_mc.csv": {"target": "target", "drop_cols": []},
        "vehicle_lr_hc.csv": {"target": "class", "drop_cols": []},
        "contaceptive_mr_lc.csv": {"target": "Contraceptive_Method", "drop_cols": []},
        "Retinopathy_Debrecen_mr_hc.csv": {"target": "class", "drop_cols": []},
        "Wine_mr_mc.csv": {"target": "quality", "drop_cols": ["Id"]},
        "abalone_hr_lc.csv": {"target": "Rings", "drop_cols": []},
        "EEG_Eye_State_hr_mc.csv": {"target": "eyeDetection", "drop_cols": []},
        "letter-recognition_hr_hc.csv": {"target": "letter", "drop_cols": []}
    },
    "imbalanced": {
        "climate_model_crashes_lr_hc.csv": {"target": "outcome", "drop_cols": ["Study", "Run"]},
        "ecoli_lr_lc.csv": {"target": "class", "drop_cols": ["sequence_name"]},
        "Indian_Liver_Patient_lr_mc.csv": {"target": "is_patient", "drop_cols": []},
        "german_credit_data_mr_hc.csv": {"target": "kredit", "drop_cols": []},
        "solar_flare_mr_mc.csv": {"target": "severe flares", "drop_cols": []},
        "yeast_mr_lc.csv": {"target": "name", "drop_cols": []},
        "HTRU_2_hr_lc.csv": {"target": "class", "drop_cols": []},
        "online_shoppers_intention_hr_hc.csv": {"target": "Revenue", "drop_cols": []},
        "page_blocks_classification_hr_mc.csv": {"target": "class", "drop_cols": []}
    }
}

def process_datasets(dataset_category="balanced", encoding_method="onehot"):
    """
    Processes a specific category of datasets (e.g., 'balanced' or 'imbalanced')
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dynamically point to ../datasets/balanced OR ../datasets/imbalanced
    input_dir = os.path.join(script_dir, f'../datasets/{dataset_category}')
    
    # Create an output directory like ../datasets/output_balanced/slack_onehot
    out_dir = os.path.join(script_dir, f'../datasets/output_{dataset_category}/slack_{encoding_method}')
    os.makedirs(out_dir, exist_ok=True)
    
    datasets = MASTER_CONFIG.get(dataset_category, {})
    print(f"\n=======================================================")
    print(f"Starting batch process for {len(datasets)} {dataset_category.upper()} datasets...")
    print(f"Encoding method: '{encoding_method}'")
    print(f"=======================================================\n")
    
    for filename, config in datasets.items():
        print(f"--- Processing {filename} ---")
        input_filepath = os.path.join(input_dir, filename)
        
        if not os.path.exists(input_filepath):
            print(f"  [!] File not found: {input_filepath}. Skipping.\n")
            continue
            
        df = pd.read_csv(input_filepath)
        
        # Clean NaNs initially
        initial_rows = len(df)
        df = df.dropna()
        if len(df) < initial_rows:
            print(f"  Dropped {initial_rows - len(df)} rows containing NaNs.")
            
        # Drop specific garbage columns defined in config
        for col in config.get("drop_cols", []):
            if col in df.columns:
                df = df.drop(columns=[col])
                
        target_col = config["target"]
        if target_col not in df.columns:
            print(f"  [!] Error: Target column '{target_col}' not found. Skipping.\n")
            continue
            
        try:
            # Apply the shared slack variable methodology
            y, X_processed = preprocess_for_pairwise(
                df=df, 
                label_col=target_col, 
                encoding_method=encoding_method
            )
            
            # Recombine features and target
            final_df = pd.concat([X_processed, y], axis=1)
            
            # Export
            out_filename = filename.replace('.csv', f'_Slack_{encoding_method}.csv')
            out_filepath = os.path.join(out_dir, out_filename)
            final_df.to_csv(out_filepath, index=False)
            
            print(f"  Total features output: {len(X_processed.columns)}")
            print(f"  Saved: {out_filename}\n")
            
        except Exception as e:
            print(f"  [!] Failed to process {filename}: {e}\n")

if __name__ == "__main__":
    # You can process both folders back-to-back automatically!
    process_datasets(dataset_category="balanced", encoding_method="onehot")
    process_datasets(dataset_category="imbalanced", encoding_method="onehot")
    
    print("All 18 datasets processed successfully!")