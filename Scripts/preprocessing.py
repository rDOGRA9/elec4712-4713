import os
import numpy as np
import pandas as pd

# =====================================================================
# 1. UNIFIED CONFIGURATION DICTIONARY
# =====================================================================
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

# =====================================================================
# 2. CORE MATH & PREPROCESSING FUNCTIONS
# =====================================================================
def encode_target(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Converts categorical target columns into integers."""
    unique_vals = df[label_col].dropna().unique()
    mapping = {val: i for i, val in enumerate(unique_vals)}
    df[label_col] = df[label_col].map(mapping)
    df[label_col] = df[label_col].astype(int)
    return df

def preprocess_for_pairwise(df: pd.DataFrame, label_col: str, encoding_method: str = "onehot") -> tuple:
    """Encodes features, normalizes, and appends the slack variable 'Adjusted_p'."""
    df = df.copy()

    target_series = None
    if label_col and label_col in df.columns:
        if not np.issubdtype(df[label_col].dtype, np.number):
            df = encode_target(df, label_col)
        target_series = df[label_col].copy()
        df.drop(columns=[label_col], inplace=True)
    
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if encoding_method.lower() == "onehot":
        cat_encoded = [pd.get_dummies(df[col], prefix=col, dtype=float) for col in cat_cols]
        cat_part = pd.concat(cat_encoded, axis=1) if cat_encoded else pd.DataFrame(index=df.index)
    elif encoding_method.lower() == "target":
        cat_part = pd.DataFrame(index=df.index)
        for col in cat_cols:
            means = target_series.groupby(df[col]).mean()
            cat_part[col + "_target"] = df[col].map(means)
    else: 
        cat_part = pd.DataFrame(index=df.index)
        for col in cat_cols:
            uniques = sorted(df[col].unique())
            mapping_dict = {cat_val: i + 1 for i, cat_val in enumerate(uniques)}
            cat_part[col + "_ord"] = df[col].map(mapping_dict)
    
    num_part = df[num_cols].copy()
    df_processed = pd.concat([num_part, cat_part], axis=1)

    df_processed.dropna(inplace=True)
    target_series = target_series.loc[df_processed.index]
    
    col_max = df_processed.max().replace(0, 1) 
    num_columns = df_processed.shape[1]
    df_normalized = df_processed.div(col_max * num_columns, axis=1)
    
    row_sums = df_normalized.sum(axis=1)
    df_normalized['Adjusted_p'] = 1 - row_sums
    
    return target_series, df_normalized

# =====================================================================
# 3. BATCH PROCESSING LOOP
# =====================================================================
def process_datasets(dataset_category="balanced", encoding_method="onehot"):
    """Processes datasets and saves them to the output folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, f'../datasets/{dataset_category}')
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
        
        initial_rows = len(df)
        df = df.dropna()
        if len(df) < initial_rows:
            print(f"  Dropped {initial_rows - len(df)} rows containing NaNs.")
            
        for col in config.get("drop_cols", []):
            if col in df.columns:
                df = df.drop(columns=[col])
                
        target_col = config["target"]
        if target_col not in df.columns:
            print(f"  [!] Error: Target column '{target_col}' not found. Skipping.\n")
            continue
            
        try:
            y, X_processed = preprocess_for_pairwise(df, target_col, encoding_method)
            final_df = pd.concat([X_processed, y], axis=1)
            
            out_filename = filename.replace('.csv', f'_Slack_{encoding_method}.csv')
            out_filepath = os.path.join(out_dir, out_filename)
            final_df.to_csv(out_filepath, index=False)
            
            print(f"  Total features output: {len(X_processed.columns)}")
            print(f"  Saved: {out_filename}\n")
            
        except Exception as e:
            print(f"  [!] Failed to process {filename}: {e}\n")

if __name__ == "__main__":
    process_datasets(dataset_category="balanced", encoding_method="onehot")
    process_datasets(dataset_category="imbalanced", encoding_method="onehot")
    print("All 18 datasets processed successfully!")