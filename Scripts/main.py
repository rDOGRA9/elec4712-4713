import os
import pandas as pd
import warnings

# Import your pipelines and configurations
from preprocessing import MASTER_CONFIG
from distances import compute_distance_matrix
from supervised_pipeline import evaluate_supervised
from unsupervised_pipeline import evaluate_unsupervised

# Suppress minor warnings for cleaner console output
warnings.filterwarnings('ignore')

# Define which metrics go to which datasets
GEOMETRIC_METRICS = [
    "euclidean", "manhattan", "minkowski", "chebyshev", 
    "canberra", "braycurtis", "hamming", "cosine"
]

PROBABILISTIC_METRICS = [
    "earth_movers", "kl_divergence", "jensen_shannon", 
    "bhattacharyya", "total_variation"
]

def load_and_split(filepath, config):
    """Loads a CSV, drops garbage columns, and splits into X and y."""
    if not os.path.exists(filepath):
        return None, None
        
    df = pd.read_csv(filepath).dropna()
    
    # Drop config-defined garbage columns if they exist
    for col in config.get("drop_cols", []):
        if col in df.columns:
            df = df.drop(columns=[col])
            
    target_col = config["target"]
    if target_col not in df.columns:
        return None, None
        
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    return X, y

def run_experiments():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_results = []
    
    print("\n=======================================================")
    print("STARTING FULL PIPELINE EXECUTION")
    print("=======================================================\n")

    for category in ["balanced", "imbalanced"]:
        datasets = MASTER_CONFIG.get(category, {})
        
        # Define base directory paths
        standard_dir = os.path.join(script_dir, f'../datasets/{category}_normalised')
        slack_dir = os.path.join(script_dir, f'../datasets/output_{category}/slack_onehot')
        
        for base_filename, config in datasets.items():
            print(f"--- Processing {base_filename} ({category.upper()}) ---")
            
            # 1. Paths (UPDATED TO MATCH YOUR FILE NAMES EXACTLY)
            standard_filename = base_filename.replace('.csv', '_Geometric.csv')
            standard_path = os.path.join(standard_dir, standard_filename)
            
            slack_filename = base_filename.replace('.csv', '_Slack_onehot.csv')
            slack_path = os.path.join(slack_dir, slack_filename)
            
            # 2. Load Data
            X_std, y_std = load_and_split(standard_path, config)
            X_slk, y_slk = load_and_split(slack_path, config)
            
            if X_std is None or X_slk is None:
                print(f"  [!] Missing files for {base_filename}. Skipping.")
                # Optional: Print exactly what it couldn't find to help debug
                if X_std is None: print(f"      -> Could not find: {standard_path}")
                if X_slk is None: print(f"      -> Could not find: {slack_path}")
                continue

            # =========================================================
            # PART A: GEOMETRIC METRICS (on Standard Normalised Data)
            # =========================================================
            for metric in GEOMETRIC_METRICS:
                try:
                    dist_matrix = compute_distance_matrix(X_std, metric)
                    
                    # Supervised
                    sup_res = evaluate_supervised(dist_matrix, y_std, metric)
                    # Unsupervised (pass X_std for DB/CH score calculations)
                    unsup_res = evaluate_unsupervised(X_std, dist_matrix, y_std, metric)
                    
                    # Tag results with dataset info and merge
                    for res in sup_res + unsup_res:
                        res['Dataset'] = base_filename
                        res['Category'] = category
                        res['Metric Type'] = 'Geometric'
                        all_results.append(res)
                        
                except Exception as e:
                    print(f"  [!] Failed Geometric '{metric}': {e}")

            # =========================================================
            # PART B: PROBABILISTIC METRICS (on Slack Normalised Data)
            # =========================================================
            for metric in PROBABILISTIC_METRICS:
                try:
                    dist_matrix = compute_distance_matrix(X_slk, metric)
                    
                    # Supervised
                    sup_res = evaluate_supervised(dist_matrix, y_slk, metric)
                    # Unsupervised (pass X_slk for DB/CH score calculations)
                    unsup_res = evaluate_unsupervised(X_slk, dist_matrix, y_slk, metric)
                    
                    # Tag results with dataset info and merge
                    for res in sup_res + unsup_res:
                        res['Dataset'] = base_filename
                        res['Category'] = category
                        res['Metric Type'] = 'Probabilistic'
                        all_results.append(res)
                        
                except Exception as e:
                    print(f"  [!] Failed Probabilistic '{metric}': {e}")
                    
            print(f"  [+] Completed {base_filename}\n")

    # =========================================================
    # EXPORT RESULTS
    # =========================================================
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns so Dataset and Algorithm info are first
        cols = ['Dataset', 'Category', 'Algorithm', 'Metric Type', 'Distance Metric']
        remaining_cols = [c for c in results_df.columns if c not in cols]
        results_df = results_df[cols + remaining_cols]
        
        output_path = os.path.join(script_dir, '../datasets/MASTER_THESIS_RESULTS.csv')
        results_df.to_csv(output_path, index=False)
        print(f"=======================================================")
        print(f"SUCCESS! Master results saved to: {output_path}")
        print(f"Total experiment configurations run: {len(results_df)}")
        print("=======================================================")
    else:
        print("No results were generated. Check your file paths.")

if __name__ == "__main__":
    run_experiments()