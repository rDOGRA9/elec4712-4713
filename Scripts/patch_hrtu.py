import os
import pandas as pd
import warnings

from distances import compute_distance_matrix
from supervised_pipeline import evaluate_supervised
from unsupervised_pipeline import evaluate_unsupervised

warnings.filterwarnings('ignore')

def patch_htru_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    master_csv_path = os.path.join(script_dir, '../datasets/MASTER_THESIS_RESULTS.csv')
    
    target_dataset_name = "HTRU_2_hr_lc.csv"
    category = "imbalanced"
    
    missing_metrics = [
        "earth_movers", "kl_divergence", "jensen_shannon", "bhattacharyya"
    ]
    
    # Loads your original, untouched preprocessed file
    htru_filepath = os.path.join(script_dir, '../datasets/output_imbalanced/slack_onehot/HTRU_2_hr_lc_Slack_onehot.csv')
    
    print(f"Loading {target_dataset_name}...")
    try:
        df = pd.read_csv(htru_filepath)
        target_col = "class" 
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # =======================================================
        # THE ISOLATED FIX: This only happens right here in memory
        # It forces the non-negative shift purely for this patch run.
        # =======================================================
        print("  Applying in-memory non-negative shift to fix probabilistic metrics...")
        X = X - X.min()
        
    except Exception as e:
        print(f"[!] Failed to load dataset: {e}")
        return

    new_results = []

    for metric in missing_metrics:
        print(f"  Running missing metric: {metric}...")
        try:
            dist_matrix = compute_distance_matrix(X, metric)
            
            sup_res = evaluate_supervised(dist_matrix, y, metric)
            for res in sup_res:
                res['Dataset'] = target_dataset_name
                res['Category'] = category
                res['Metric Type'] = 'Probabilistic'
                new_results.append(res)
                
            unsup_res = evaluate_unsupervised(X, dist_matrix, y, metric)
            for res in unsup_res:
                res['Dataset'] = target_dataset_name
                res['Category'] = category
                res['Metric Type'] = 'Probabilistic'
                new_results.append(res)
                
        except Exception as e:
            print(f"  [!] Failed on {metric}: {e}")

    if new_results:
        print("\nAppending new results to MASTER_THESIS_RESULTS.csv...")
        patch_df = pd.DataFrame(new_results)
        master_df = pd.read_csv(master_csv_path)
        updated_master_df = pd.concat([master_df, patch_df], ignore_index=True)
        updated_master_df.to_csv(master_csv_path, index=False)
        print(f"SUCCESS! Added {len(patch_df)} new rows to your master dataset.")
    else:
        print("\nNo new results were generated.")

if __name__ == "__main__":
    patch_htru_results()