import os
import pandas as pd
import warnings
# Import your configuration so it automatically knows which datasets to process
from preprocessing import MASTER_CONFIG

warnings.filterwarnings('ignore')

def generate_dataset_statistics():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_records = []
    
    print("\n=======================================================")
    print("CALCULATING DATASET STATISTICS (Mean, Std, Skew, Kurtosis)")
    print("=======================================================\n")
    
    for category in ["balanced", "imbalanced"]:
        datasets = MASTER_CONFIG.get(category, {})
        input_dir = os.path.join(script_dir, f'../datasets/{category}')
        
        for filename, config in datasets.items():
            input_filepath = os.path.join(input_dir, filename)
            
            if not os.path.exists(input_filepath):
                print(f"  [!] Missing file: {filename}. Skipping.")
                continue
                
            print(f"--- Processing {filename} ---")
            
            # Load the raw dataset
            df = pd.read_csv(input_filepath).dropna()
            
            # Drop the target column and any garbage columns before calculating stats
            cols_to_drop = config.get("drop_cols", [])
            target_col = config.get("target")
            if target_col:
                cols_to_drop.append(target_col)
                
            features_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
            
            # Keep only numerical columns (we cannot calculate skewness on text/categories)
            num_df = features_df.select_dtypes(include=['number'])
            
            # Calculate the 4 statistical moments for each feature
            for feature in num_df.columns:
                stats_records.append({
                    'Dataset': filename,
                    'Category': category,
                    'Feature': feature,
                    'Mean': num_df[feature].mean(),
                    'Std Deviation': num_df[feature].std(),
                    'Skewness': num_df[feature].skew(),
                    'Kurtosis': num_df[feature].kurtosis()
                })
                
    # Export the results to CSVs
    if stats_records:
        # 1. Save the original detailed feature-by-feature report
        stats_df = pd.DataFrame(stats_records)
        output_path = os.path.join(script_dir, '../datasets/SUPERVISOR_STATS_REPORT.csv')
        stats_df.to_csv(output_path, index=False)
        
        # 2. NEW CODE: Group by dataset and calculate averages
        avg_stats_df = stats_df.groupby(['Dataset', 'Category']).agg({
            'Mean': 'mean',
            'Std Deviation': 'mean',
            'Skewness': 'mean',
            'Kurtosis': 'mean'
        }).reset_index()
        
        # Rename the columns to match your requested format
        avg_stats_df = avg_stats_df.rename(columns={
            'Mean': 'average_mean',
            'Std Deviation': 'average_deviation',
            'Skewness': 'avg_skewness',
            'Kurtosis': 'avg_kurtosis'
        })
        
        # Save the averaged data to a new CSV file
        avg_output_path = os.path.join(script_dir, '../datasets/AVERAGE_DATASET_METRICS.csv')
        avg_stats_df.to_csv(avg_output_path, index=False)
        
        print(f"\n=======================================================")
        print(f"SUCCESS! Detailed statistics report saved to: {output_path}")
        print(f"SUCCESS! Averaged metrics report saved to: {avg_output_path}")
        print("You can hand these CSVs directly to your supervisor.")
        print(f"=======================================================")

if __name__ == "__main__":
    generate_dataset_statistics()