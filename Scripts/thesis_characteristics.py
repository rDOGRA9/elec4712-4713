import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner execution
warnings.filterwarnings('ignore')

def enrich_dataset_features(df):
    """
    Parses the 'Dataset' column (e.g., 'vehicle_lr_hc.csv') 
    using regex to safely extract Dataset Size and Feature Count attributes.
    """
    def extract_tags(filename):
        # Look for _lr_, _mr_, _hr_ (or before a dot)
        size_match = re.search(r'_(lr|mr|hr)(_|\.)', filename)
        # Look for _lc_, _mc_, _hc_ (or before a dot)
        feat_match = re.search(r'_(lc|mc|hc)(_|\.)', filename)
        
        size_tag = size_match.group(1) if size_match else 'unknown'
        feat_tag = feat_match.group(1) if feat_match else 'unknown'
        
        return size_tag, feat_tag

    # Apply extraction
    df[['Size_Tag', 'Feat_Tag']] = df['Dataset'].apply(lambda x: pd.Series(extract_tags(x)))
    
    # Map to readable labels
    size_mapping = {'lr': 'Low Rows', 'mr': 'Medium Rows', 'hr': 'High Rows'}
    feat_mapping = {'lc': 'Low Columns', 'mc': 'Medium Columns', 'hc': 'High Columns'}
    
    df['Dataset Size'] = df['Size_Tag'].map(size_mapping).fillna('Unknown')
    df['Feature Count'] = df['Feat_Tag'].map(feat_mapping).fillna('Unknown')
    
    # Drop the temporary tags
    df = df.drop(columns=['Size_Tag', 'Feat_Tag'])
    return df

def collect_conditional_rankings(df, algo, metric, all_rankings_list, is_lower_better=False):
    """
    Calculates top 3 distance metrics for every specific data constraint
    and appends them to a master list for CSV export.
    """
    algo_df = df[df['Algorithm'] == algo].dropna(subset=[metric])
    if algo_df.empty:
        return
        
    conditions = {
        "Balanced (Low Imbalance)": algo_df[algo_df['Category'] == 'balanced'],
        "Imbalanced (High Imbalance)": algo_df[algo_df['Category'] == 'imbalanced'],
        
        "Size: Low Rows": algo_df[algo_df['Dataset Size'] == 'Low Rows'],
        "Size: Medium Rows": algo_df[algo_df['Dataset Size'] == 'Medium Rows'],
        "Size: High Rows": algo_df[algo_df['Dataset Size'] == 'High Rows'],
        
        "Features: Low Columns": algo_df[algo_df['Feature Count'] == 'Low Columns'],
        "Features: Medium Columns": algo_df[algo_df['Feature Count'] == 'Medium Columns'],
        "Features: High Columns": algo_df[algo_df['Feature Count'] == 'High Columns']
    }
    
    for condition_name, subset in conditions.items():
        if subset.empty: 
            continue
        
        # If the metric is Davies-Bouldin, lower scores are better (ascending=True)
        # For everything else, higher is better (ascending=False)
        avg_scores = subset.groupby('Distance Metric')[metric].mean().sort_values(ascending=is_lower_better)
        
        # Save top 3 distances to our export list
        for rank, (dist, score) in enumerate(avg_scores.head(3).items(), 1):
            all_rankings_list.append({
                'Algorithm': algo,
                'Evaluation Metric': metric,
                'Dataset Condition': condition_name,
                'Rank': rank,
                'Distance Metric': dist,
                'Average Score': round(score, 4)
            })

def generate_characteristic_heatmap(df, algo, metric, output_dir, is_lower_better=False):
    """
    Plots Distance Metrics (Y-axis) against aggregated Data Characteristics (X-axis).
    """
    algo_df = df[df['Algorithm'] == algo].dropna(subset=[metric])
    if algo_df.empty:
        return
        
    characteristics = ['Category', 'Dataset Size', 'Feature Count']
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True)
    fig.suptitle(f'{algo} ({metric}) Performance by Data Characteristics', fontsize=18, fontweight='bold')
    
    # Reverse colormap for lower_is_better metrics so the "best" scores are still visually distinct
    cmap = 'viridis_r' if is_lower_better else 'viridis'
    
    for i, char in enumerate(characteristics):
        pivot = algo_df.pivot_table(
            index='Distance Metric', 
            columns=char, 
            values=metric, 
            aggfunc='mean'
        )
        
        if pivot.empty:
            continue
            
        sns.heatmap(
            pivot, 
            annot=True, 
            cmap=cmap, 
            fmt=".3f", 
            linewidths=.5, 
            ax=axes[i],
            cbar=(i == 2) # Only show colorbar on the last plot
        )
        
        axes[i].set_title(f'By {char}', fontsize=14, pad=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Distance Metric' if i == 0 else '')
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    clean_metric_name = metric.replace(' ', '_').replace('-', '_')
    out_path = os.path.join(output_dir, f'{algo}_{clean_metric_name}_Characteristics_Heatmap.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [+] Saved High-Res Heatmap: {algo}_{clean_metric_name}_Characteristics_Heatmap.png")

def run_supervisor_analysis():
    # Robust path handling to ensure it works in IDEs, Terminals, AND Jupyter Notebooks
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        
    input_csv = os.path.join(script_dir, "../datasets/MASTER_THESIS_RESULTS.csv")
    output_dir = os.path.join(script_dir, "../thesis_exports")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading datasets from: {input_csv}")
    if not os.path.exists(input_csv):
        print(f"[!] Error: Cannot find master dataset at {input_csv}")
        return

    df = pd.read_csv(input_csv)
    df = enrich_dataset_features(df)
    
    # Define all metrics to iterate over for each algorithm type
    algo_metrics = {
        'KNN': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'SVM': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'K-Medoids': ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Adjusted Rand Index', 'Normalized Mutual Info'],
        'DBSCAN': ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Adjusted Rand Index', 'Normalized Mutual Info']
    }
    
    # Specify metrics where a LOWER score is considered better
    lower_is_better_metrics = ['Davies-Bouldin Score']
    
    # We will collect all text rankings here instead of printing them
    all_rankings_data = []
    
    print("\nStarting generation of Characteristic Analysis...")
    for algo, metrics_list in algo_metrics.items():
        print(f"\nProcessing {algo}...")
        
        for metric in metrics_list:
            if metric not in df.columns:
                print(f"  [!] Metric '{metric}' not found in dataset. Skipping.")
                continue
                
            print(f"  -> Metric: {metric}")
            is_lower_better = metric in lower_is_better_metrics
            
            # 1. Collect rankings for CSV
            collect_conditional_rankings(df, algo, metric, all_rankings_data, is_lower_better)
            # 2. Generate and save heatmap images
            generate_characteristic_heatmap(df, algo, metric, output_dir, is_lower_better)

    # 3. Export all aggregated rankings to a single CSV
    if all_rankings_data:
        rankings_df = pd.DataFrame(all_rankings_data)
        
        # Sort so the CSV is beautifully organized by Algorithm -> Evaluation Metric -> Condition -> Rank
        rankings_df = rankings_df.sort_values(by=['Algorithm', 'Evaluation Metric', 'Dataset Condition', 'Rank'])
        
        csv_output_path = os.path.join(output_dir, "Conditional_Distance_Rankings.csv")
        rankings_df.to_csv(csv_output_path, index=False)
        
        print("\n=======================================================")
        print(f"SUCCESS! All analysis saved to: {os.path.abspath(output_dir)}")
        print(f"  - Generated CSV Data: Conditional_Distance_Rankings.csv")
        print(f"  - Generated Visuals: Heatmap PNGs for all metrics")
        print("=======================================================")

if __name__ == "__main__":
    run_supervisor_analysis()