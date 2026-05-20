import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def generate_thesis_visuals(input_csv="../datasets/MASTER_THESIS_RESULTS.csv", output_dir="thesis_visuals"):
    """
    Generates Detailed Borda Count rankings and high-resolution Heatmaps for all metrics.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # All metrics mapped to their respective algorithms
    algo_metrics = {
        'KNN': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'SVM': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'K-Medoids': ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Adjusted Rand Index', 'Normalized Mutual Info'],
        'DBSCAN': ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Adjusted Rand Index', 'Normalized Mutual Info']
    }

    # Specify metrics where a LOWER score is considered better for Borda ranking
    lower_is_better = ['Davies-Bouldin Score']

    for algo, metrics_list in algo_metrics.items():
        print(f"\nProcessing {algo}...")
        
        for metric in metrics_list:
            if metric not in df.columns:
                print(f"  [!] Metric '{metric}' not found in the dataset. Skipping.")
                continue

            print(f"  -> Metric: {metric}")
            
            # Filter for the algorithm and drop rows where the metric wasn't recorded (e.g., supervised metrics for unsupervised models)
            algo_df = df[df['Algorithm'] == algo].dropna(subset=[metric])
            if algo_df.empty:
                print(f"     [!] No valid data found for {algo} using {metric}. Skipping.")
                continue
                
            # 1. Pivot for Scores (Using pivot_table to safely handle any duplicate runs in the datasets)
            # 1. Pivot for Scores (Using pivot_table to safely handle any duplicate runs in the datasets)
            pivot_scores = algo_df.pivot_table(index='Distance Metric', columns='Dataset', values=metric, aggfunc='mean')
            
            # 2. Handle NaNs by assigning them the worst possible value for ranking purposes
            if metric in lower_is_better:
                # If lower is better (like Davies-Bouldin), NaNs get infinity (the worst possible score)
                filled_scores = pivot_scores.fillna(np.inf)
            else:
                # If higher is better (like Silhouette, F1), NaNs get -infinity (the worst possible score)
                filled_scores = pivot_scores.fillna(-np.inf)
            
            # 3. Pivot for Ranks (1 point for worst, N points for best)
            is_ascending = metric not in lower_is_better
            
            # Use 'min' method so that all failures (NaNs) tie for the lowest rank 
            # rather than getting arbitrary different ranks at the bottom.
            pivot_ranks = filled_scores.rank(ascending=is_ascending, method='min')
            
            # 4. Calculate Total Borda
            borda_totals = pivot_ranks.sum(axis=1)
            
            # ==========================================
            # DETAILED BORDA TABLE GENERATION
            # ==========================================
            detailed_df = pd.DataFrame(index=pivot_scores.index)
            
            for dataset in pivot_scores.columns:
                clean_name = dataset.replace('.csv', '')
                detailed_df[f"{clean_name} ({metric})"] = pivot_scores[dataset]
                detailed_df[f"{clean_name} (Rank)"] = pivot_ranks[dataset]
                
            detailed_df['FINAL BORDA SCORE'] = borda_totals
            detailed_df = detailed_df.sort_values(by='FINAL BORDA SCORE', ascending=False)
            
            clean_metric_name = metric.replace(' ', '_').replace('-', '_')
            
            detailed_filepath = os.path.join(output_dir, f'{algo}_{clean_metric_name}_Detailed_Borda_Ranking.csv')
            detailed_df.to_csv(detailed_filepath)
            print(f"     [+] Saved Detailed Rankings: {detailed_filepath}")

            # ==========================================
            # HEATMAP GENERATION
            # ==========================================
            # Adjusted width slightly to comfortably fit 18-20 dataset names
            plt.figure(figsize=(18, 10)) 
            cmap = 'Blues' if algo in ['KNN', 'SVM'] else 'Oranges'
            
            pivot_scores_sorted = pivot_scores.loc[detailed_df.index]
            
            ax = sns.heatmap(
                pivot_scores_sorted, 
                annot=True,          
                cmap=cmap,           
                fmt=".3f",           
                linewidths=.5,       
                cbar_kws={'label': metric}
            )
            
            plt.title(f'Performance of Distance Metrics on {algo} ({metric})', fontsize=18, pad=20, fontweight='bold')
            plt.xlabel('Dataset', fontsize=14, fontweight='bold', labelpad=10)
            plt.ylabel('Distance Metric', fontsize=14, fontweight='bold', labelpad=10)
            
            clean_dataset_names = [name.get_text().replace('.csv', '') for name in ax.get_xticklabels()]
            ax.set_xticklabels(clean_dataset_names, rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=11)
            
            plt.tight_layout()
            heatmap_filepath = os.path.join(output_dir, f'{algo}_{clean_metric_name}_Heatmap.png')
            plt.savefig(heatmap_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"     [+] Saved High-Res Heatmap: {heatmap_filepath}")

    print("\n=======================================================")
    print(f"All visuals and detailed tables successfully saved to: ./{output_dir}/")
    print("=======================================================")

if __name__ == "__main__":
    generate_thesis_visuals(input_csv="../datasets/MASTER_THESIS_RESULTS.csv")