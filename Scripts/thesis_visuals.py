import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def generate_thesis_visuals(input_csv="../datasets/MASTER_THESIS_RESULTS.csv", output_dir="thesis_exports"):
    """
    Generates Detailed Borda Count rankings and high-resolution Heatmaps.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    algo_metrics = {
        'KNN': 'F1 Score',
        'SVM': 'F1 Score',
        'K-Medoids': 'Silhouette Score',
        'DBSCAN': 'Silhouette Score'
    }

    for algo, metric in algo_metrics.items():
        print(f"\nProcessing {algo}...")
        
        algo_df = df[df['Algorithm'] == algo].dropna(subset=[metric])
        if algo_df.empty:
            print(f"  [!] No valid data found for {algo}. Skipping.")
            continue
            
        # 1. Pivot for Scores
        pivot_scores = algo_df.pivot(index='Distance Metric', columns='Dataset', values=metric)
        
        # 2. Pivot for Ranks (1 point for worst, N points for best)
        pivot_ranks = pivot_scores.rank(ascending=True)
        
        # 3. Calculate Total Borda
        borda_totals = pivot_ranks.sum(axis=1)
        
        # ==========================================
        # DETAILED BORDA TABLE GENERATION
        # ==========================================
        # Build a new DataFrame that interleaves Scores and Ranks
        detailed_df = pd.DataFrame(index=pivot_scores.index)
        
        for dataset in pivot_scores.columns:
            # Clean dataset name for column headers
            clean_name = dataset.replace('.csv', '')
            detailed_df[f"{clean_name} ({metric})"] = pivot_scores[dataset]
            detailed_df[f"{clean_name} (Rank)"] = pivot_ranks[dataset]
            
        # Add the final score at the very end
        detailed_df['FINAL BORDA SCORE'] = borda_totals
        
        # Sort the entire table by the Final Borda Score so the winner is at the top
        detailed_df = detailed_df.sort_values(by='FINAL BORDA SCORE', ascending=False)
        
        # Save Detailed CSV
        detailed_filepath = os.path.join(output_dir, f'{algo}_Detailed_Borda_Ranking.csv')
        detailed_df.to_csv(detailed_filepath)
        print(f"  [+] Saved Detailed Rankings: {detailed_filepath}")

        # ==========================================
        # HEATMAP GENERATION
        # ==========================================
        plt.figure(figsize=(16, 9))
        cmap = 'Blues' if algo in ['KNN', 'SVM'] else 'Oranges'
        
        # We plot the Heatmap based on the original SCORES, ordered by the winning ranks
        # Sort the pivot_scores by the borda totals so the heatmap matches the table!
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
        heatmap_filepath = os.path.join(output_dir, f'{algo}_Heatmap.png')
        plt.savefig(heatmap_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [+] Saved High-Res Heatmap: {heatmap_filepath}")

    print("\n=======================================================")
    print(f"All visuals and detailed tables successfully saved to: ./{output_dir}/")
    print("=======================================================")

if __name__ == "__main__":
    generate_thesis_visuals(input_csv="../datasets/MASTER_THESIS_RESULTS.csv")