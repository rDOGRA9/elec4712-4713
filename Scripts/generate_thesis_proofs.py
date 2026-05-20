import pandas as pd
import glob
import os

# 1. Configure File Path
# Since this script runs from 'Scripts', it looks into the 'thesis_visuals' subfolder
data_path = os.path.join("thesis_visuals", "*_Detailed_Borda_Ranking.csv")
files = glob.glob(data_path)

if not files:
    print(f"Error: No files found in {data_path}. Please check your folder structure.")
    exit()

# 2. Define Dataset Topologies (Now including Mediums)
balanced = [
    "bupa_lr_lc", "Heart_disease_lr_mc", "vehicle_lr_hc", "contaceptive_mr_lc", 
    "Retinopathy_Debrecen_mr_hc", "Wine_mr_mc", "abalone_hr_lc", 
    "EEG_Eye_State_hr_mc", "letter-recognition_hr_hc"
]
imbalanced = [
    "climate_model_crashes_lr_hc", "ecoli_lr_lc", "Indian_Liver_Patient_lr_mc", 
    "german_credit_data_mr_hc", "solar_flare_mr_mc", "yeast_mr_lc", 
    "HTRU_2_hr_lc", "online_shoppers_intention_hr_hc", "page_blocks_classification_hr_mc"
]

all_datasets = balanced + imbalanced

# Extracting all 3 tiers for Rows
lr_datasets = [d for d in all_datasets if "_lr_" in d]
mr_datasets = [d for d in all_datasets if "_mr_" in d]
hr_datasets = [d for d in all_datasets if "_hr_" in d]

# Extracting all 3 tiers for Columns
lc_datasets = [d for d in all_datasets if "_lc" in d]
mc_datasets = [d for d in all_datasets if "_mc" in d]
hc_datasets = [d for d in all_datasets if "_hc" in d]

# 3. Process Files and Calculate Borda Points
all_results = []

for f in files:
    # Determine the algorithm paradigm
    filename = os.path.basename(f)
    if filename.startswith("DBSCAN") or filename.startswith("K-Medoids"):
        paradigm = "Unsupervised"
        algo = filename.split("_")[0]
    else:
        paradigm = "Supervised"
        algo = filename.split("_")[0]
        
    df = pd.read_csv(f)
    rank_cols = [c for c in df.columns if "(Rank)" in c]
    
    def sum_scores(ds_list):
        cols = [c for c in rank_cols if c.split(" (Rank)")[0] in ds_list]
        return df[cols].sum(axis=1) if cols else 0

    df['Balanced_Score'] = sum_scores(balanced)
    df['Imbalanced_Score'] = sum_scores(imbalanced)
    
    df['LR_Score'] = sum_scores(lr_datasets)
    df['MR_Score'] = sum_scores(mr_datasets)
    df['HR_Score'] = sum_scores(hr_datasets)
    
    df['LC_Score'] = sum_scores(lc_datasets)
    df['MC_Score'] = sum_scores(mc_datasets)
    df['HC_Score'] = sum_scores(hc_datasets)
    
    df['Paradigm'] = paradigm
    df['Algorithm'] = algo
    
    cols_to_keep = ['Distance Metric', 'Paradigm', 'Algorithm', 'FINAL BORDA SCORE', 
                    'Balanced_Score', 'Imbalanced_Score', 
                    'LR_Score', 'MR_Score', 'HR_Score', 
                    'LC_Score', 'MC_Score', 'HC_Score']
    all_results.append(df[cols_to_keep])

master_df = pd.concat(all_results, ignore_index=True)

# Separate into paradigms
unsup_df = master_df[master_df['Paradigm'] == 'Unsupervised'].groupby('Distance Metric').sum().reset_index()
sup_df = master_df[master_df['Paradigm'] == 'Supervised'].groupby('Distance Metric').sum().reset_index()

# Utility to print nice tables
def print_top(df, sort_col, title, top_n=3):
    print(f"\n{title}")
    print("-" * len(title))
    sorted_df = df.sort_values(sort_col, ascending=False)[['Distance Metric', sort_col]].head(top_n)
    print(sorted_df.to_string(index=False))

print("=========================================================")
print("          THESIS TREND PROOFS (BORDA RANKINGS)           ")
print("=========================================================")

# TREND 1 & 2: BEST OVERALL & CLUSTERING
print("\n[TREND 1 & 2]: BEST OVERALL & BEST FOR CLUSTERING (UNSUPERVISED)")
print_top(unsup_df, 'FINAL BORDA SCORE', "Top Distances for Unsupervised (DBSCAN + K-Medoids)", 4)

# TREND 3: BEST FOR SUPERVISED
print("\n\n[TREND 3]: BEST FOR SUPERVISED LEARNING (KNN + SVM)")
print_top(sup_df, 'FINAL BORDA SCORE', "Top Distances for Supervised (KNN + SVM)", 4)

# TREND 4: IMPACT OF DATA SIZE (ROWS)
print("\n\n[TREND 4]: HOW ROW SIZE AFFECTS METRICS (Low -> Med -> High)")
print_top(unsup_df, 'LR_Score', "Unsupervised: Low Row (LR) Datasets")
print_top(unsup_df, 'MR_Score', "Unsupervised: Medium Row (MR) Datasets")
print_top(unsup_df, 'HR_Score', "Unsupervised: High Row (HR) Datasets")

print_top(sup_df, 'LR_Score', "Supervised: Low Row (LR) Datasets")
print_top(sup_df, 'MR_Score', "Supervised: Medium Row (MR) Datasets")
print_top(sup_df, 'HR_Score', "Supervised: High Row (HR) Datasets")

# TREND 5: IMPACT OF DIMENSIONALITY (COLUMNS)
print("\n\n[TREND 5]: HOW COLUMN SIZE AFFECTS METRICS (Low -> Med -> High)")
print_top(unsup_df, 'LC_Score', "Unsupervised: Low Column (LC) Datasets")
print_top(unsup_df, 'MC_Score', "Unsupervised: Medium Column (MC) Datasets")
print_top(unsup_df, 'HC_Score', "Unsupervised: High Column (HC) Datasets")

print_top(sup_df, 'LC_Score', "Supervised: Low Column (LC) Datasets")
print_top(sup_df, 'MC_Score', "Supervised: Medium Column (MC) Datasets")
print_top(sup_df, 'HC_Score', "Supervised: High Column (HC) Datasets")

# TREND 6: IMPACT OF DATA IMBALANCE
print("\n\n[TREND 6]: HOW DATA IMBALANCE AFFECTS METRICS")
print_top(unsup_df, 'Balanced_Score', "Unsupervised: Balanced Datasets")
print_top(unsup_df, 'Imbalanced_Score', "Unsupervised: Imbalanced Datasets")

svm_df = master_df[master_df['Algorithm'] == 'SVM'].groupby('Distance Metric').sum().reset_index()
print_top(svm_df, 'Balanced_Score', "SVM Only: Balanced Datasets")
print_top(svm_df, 'Imbalanced_Score', "SVM Only: Imbalanced Datasets")
print("\n=========================================================")