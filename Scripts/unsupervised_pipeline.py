import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import warnings

# Suppress warnings for poorly shaped clusters (common in experimental DBSCAN)
warnings.filterwarnings('ignore')

def evaluate_unsupervised(X, dist_matrix, y_true, distance_metric_name):
    """
    Runs K-Medoids and DBSCAN using precomputed distances.
    
    Args:
        X (pd.DataFrame): The normalized feature matrix (needed for DB and CH scores).
        dist_matrix (np.ndarray): The N x N precomputed distance matrix.
        y_true (pd.Series): True labels (used to determine 'k' for K-Medoids and for ARI/NMI).
        distance_metric_name (str): Name of the metric for logging.
        
    Returns:
        list of dicts: The clustering metrics for K-Medoids and DBSCAN.
    """
    results = []
    
    # We cheat slightly in unsupervised experiments by looking at y_true 
    # just to figure out how many clusters K-Medoids should search for.
    n_clusters = len(np.unique(y_true))

    # ==========================================
    # 1. K-MEDOIDS
    # ==========================================
    try:
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
        kmedoids_labels = kmedoids.fit_predict(dist_matrix)
        
        # Guard against edge cases where everything is assigned to 1 cluster
        if len(set(kmedoids_labels)) > 1:
            results.append({
                'Algorithm': 'K-Medoids',
                'Distance Metric': distance_metric_name,
                'Silhouette Score': silhouette_score(dist_matrix, kmedoids_labels, metric='precomputed'),
                'Davies-Bouldin Score': davies_bouldin_score(X, kmedoids_labels),
                'Calinski-Harabasz Score': calinski_harabasz_score(X, kmedoids_labels),
                'Adjusted Rand Index': adjusted_rand_score(y_true, kmedoids_labels),
                'Normalized Mutual Info': normalized_mutual_info_score(y_true, kmedoids_labels)
            })
        else:
            raise ValueError("K-Medoids collapsed into a single cluster.")
            
    except Exception as e:
        results.append({
            'Algorithm': 'K-Medoids', 'Distance Metric': distance_metric_name,
            'Silhouette Score': np.nan, 'Davies-Bouldin Score': np.nan, 
            'Calinski-Harabasz Score': np.nan, 'Adjusted Rand Index': np.nan, 
            'Normalized Mutual Info': np.nan
        })

    # ==========================================
    # 2. DBSCAN
    # ==========================================
    try:
        # Heuristic: Set eps to the 10th percentile of all non-zero distances.
        # This helps DBSCAN adapt to the varying scales of probabilistic distances.
        non_zero_dists = dist_matrix[dist_matrix > 0]
        eps_heuristic = np.percentile(non_zero_dists, 10) if len(non_zero_dists) > 0 else 0.5
        
        dbscan = DBSCAN(eps=eps_heuristic, min_samples=5, metric='precomputed')
        dbscan_labels = dbscan.fit_predict(dist_matrix)
        
        # DBSCAN tags noise as -1. We need at least one valid cluster + noise, or 2+ clusters
        if len(set(dbscan_labels)) > 1:
            results.append({
                'Algorithm': 'DBSCAN',
                'Distance Metric': distance_metric_name,
                'Silhouette Score': silhouette_score(dist_matrix, dbscan_labels, metric='precomputed'),
                'Davies-Bouldin Score': davies_bouldin_score(X, dbscan_labels),
                'Calinski-Harabasz Score': calinski_harabasz_score(X, dbscan_labels),
                'Adjusted Rand Index': adjusted_rand_score(y_true, dbscan_labels),
                'Normalized Mutual Info': normalized_mutual_info_score(y_true, dbscan_labels)
            })
        else:
            raise ValueError("DBSCAN assigned everything to noise or one cluster.")
            
    except Exception as e:
        results.append({
            'Algorithm': 'DBSCAN', 'Distance Metric': distance_metric_name,
            'Silhouette Score': np.nan, 'Davies-Bouldin Score': np.nan, 
            'Calinski-Harabasz Score': np.nan, 'Adjusted Rand Index': np.nan, 
            'Normalized Mutual Info': np.nan
        })

    return results