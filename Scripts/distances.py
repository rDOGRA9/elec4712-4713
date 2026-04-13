import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import wasserstein_distance

# =====================================================================
# PROBABILISTIC DISTANCE FUNCTIONS
# Note: These assume inputs p and q are probability distributions 
# (non-negative, summing to 1). Your 'Adjusted_p' preprocessing handles this.
# =====================================================================

EPSILON = 1e-10  # Prevent log(0) and division by zero errors

def kl_divergence(p, q):
    """Symmetric Kullback-Leibler divergence."""
    p_safe = p + EPSILON
    q_safe = q + EPSILON
    
    # Standard KL
    kl_pq = np.sum(p_safe * np.log(p_safe / q_safe))
    kl_qp = np.sum(q_safe * np.log(q_safe / p_safe))
    
    # Return symmetric KL to ensure valid distance matrices for ML algorithms
    return kl_pq + kl_qp

def bhattacharyya(p, q):
    """Bhattacharyya distance."""
    p_safe = p + EPSILON
    q_safe = q + EPSILON
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p_safe * q_safe))
    return -np.log(bc)

def total_variation(p, q):
    """Total Variation distance."""
    return 0.5 * np.sum(np.abs(p - q))

def earth_movers(p, q):
    """
    Earth Mover's Distance (1D Wasserstein distance).
    Since p and q are weights across the same feature indices, we use 
    feature indices as the coordinate space.
    """
    indices = np.arange(len(p))
    return wasserstein_distance(indices, indices, u_weights=p, v_weights=q)

# =====================================================================
# MASTER DISTANCE DICTIONARY
# =====================================================================

DISTANCE_METRICS = {
    # Standard Scipy distances (string names map to highly optimized C-code in Scipy)
    "euclidean": "euclidean",
    "manhattan": "cityblock",      # Scipy uses 'cityblock' for Manhattan
    "minkowski": "minkowski",
    "chebyshev": "chebyshev",
    "canberra": "canberra",
    "braycurtis": "braycurtis",
    "hamming": "hamming",
    "cosine": "cosine",

    # Custom probabilistic distances (callable functions)
    "earth_movers": earth_movers,
    "kl_divergence": kl_divergence,
    "jensen_shannon": jensenshannon, # Natively supported by scipy.spatial.distance
    "bhattacharyya": bhattacharyya,
    "total_variation": total_variation
}

# =====================================================================
# MATRIX GENERATOR
# =====================================================================

def compute_distance_matrix(X, metric_name):
    """
    Computes a pairwise distance matrix for dataset X using the specified metric.
    
    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix (must not contain the target).
        metric_name (str): The name of the distance metric to use.
        
    Returns:
        np.ndarray: A square, symmetric pairwise distance matrix.
    """
    if metric_name not in DISTANCE_METRICS:
        valid_keys = list(DISTANCE_METRICS.keys())
        raise ValueError(f"Metric '{metric_name}' not supported. Choose from: {valid_keys}")
    
    metric = DISTANCE_METRICS[metric_name]
    X_array = np.array(X)
    
    # cdist handles both fast string metrics and slow custom python functions
    dist_matrix = cdist(X_array, X_array, metric=metric)
    
    # Post-processing: Force strict symmetry and zero out the diagonal. 
    # Custom float math can sometimes result in numbers like 1e-16 on the diagonal 
    # instead of true 0, which can confuse SVMs and DBSCAN.
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    
    dist_matrix = np.maximum(dist_matrix, 0.0)
    
    return dist_matrix