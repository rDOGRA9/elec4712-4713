import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Suppress UndefinedMetricWarning for precision/recall if a fold misses a class prediction
warnings.filterwarnings('ignore', category=UserWarning)

def distance_to_kernel(dist_matrix, gamma=None):
    """
    Converts a distance matrix into a kernel (similarity) matrix for SVM.
    Uses the RBF transformation: K = exp(-gamma * distance)
    """
    if gamma is None:
        # Heuristic: 1 / median distance (avoids massive exponentiation issues)
        median_dist = np.median(dist_matrix)
        gamma = 1.0 / (median_dist if median_dist > 0 else 1.0)
        
    return np.exp(-gamma * dist_matrix)

def evaluate_supervised(dist_matrix, y, distance_metric_name, n_splits=5):
    """
    Runs 5-fold Stratified CV for KNN and SVM using a precomputed distance matrix.
    
    Args:
        dist_matrix (np.ndarray): The full N x N precomputed distance matrix.
        y (pd.Series or np.ndarray): Target labels.
        distance_metric_name (str): Name of the metric (for logging/results).
        n_splits (int): Number of CV folds.
        
    Returns:
        list of dicts: The averaged metrics for both KNN and SVM.
    """
    y_array = np.array(y)
    
    # Convert distance to kernel for SVM
    kernel_matrix = distance_to_kernel(dist_matrix)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Storage for fold metrics
    knn_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    svm_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    
    for train_idx, test_idx in skf.split(np.zeros(len(y_array)), y_array):
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        # ---------------------------------------------------------
        # MATRIX SLICING FOR PRECOMPUTED DISTANCES/KERNELS
        # KNN needs (n_samples, n_train_samples)
        # ---------------------------------------------------------
        
        # KNN Distance Slices
        dist_train = dist_matrix[train_idx][:, train_idx] # Train vs Train
        dist_test = dist_matrix[test_idx][:, train_idx]   # Test vs Train
        
        # SVM Kernel Slices
        kernel_train = kernel_matrix[train_idx][:, train_idx] # Train vs Train
        kernel_test = kernel_matrix[test_idx][:, train_idx]   # Test vs Train

        # ---------------------------------------------------------
        # K-NEAREST NEIGHBORS
        # ---------------------------------------------------------
        knn = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
        knn.fit(dist_train, y_train)
        knn_preds = knn.predict(dist_test)
        
        knn_metrics['acc'].append(accuracy_score(y_test, knn_preds))
        knn_metrics['prec'].append(precision_score(y_test, knn_preds, average='weighted', zero_division=0))
        knn_metrics['rec'].append(recall_score(y_test, knn_preds, average='weighted', zero_division=0))
        knn_metrics['f1'].append(f1_score(y_test, knn_preds, average='weighted', zero_division=0))
        
        # ---------------------------------------------------------
        # SUPPORT VECTOR MACHINE
        # ---------------------------------------------------------
        svm = SVC(kernel='precomputed', class_weight='balanced')
        svm.fit(kernel_train, y_train)
        svm_preds = svm.predict(kernel_test)
        
        svm_metrics['acc'].append(accuracy_score(y_test, svm_preds))
        svm_metrics['prec'].append(precision_score(y_test, svm_preds, average='weighted', zero_division=0))
        svm_metrics['rec'].append(recall_score(y_test, svm_preds, average='weighted', zero_division=0))
        svm_metrics['f1'].append(f1_score(y_test, svm_preds, average='weighted', zero_division=0))

    # Average the metrics across all 5 folds
    results = [
        {
            'Algorithm': 'KNN',
            'Distance Metric': distance_metric_name,
            'Accuracy': np.mean(knn_metrics['acc']),
            'Precision': np.mean(knn_metrics['prec']),
            'Recall': np.mean(knn_metrics['rec']),
            'F1 Score': np.mean(knn_metrics['f1'])
        },
        {
            'Algorithm': 'SVM',
            'Distance Metric': distance_metric_name,
            'Accuracy': np.mean(svm_metrics['acc']),
            'Precision': np.mean(svm_metrics['prec']),
            'Recall': np.mean(svm_metrics['rec']),
            'F1 Score': np.mean(svm_metrics['f1'])
        }
    ]
    
    return results