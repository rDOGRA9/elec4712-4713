import numpy as np
import pandas as pd

def encode_target(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Converts categorical target columns into integers. 
    Works for binary (0, 1) and multi-class (0, 1, 2, 3...)
    """
    unique_vals = df[label_col].dropna().unique()
    
    # Dynamically map all unique string classes to an integer
    mapping = {val: i for i, val in enumerate(unique_vals)}
    df[label_col] = df[label_col].map(mapping)
    df[label_col] = df[label_col].astype(int)
    
    return df

def preprocess_for_pairwise(df: pd.DataFrame, label_col: str, encoding_method: str = "onehot") -> tuple:
    """
    Encodes categorical features, normalizes by max values, and appends a slack variable 'Adjusted_p'.
    Returns a tuple of (target_series, processed_dataframe).
    """
    df = df.copy()

    target_series = None
    if label_col and label_col in df.columns:
        # If the target isn't already a number, convert it safely
        if not np.issubdtype(df[label_col].dtype, np.number):
            df = encode_target(df, label_col)
            
        target_series = df[label_col].copy()
        df.drop(columns=[label_col], inplace=True)
    
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Encoding Logic
    if encoding_method.lower() == "onehot":
        cat_encoded = []
        for col in cat_cols:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float) 
            cat_encoded.append(dummies)
        if cat_encoded:
            cat_part = pd.concat(cat_encoded, axis=1)
        else:
            cat_part = pd.DataFrame(index=df.index)
            
    elif encoding_method.lower() == "target":
        cat_part = pd.DataFrame(index=df.index)
        for col in cat_cols:
            means = target_series.groupby(df[col]).mean()
            cat_part[col + "_target"] = df[col].map(means)
            
    else: # Ordinal
        cat_part = pd.DataFrame(index=df.index)
        for col in cat_cols:
            uniques = sorted(df[col].unique())
            mapping_dict = {cat_val: i + 1 for i, cat_val in enumerate(uniques)}
            cat_part[col + "_ord"] = df[col].map(mapping_dict)
    
    num_part = df[num_cols].copy()
    df_processed = pd.concat([num_part, cat_part], axis=1)

    # Clean any resulting NaNs and align target_series to the remaining rows
    df_processed.dropna(inplace=True)
    target_series = target_series.loc[df_processed.index]
    
    # Normalization & Slack Variable (Adjusted_p)
    col_max = df_processed.max()
    col_max = col_max.replace(0, 1) # Prevent division by zero
    
    num_columns = df_processed.shape[1]
    df_normalized = df_processed.div(col_max * num_columns, axis=1)
    row_sums = df_normalized.sum(axis=1)
    df_normalized['Adjusted_p'] = 1 - row_sums
    
    return target_series, df_normalized