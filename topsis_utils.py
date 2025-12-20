import numpy as np
import pandas as pd
from typing import List

def calculate_topsis(decision_matrix: pd.DataFrame, 
                     weights: List[float], 
                     criteria_type: List[str]) -> pd.DataFrame:
    """
    Melakukan perhitungan TOPSIS.
    """
    if not np.isclose(sum(weights), 1.0):
        weights = np.array(weights) / sum(weights)
    
    X = np.nan_to_num(decision_matrix.values, nan=0.0)
    denominator = np.sqrt((X ** 2).sum(axis=0))
    denominator[denominator == 0] = 1e-9
    R = X / denominator
    V = R * np.array(weights)
    
    A_plus = np.zeros(V.shape[1])
    A_minus = np.zeros(V.shape[1])
    
    for j in range(V.shape[1]):
        if criteria_type[j].lower() == 'benefit':
            A_plus[j] = V[:, j].max()
            A_minus[j] = V[:, j].min()
        elif criteria_type[j].lower() == 'cost':
            A_plus[j] = V[:, j].min()
            A_minus[j] = V[:, j].max()
    
    S_plus = np.sqrt(((V - A_plus)**2).sum(axis=1))
    S_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))
    Closeness = S_minus / (S_plus + S_minus)
    
    closeness_series = pd.Series(Closeness, index=decision_matrix.index)
    
    results_df = pd.DataFrame({
        'Strategy': decision_matrix.index,
        'Closeness_Score': Closeness,
        'Rank': closeness_series.rank(method='dense', ascending=False).astype(int)
    }).sort_values(by='Closeness_Score', ascending=False).set_index('Strategy')
    
    return results_df