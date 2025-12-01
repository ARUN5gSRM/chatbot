# cb_app/sub_models/utils.py
from typing import List
import numpy as np

def normalize_vector(vec: List[float], dim: int = 384) -> List[float]:
    a = np.array(vec, dtype=float)
    norm = np.linalg.norm(a)
    if norm == 0 or np.isnan(norm):
        return [0.0] * dim
    return (a / norm).tolist()
