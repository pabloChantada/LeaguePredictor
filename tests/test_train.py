import numpy as np
import pytest
from sklearn.model_selection import GroupShuffleSplit

def test_group_shuffle_split_no_leakage():
    """
    Comprueba que el GroupShuffleSplit nunca mezcla un match_id
    en el set de train y en el de test al mismo tiempo.
    """
    # 6 filas, correspondientes a 3 partidas distintas (grupos)
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 1, 1, 0, 1])
    groups = np.array(["match_1", "match_1", "match_2", "match_2", "match_3", "match_3"])
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    # Obtenemos qué partidas cayeron en train y cuáles en test
    train_matches = set(groups[train_idx])
    test_matches = set(groups[test_idx])
    
    # No debe haber NINGUNA intersección entre los dos sets (0 leakage)
    assert len(train_matches.intersection(test_matches)) == 0