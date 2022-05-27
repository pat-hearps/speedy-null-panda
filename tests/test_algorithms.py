from typing import Callable

import numpy as np
import pandas as pd
import pytest

from src.algorithms import index_find_max_consec_nulls, _index_find_max_consec_nulls

DATA = [
    [1, 2, 3, 4, 5],
    [1, np.NaN, 3, 4, 5],
    [1, np.NaN, 3, np.NaN, np.NaN],
    [np.NaN, np.NaN, np.NaN, 4, 5],
    [np.NaN, 2, np.NaN, np.NaN, np.NaN],
    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
]
EXPECTED = [0, 1, 2, 3, 3, 5]


@pytest.fixture()
def df_test(data: np.array = DATA) -> pd.DataFrame:
    return pd.DataFrame(data)


functions = [index_find_max_consec_nulls, _index_find_max_consec_nulls]


@pytest.mark.parametrize("data, expected", zip(DATA, EXPECTED))
@pytest.mark.parametrize("func", functions)
def test_algorithm(data, expected, func: Callable) -> None:
    """Test algorithm function, one row are a time"""
    print("data=", data)
    result = func(pd.Series(data))
    print(f"result= {result}, expected= {expected}")
    assert result == expected
