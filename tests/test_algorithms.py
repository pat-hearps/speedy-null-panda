from typing import Callable

import numpy as np
import pandas as pd
import pytest

from src.algorithms import (
    index_find_max_consec_nulls,
    _index_find_max_consec_nulls,
    pdgroupby_find_max_consec_nulls,
    _pdgroupby_find_max_consec_nulls,
    class_find_max_consec_nulls,
    dumb_find_max_consec_nulls,
)

DATA = [
    [1, 2, 3, 4, 5],
    [1, np.NaN, 3, 4, 5],
    [1, np.NaN, 3, np.NaN, np.NaN],
    [np.NaN, np.NaN, np.NaN, 4, 5],
    [np.NaN, 2, np.NaN, np.NaN, np.NaN],
    [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
]
EXPECTED = [0, 1, 2, 3, 3, 5]

functions = [
    (index_find_max_consec_nulls, False),
    (_index_find_max_consec_nulls, False),
    (pdgroupby_find_max_consec_nulls, True),
    (_pdgroupby_find_max_consec_nulls, True),
    (class_find_max_consec_nulls, True),
    (dumb_find_max_consec_nulls, True),
]


@pytest.mark.parametrize("data, expected", zip(DATA, EXPECTED))
@pytest.mark.parametrize("func, pre_null", functions)
def test_algorithm(data, expected, func: Callable, pre_null: bool) -> None:
    """Test algorithm function, one row are a time"""
    print("data=", data)
    if pre_null:
        series = pd.Series(data).isnull()
    else:
        series = pd.Series(data)

    result = func(series)
    print(f"result= {result}, expected= {expected}")
    assert result == expected
