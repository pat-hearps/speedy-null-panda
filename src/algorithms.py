import pandas as pd


def index_find_max_consec_nulls(series: pd.Series) -> int:
    """Series should be raw data with nulls, without pd.isnull() applied"""
    return int(pd.Series(series.dropna().index).diff().max())


def _index_find_max_consec_nulls(series: pd.Series) -> int:
    """Separate logical steps of index_find_max_consec_nulls"""
    if series.notnull().all():
        answer = 0
    elif series.isnull().all():
        answer = len(series)
    else:
        with_sequential_index = series.reset_index(drop=True)
        only_non_null_indices = with_sequential_index.dropna().index
        series_non_null_indices = pd.Series(only_non_null_indices)
        size_of_null_gaps = series_non_null_indices.diff() - 1
        largest_null_gap = int(size_of_null_gaps.max())
        answer = largest_null_gap
    return answer
