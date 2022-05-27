import pandas as pd


DUMMY = pd.Series(1)


def index_find_max_consec_nulls(series: pd.Series) -> int:
    """Series should be raw data with nulls, without pd.isnull() applied"""
    if series.notnull().all():
        answer = 0
    elif series.isnull().all():
        answer = len(series)
    else:
        series = pd.concat([DUMMY, series, DUMMY])
        answer = int((pd.Series(series.reset_index(drop=True).dropna().index).diff() - 1).max())
    return answer


def _index_find_max_consec_nulls(series: pd.Series) -> int:
    """Separated logical steps of index_find_max_consec_nulls()"""
    if series.notnull().all():
        answer = 0
    elif series.isnull().all():
        answer = len(series)
    else:
        series = pd.concat([DUMMY, series, DUMMY])
        with_sequential_index = series.reset_index(drop=True)
        only_non_null_indices = with_sequential_index.dropna().index
        series_non_null_indices = pd.Series(only_non_null_indices)
        # minus 1 because we only want size of gap between indices
        size_of_null_gaps = series_non_null_indices.diff() - 1
        largest_null_gap = int(size_of_null_gaps.max())
        answer = largest_null_gap
    return answer


def pdgroupby_find_max_consec_nulls(series: pd.Series) -> int:
    """Series should have had pd.isnull() pre-applied"""
    return (series * (series.groupby((series != series.shift(1)).cumsum()).cumcount() + 1)).max()


def _pdgroupby_find_max_consec_nulls(series: pd.Series) -> int:
    """Separated logical steps of pdgroupby_find_max_consec_nulls()"""
    # find only the points where there is a change from null to not-null or vice-versa
    changes_rel_to_previous = series != series.shift(1)
    # label each row with an integer indicating which group of consecutive nulls/not nulls is is in
    labelled_groups_of_sequential_null_status = changes_rel_to_previous.cumsum()
    # for each group, run a cumulative count along each member in the group to count how long it is
    length_of_each_consecutive_group = (
        series.groupby(labelled_groups_of_sequential_null_status).cumcount() + 1
    )
    # filter out the groups that are not null
    only_null_consecutive_groups = series * length_of_each_consecutive_group
    # find the highest count of any null group
    return only_null_consecutive_groups.max()
