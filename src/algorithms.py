import dataclasses as dc

import numpy as np
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


@dc.dataclass
class NullTracker:
    """Helper class to track consecutive nulls in a pandas series"""

    consecutive_count: int = 0
    max_consec_nulls: int = 0

    def next_two(self, inseries: pd.Series):
        """Wrapper to extract two values from pd.Series of len 2"""
        x1, x2 = inseries
        self._next_two(x1=x1, x2=x2)
        return self.max_consec_nulls

    def _next_two(self, x1: bool, x2: bool):
        """Update consecutive_count and max_consec_nulls based on current and next values."""
        if x2:  # next value is null
            self.consecutive_count += 1

        elif x1 and not x2:  # reached end of consecutive nulls
            if self.consecutive_count > self.max_consec_nulls:
                self.max_consec_nulls = self.consecutive_count
            self.consecutive_count = 0


def class_find_max_consec_nulls(series: pd.Series):
    """Find max number of consecutive nulls in a pandas series. Series needs
    to already be boolean using pd.isnull()"""
    tracker = NullTracker()
    if series.iloc[0]:  # if first value is null, need to manually add it
        tracker.consecutive_count = 1
    series.rolling(window=2, min_periods=2).apply(tracker.next_two)
    # if last value was null, max_consec_nulls would not have been updated
    return max(tracker.max_consec_nulls, tracker.consecutive_count)


def dumb_find_max_consec_nulls(series: pd.Series) -> int:
    """Helper func to reformat into numpy array for numba-optimised function.
    Inseries needs to be boolean using pd.isnull()
    """
    return _dumb_find_max_consec_nulls(series.to_numpy())


def _dumb_find_max_consec_nulls(array: np.array) -> int:
    """Find highest count of consecutive nulls in a 1D numpy array of bools."""
    consec_count = int(array[0])  # will start from 1 if first value is null
    max_consec_count = 0
    for x1, x2 in zip(array[:-1], array[1:]):
        if x2:  # next value is null
            consec_count += 1

        elif x1 and not x2:  # end of consecutive nulls
            if consec_count > max_consec_count:
                max_consec_count = consec_count
            consec_count = 0
    # if last value was null, we didn't get to update max_consec_count
    return max(consec_count, max_consec_count)
