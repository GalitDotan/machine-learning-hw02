#################################
# Your name: Galit Dotan
#################################
from enum import Enum
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit

from intervals import find_best_interval


class Tag(Enum):
    L1 = 1
    U1 = 2
    L2 = 3
    U2 = 4

    def __repr__(self):
        return self.name


class State(Enum):
    InIn = 1
    InOut = 2
    OutIn = 3
    OutOut = 4


Interval = tuple[float, float]
Intervals = list[Interval]
Point = tuple[float, Tag]
Points = list[Point]

STATE_SWITCHER: dict[State, dict[Tag, State]] = {
    State.InIn: {
        Tag.U1: State.OutIn,
        Tag.U2: State.InOut,
    },
    State.InOut: {
        Tag.U1: State.OutOut,
        Tag.L2: State.InIn,
    },
    State.OutIn: {
        Tag.L1: State.InIn,
        Tag.U2: State.OutOut,
    },
    State.OutOut: {
        Tag.L1: State.InOut,
        Tag.L2: State.OutIn,
    },
}

# note: if the intervals didn't start at 0.0 and didn't end at 1.0, we would have had to add 0-length intervals
# for the merge to work correctly
IN_INTERVALS = [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]
IN_INTERVALS_NP = np.array([(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)], dtype=[('l', 'f4'), ('u', 'f4')])

_IN_IN_PROB = 0.8
_OUT_IN_PROB = 0.1

STATE_TO_PROB = {
    State.InIn: _IN_IN_PROB,
    State.InOut: 1 - _IN_IN_PROB,
    State.OutIn: _OUT_IN_PROB,
    State.OutOut: 1 - _OUT_IN_PROB
}


class Assignment2(object):
    def sample_from_D(self, m: int) -> ndarray[Any, dtype[floating[_64Bit]]]:
        """
        Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2):
                A two-dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_samples = np.random.uniform(0, 1, m)  # m values which are sampled IID over [0, 1]
        x_samples.sort()
        tag_per_sample = np.vectorize(self.is_in_intervals)(x_samples)
        labels = np.random.binomial(1, p=np.where(tag_per_sample, _IN_IN_PROB, _OUT_IN_PROB))
        result = np.column_stack((x_samples, labels))
        return result

    def experiment_m_range_erm(self, m_first: int, m_last: int, step: int, k: int, T: int):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two-dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        # part (b)
        n_values = np.arange(m_first, m_last + 1, step)
        empirical_errs = []
        true_errs = []

        for n in n_values:
            sample = self.sample_from_D(n)
            intervals, error_cnt = find_best_interval(sample[:, 0], sample[:, 1], k)
            empirical_errs.append(error_cnt / n)
            print(intervals)
            intervals = [(float(x[0]), float(x[1])) for x in intervals]
            true_errs.append(self.calc_true_error(intervals))

        print(true_errs)
        print(empirical_errs)

        plt.plot(n_values, empirical_errs, label='Empirical Error')
        plt.plot(n_values, true_errs, label='True Error')
        plt.xlabel("Sample Size (n)")
        plt.ylabel("Error")
        plt.title(f"Empirical and True Errors for k={k}, Averaged Over {T} Runs")
        plt.legend()
        plt.grid()
        plt.show()

        # Step 6: Return errors as a NumPy array
        return np.column_stack((empirical_errs, true_errs))

    def experiment_k_range_erm(self, m: int, k_first: int, k_last: int, step: int):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # part (c)
        # TODO: Implement the loop
        pass

    def experiment_k_range_srm(self, m: int, k_first: int, k_last: int, step: int):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # part (d)
        # TODO: Implement the loop
        pass

    def cross_validation(self, m: int):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross-validation algorithm.
        """
        # part (e)
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods

    @staticmethod
    def merge_intervals(true_intervals: Intervals, intervals: Intervals) -> Points:
        """
        Merges two sorted lists into a single sorted list.
        """
        merged: Points = []
        i, j = 0, 0  # Pointers for list1 and list2
        i_loc, j_loc = 0, 0

        # Merge elements from both lists until one is exhausted
        while i < len(true_intervals) and j < len(intervals):
            if true_intervals[i][i_loc] < intervals[j][j_loc]:
                if i_loc == 0:  # move to the next index in the same interval
                    merged.append((true_intervals[i][i_loc], Tag.L1))
                    i_loc = 1
                else:  # move to the next interval
                    merged.append((true_intervals[i][i_loc], Tag.U1))
                    i += 1
                    i_loc = 0
            else:
                if j_loc == 0:  # move to the next index in the same interval
                    merged.append((intervals[j][j_loc], Tag.L2))
                    j_loc = 1
                else:  # move to the next interval
                    merged.append((intervals[j][j_loc], Tag.U2))
                    j += 1
                    j_loc = 0

        # Append any remaining elements from list1
        if i < len(true_intervals):
            if i_loc == 1:
                merged.append((true_intervals[i][1], Tag.U1))
                i += 1
            for interval in true_intervals[i:]:
                merged.append((interval[0], Tag.L1))
                merged.append((interval[1], Tag.U1))

        # Append any remaining elements from list2
        if j < len(intervals):
            if j_loc == 1:
                merged.append((intervals[j][1], Tag.U2))
                j += 1
            for interval in intervals[j:]:
                merged.append((interval[0], Tag.L2))
                merged.append((interval[1], Tag.U2))

        return merged

    def tag_intervals(self, intervals: Intervals,
                      true_intervals: Intervals) -> dict[State, Intervals]:
        """
        This function goes over `intervals` (which are assumed to be sorted)
        And builds four new lists of intervals:
            1. Parts that are both in `intervals` and in `true_intervals`
            2. Parts that are in `intervals` but not in `true_intervals`
            3. Parts that are in `true_intervals` but not in `intervals`
            4. Parts that are not in `intervals` and not in `true_intervals`
        """
        states_to_intervals: dict[State, Intervals] = {
            State.InIn: [],
            State.InOut: [],
            State.OutIn: [],
            State.OutOut: [],
        }

        sorted_tagged_points = self.merge_intervals(true_intervals, intervals)

        prev, prev_tag = 0.0, Tag.U1
        state: State = State.OutOut

        for curr, curr_tag in sorted_tagged_points:
            states_to_intervals[state].append((prev, curr))
            state = STATE_SWITCHER[state][curr_tag]
            prev, prev_tag = curr, curr_tag

        return states_to_intervals

    def calc_true_error(self, intervals: Intervals,
                        true_intervals: Intervals = tuple(IN_INTERVALS)):
        """
        Calculates the true error of the given intervals, based on the "true intervals".
        """
        states_to_intervals = self.tag_intervals(intervals, true_intervals)
        states_to_length = {state: sum([u - l for l, u in intervals]) for state, intervals in
                            states_to_intervals.items()}

        true_error = sum(STATE_TO_PROB[state] * states_to_length[state] for state in State)
        return true_error

    # def calc_empirical_error(self, hypothesis, ):
    #    pass

    @staticmethod
    def is_in_intervals(x: float, intervals: np.array = IN_INTERVALS_NP) -> bool:
        return np.any((x >= intervals['l']) & (x <= intervals['u']))


#################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
