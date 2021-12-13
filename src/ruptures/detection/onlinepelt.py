r"""OnlinePelt"""
from math import floor
from numpy import vstack, array
from ruptures.costs import cost_factory
from ruptures.base import BaseCost
from ruptures.exceptions import BadSegmentationParameters
from ruptures.utils import sanity_check


class OnlinePelt:
    """Penalized online change point detection.

    For a given model and penalty level, computes the segmentation which
    minimizes the constrained sum of approximation errors.
    """

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Initialize a Pelt instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None

        # initialization
        # partitions[t] contains the optimal partition of signal[0:t]
        self.partitions = dict()  # this dict will be recursively filled
        self.partitions[0] = {(0, 0): 0}
        self.last_admissible = []
        self.last_n = 0

    def _seg(self, pen, debug=False):
        """Computes the segmentation for a given penalty using PELT (or a list
        of penalties).

        Args:
            penalty (float): penalty value

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """

        admissible = self.last_admissible
        # Recursion
        ind = [k for k in range(self.last_n, self.n_samples, self.jump) if k >= self.min_size]
        ind += [self.n_samples]
        for bkp in ind:
            # adding a point to the admissible set from the previous loop.
            new_adm_pt = floor((bkp - self.min_size) / self.jump)
            new_adm_pt *= self.jump
            if new_adm_pt not in admissible:
                admissible.append(new_adm_pt)
            if debug:
                print(f"\tCP: {bkp}, admissible: {admissible}")
            subproblems = list()
            for t in admissible:
                # left partition
                try:
                    tmp_partition = self.partitions[t].copy()
                except KeyError:  # no partition of 0:t exists. Example:  t= 1: 1 < min_size; t=0 is initial case
                    continue
                # we update with the right partition
                tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
                subproblems.append(tmp_partition)
                if debug:
                    print(f"\t\t t={t}, subproblem = {subproblems[-1]}")
            # finding the optimal partition
            self.partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
            if debug:
                print(f"\t\t Best Bkps : {self.partitions[bkp]}")
            # trimming the admissible set
            admissible = [
                t
                for t, partition in zip(admissible, subproblems)
                if sum(partition.values()) <= sum(self.partitions[bkp].values()) + pen
            ]
            if debug:
                print(f"\t\t- new admissible: {admissible}")
        self.last_admissible = admissible
        self.last_n = self.n_samples
        best_partition = self.partitions[self.n_samples].copy()
        del best_partition[(0, 0)]
        return best_partition

    def predict(self, pen, debug=False):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.pelt.Pelt.fit].

        Raises:
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        # raise an exception in case of impossible segmentation configuration
        if self.cost.signal.shape[0] < self.min_size:
            return []
        if not sanity_check(
                n_samples=self.cost.signal.shape[0],
                n_bkps=0,
                jump=self.jump,
                min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        partition = self._seg(pen, debug)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def update_signal(self, subsignal, debug=False):
        """Set params.

                Args:
                    debug: True for print()-Debugging
                    subsignal (array): Append subsignal of shape (n_samples, n_features) or (n_samples,).

                Returns:
                    self
                """
        s = array(subsignal)
        if self.cost.signal is None:
            if s.ndim <= 1:
                self.cost.fit(s.reshape(-1, 1))
            else:
                self.cost.fit(s)
        else:
            _, d = self.cost.signal.shape
            if s.ndim <= 1:
                self.cost.fit(vstack((self.cost.signal, s.reshape(-1, d))))
                # hier könnte bei falschem Input auch ohne Fehler viel falsch gehen!
            else:
                self.cost.fit(vstack((self.cost.signal, s)))

        n_samples, _ = self.cost.signal.shape
        self.n_samples = n_samples
        if debug:
            print(self.cost.signal)
            print(self.n_samples)
        return self


if __name__ == "__main__":
    from ruptures.datasets import pw_constant
    from ruptures.detection import Pelt
    from time import time
    # import numpy as np
    from ruptures.show import display
    import matplotlib.pyplot as plt

    DEBUG = False
    n, dim = 20000, 2
    n_cps, sigma = 3000, 1.5
    signal, cps = pw_constant(n, dim, n_cps, noise_std=sigma)

    online = OnlinePelt(model="l1", min_size=1, jump=1)
    online.update_signal(signal[:19000]).predict(5)
    start = time()
    online.update_signal(signal[19000:]).predict(5)
    print(time() - start)  # 3.75
    offline = Pelt(model="l1", min_size=1, jump=1)
    start = time()
    offline.fit_predict(signal, 5)
    print(time() - start)  # 41.9

