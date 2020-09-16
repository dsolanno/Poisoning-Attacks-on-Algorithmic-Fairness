"""
.. module:: CLineSearchBisect
   :synopsis: Binary line search.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
import numpy as np

from secml.optim.optimizers.line_search import CLineSearch
from secml.array import CArray


class CLineSearchBisect(CLineSearch):
    """Binary line search.

    Attributes
    ----------
    class_type : 'bisect'

    """
    __class_type = 'bisect'

    def __init__(self, fun, constr=None, bounds=None,
                 eta=1e-4, eta_min=0.1, eta_max=None,
                 max_iter=20):

        CLineSearch.__init__(
            self, fun=fun, constr=constr, bounds=bounds,
            eta=eta, max_iter=max_iter)

        # init attributes
        self._eta_max = None
        self._eta_min = None

        # init attributes (with setters)
        self.eta_max = eta_max
        self.eta_min = eta_min

        # other internal parameters
        self._n_iter = 0
        self._fx = None  # cached value of fun at x (initial point)
        self._fz = None  # cached value of fun at current z during line search
        self._fun_idx_max = None
        self._fun_idx_min = None

    @property
    def eta_max(self):
        return self._eta_max

    @eta_max.setter
    def eta_max(self, value):
        """Sets eta_max to value (multiple of eta).

        Parameters
        ----------
        value: CArray or None

        """
        if value is None:
            self._eta_max = None
            return
        # set eta_max >= t*eta, t >= 1 (integer)
        self._eta_max = self.eta * max(value / self.eta, 1)

    @property
    def eta_min(self):
        return self._eta_min

    @eta_min.setter
    def eta_min(self, value):
        """Sets eta_min to value (multiple of eta).

        Parameters
        ----------
        value: CArray or None

        """
        if value is None:
            self._eta_min = None
            return

        # set eta_min >= t*eta, t >= 1 (integer)
        t = CArray(value / self.eta).round()
        t = t if t > 1 else CArray([1])
        self._eta_min = self.eta * t

    @property
    def n_iter(self):
        return self._n_iter

    def _update_z(self, x, eta, d):
        """Update z and its cached score fz."""
        z = x + eta * d
        self._fz = self.fun.fun(z)
        return z

    def _is_feasible(self, x):
        """Checks if x is within the feasible domain."""
        constr_violation = False if self.constr is None else \
            self.constr.is_violated(x)
        bounds_violation = False if self.bounds is None else \
            self.bounds.is_violated(x)

        if constr_violation or bounds_violation:
            return False

        return True

    def _select_best_point(self, x, d, idx_min, idx_max, **kwargs):
        """Returns best point among x and the two points found by the search.
        In practice, if f(x + eta*d) increases on d, we return x."""

        # dtype of x1 and x2 depends on x and eta (the grid discretization)
        if np.issubdtype(x.dtype, np.floating):
            # if x is float res dtype should be float
            dtype = x.dtype
        else:  # x is int, so the res dtype depends on the grid discretization
            dtype = self.eta.dtype

        x1 = CArray(x + d * self.eta * idx_min,
                    dtype=dtype, tosparse=x.issparse)
        x2 = CArray(x + d * self.eta * idx_max,
                    dtype=dtype, tosparse=x.issparse)

        self.logger.info("Select best point between: " +
                         str(x + self.eta * d * idx_min) + ", " +
                         str(x + self.eta * d * idx_max) + ", " +
                         str(x))
        self.logger.debug("f[a], f[b]: [" +
                          str(self._fun_idx_min) + "," +
                          str(self._fun_idx_max) + "]")
        self.logger.debug("f[x] " + str(self._fx))

        f0 = self._fx

        if not self._is_feasible(x1) and \
                not self._is_feasible(x2):
            self.logger.debug("x1 and x2 are not feasible. Returning x.")
            return x, f0

        # uses cached values (if available) to save computations
        f1 = self._fun_idx_min if self._fun_idx_min is not None else \
            self.fun.fun(x1, **kwargs)

        if not self._is_feasible(x2):
            if f1 < f0:
                self.logger.debug("x2 not feasible. Returning x1."
                                  " f(x): " + str(f0) +
                                  ", f(x1): " + str(f1))
                return x1, f1
            self.logger.debug("x2 not feasible. Returning x."
                              " f(x): " + str(f0) +
                              ", f(x1): " + str(f1))
            return x, f0

        # uses cached values (if available) to save computations
        f2 = self._fun_idx_max if self._fun_idx_max is not None else \
            self.fun.fun(x2, **kwargs)

        if not self._is_feasible(x1):
            if f2 < f0:
                self.logger.debug("x1 not feasible. Returning x2.")
                return x2, f2
            self.logger.debug("x1 not feasible. Returning x.")
            return x, f0

        # else return best point among x1, x2 and x
        self.logger.debug("f0: {:}, f1: {:}, f2: {:}".format(f0, f1, f2))

        if f2 <= f0 and f2 < f1:
            self.logger.debug("Returning x2.")
            return x2, f2

        if f1 <= f0 and f1 < f2:
            self.logger.debug("Returning x1.")
            return x1, f1

        self.logger.debug("Returning x.")
        self.logger.debug("f0: {:}".format(f0))
        return x, f0

    def _is_decreasing(self, x, d, **kwargs):
        """
        Returns True if function at `x + eps*d` is decreasing,
        or False if it is increasing or out of feasible domain.
        """
        # IMPORTANT: requires self._fz to be set to fun.fun(z)
        # This is done to save function evaluations

        if not self._is_feasible(x):
            # point is outside of feasible domain
            return False

        # this could be in the order of 1e-10 or 1e-12, if eta is very small
        delta = self.fun.fun(x + 0.1 * self.eta * d, **kwargs) - self._fz

        if delta <= 0:
            # feasible point, decreasing / stationary score
            return True

        # feasible point, increasing score
        return False

    def _compute_eta_max(self, x, d, **kwargs):

        # double eta each time until function increases or goes out of bounds
        eta = self.eta if self.eta_min is None else self.eta_min

        # eta_min may be too large, going out of bounds,
        # or jumping out of the local minimum
        # it this happens, we reduce it,
        # ensuring a feasible point or a minimal step (multiple of self.eta)
        # this helps getting closer to the violated constraint
        t = CArray(eta / self.eta).round()

        self.logger.debug(
            "[_compute_eta_max] eta: " + str(eta) + ", x: " +
            str(x[x != 0]) + ", f(x): " + str(self._fx))
        # update z and fz
        z = self._update_z(x, eta, d)

        self.logger.debug(
            "[_compute_eta_max] eta max, eta: " + str(eta) + ", z: " +
            str(z[z != 0]) + ", f(z): " + str(self._fz))

        # divide eta by 2 if x+eta*d goes out of bounds or fz decreases
        # update (if required) z and fz
        while eta > self.eta and \
                (not self._is_feasible(z) or self._fz > self._fx):
            t = CArray(t / 2).round()
            eta = t * self.eta

            # store fz (for current point)
            z = self._update_z(x, eta, d)

        # exponential line search starts here
        while self._n_iter < self.max_iter:

            # cache f_min
            self._fun_idx_min = self._fz

            eta *= 2

            # update z and fz
            z = self._update_z(x, eta, d)

            # cache f_max
            self._fun_idx_max = self._fz

            self.logger.debug(
                "[_compute_eta_max] eta: " + str(eta) + ", z: " +
                str(z[z != 0]) + ", f(z): " + str(self._fz))

            self._n_iter += 1

            # function started increasing or end of bounds
            if not self._is_decreasing(z, d, **kwargs):
                return eta

        self.logger.debug('Maximum iterations reached. Exiting.')
        return eta

    def minimize(self, x, d, fx=None, tol=1e-4, **kwargs):
        """Bisect line search (on discrete grid).

        The function `fun( x + a*eta*d )` with `a = {0, 1, 2, ... }`
        is minimized along the descent direction d.

        If `fun(x) >= 0` -> step_min = step
        else step_max = step

        If eta_max is not None, it runs a bisect line search in
        `[x + eta_min*d, x + eta_max*d];
        otherwise, it runs an exponential line search in
        `[x + eta*d, ..., x + eta_min*d, ...]`

        Parameters
        ----------
        x : CArray
            The input point.
        d : CArray
            The descent direction along which `fun(x)` is minimized.
        fx : int or float or None, optional
            The current value of `fun(x)` (if available).
        tol : float, optional
            Tolerance for convergence to the local minimum.
        kwargs : dict
            Additional parameters required to evaluate `fun(x, **kwargs)`.

        Returns
        -------
        x' : CArray
            Point `x' = x + eta * d` that approximately
            solves `min f(x + eta*d)`.
        fx': int or float or None, optional
            The value `f(x')`.

        """
        d = CArray(d, tosparse=d.issparse).ravel()

        self._n_iter = 0

        # func eval
        self.logger.debug("received fx: {:}".format(fx))

        self._fx = self.fun.fun(x) if fx is None else fx
        self._fz = self._fx

        self.logger.info(
            "line search: " + str(x[x != 0]) +
            ", f(x): " + str(self._fx))

        # reset cached values
        self._fun_idx_min = None
        self._fun_idx_max = None

        # exponential search
        if self.eta_max is None:
            self.logger.debug("Exponential search ")
            eta_max = self._compute_eta_max(x, d, **kwargs)
            idx_max = (eta_max / self.eta).ceil().astype(int)
            idx_min = (idx_max / 2).astype(int)
            # this only searches within [eta, 2*eta]
            # the values fun_idx_min and fun_idx_max are already cached
        else:
            self.logger.debug("Binary search ")
            idx_max = (self.eta_max / self.eta).ceil().astype(int)
            idx_min = 0
            self._fun_idx_min = self._fx
            self._fun_idx_max = None  # this has not been cached

        self.logger.info("Running binary line search in: [" +
                         str(x + self.eta * d * idx_min) + "," +
                         str(x + self.eta * d * idx_max) + "]")
        self.logger.debug("f[a], f[b]: [" +
                          str(self._fun_idx_min) + "," +
                          str(self._fun_idx_max) + "]")

        while self._n_iter < self.max_iter:

            if idx_min == 0:
                if (idx_max <= 1).any():
                    # local minimum found
                    self.logger.debug("local minimum found")
                    return self._select_best_point(
                        x, d, idx_min, idx_max, **kwargs)
            else:
                if (idx_max - idx_min <= 1).any():
                    # local minimum found
                    self.logger.debug("local minimum found")
                    return self._select_best_point(
                        x, d, idx_min, idx_max, **kwargs)

            # else, continue...
            idx = (0.5 * (idx_min + idx_max)).astype(int)

            fz_prev = self._fz

            # update z, fz
            z = self._update_z(x, self.eta, d * idx)

            self.logger.debug(
                ", z: " + str(z[z != 0]) +
                ", f(z): " + str(self._fz))

            self._n_iter += 1

            if self._is_decreasing(z, d, **kwargs):
                idx_min = idx
                self._fun_idx_min = self._fz
            else:
                idx_max = idx
                self._fun_idx_max = self._fz

            # check if we are approaching the minimum (flat region)
            if self._is_feasible(z) and abs(self._fz - fz_prev) <= tol:
                self.logger.debug('Reached flat region. Exiting.')
                return self._select_best_point(
                    x, d, idx_min, idx_max, **kwargs)

        self.logger.debug('Maximum iterations reached. Exiting.')
        return self._select_best_point(x, d, idx_min, idx_max, **kwargs)
