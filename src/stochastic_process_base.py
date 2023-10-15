from abc import ABC, abstractmethod
from typing import Optional, Union, List, NoReturn, Any, Protocol, Callable
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import numpy as np
from src.stocastic_interfaces import Drift, Sigma, Init_P


class Brownian_Motion:
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed=seed)

    def get_dW(self, intervals: int) -> np.ndarray:
        return self.generator.normal(loc=0.0, scale=1.0, size=intervals)

    def get_W(self, intervals: int) -> np.ndarray:
        dW = self.get_dW(intervals)
        dW_cs = dW.cumsum()
        return dW_cs

    def _get_correlated_dW(self, dW: np.ndarray, rho: float) -> np.ndarray:
        dW2 = self.get_dW(len(dW))

        if np.array_equal(dW2, dW):
            raise ValueError(
                "Brownian Increment error, try choosing different random state."
            )

        return rho * dW + np.sqrt(1 - rho**2) * dW2

    def _get_corr_ref_dW(self, dWs: list[np.ndarray], i: int) -> np.ndarray:
        random_proc_idx = self.generator.choice(i)
        return dWs[random_proc_idx]

    def get_corr_dW_matrix(
        self,
        intervals: int,
        n_procs: int,
        rho: Optional[float] = None,
    ) -> np.ndarray:
        dWs: list[np.ndarray] = []
        for i in range(n_procs):
            if i == 0 or rho is None:
                dW_i = self.get_dW(intervals)
            else:
                dW_corr_ref = self._get_corr_ref_dW(dWs, i)
                dW_i = self._get_correlated_dW(dW_corr_ref, rho)

            dWs.append(dW_i)

        return np.asarray(dWs).T


@dataclass
class Stochastic_Params_Base:
    mean_reversion: float
    asymptotic_mean: float
    std_dev: float


class Stocashtic_Process_Base(ABC):
    def __init__(
        self,
        seed: float,
        param_obj: Union[Stochastic_Params_Base, List[Stochastic_Params_Base]],
    ):
        brwn_inst = Brownian_Motion(seed)
        self.brownian_motion = brwn_inst
        self.model_params = param_obj

    @property
    def brownian_motion(self) -> Brownian_Motion:
        return self._brownian_motion

    @brownian_motion.setter
    def brownian_motion(self, brwn_inst: Brownian_Motion) -> None:
        self._brownian_motion = brwn_inst

    @property
    def model_params(
        self,
    ) -> Union[Stochastic_Params_Base, List[Stochastic_Params_Base]]:
        return self._model_params

    @model_params.setter
    def model_params(
        self, model_params: Union[Stochastic_Params_Base, List[Stochastic_Params_Base]]
    ) -> None:
        self._model_params = model_params

    @abstractmethod
    def estimate_params(self, time_series: np.ndarray) -> Stochastic_Params_Base:
        raise NotImplementedError("Must be implemented in sub-class")

    @abstractmethod
    def create_sim(
        self,
        intervals: int,
        stoch_params: Stochastic_Params_Base,
        dW: np.array,
        X_0: Optional[float] = None,
    ) -> np.ndarray:
        raise NotImplementedError("Must be implemented in sub-class")

    def create_correlated_sims(
        self,
        intervals: int,
        n_procs: Optional[int] = None,
        proc_correlation: Optional[float] = None,
    ) -> np.ndarray:
        _n_procs = self._get_n_procs(self.model_params, n_procs)

        corr_dWs = self.brownian_motion.get_corr_dW_matrix(
            intervals, _n_procs, proc_correlation
        )

        sim_list = []
        for i in range(_n_procs):
            if isinstance(self.model_params, list):
                sim_params_i = self.model_params[i]
            else:
                sim_params_i = self.model_params

            dW_i = corr_dWs[:, i]

            ou_sim = self.create_sim(intervals, sim_params_i, dW_i)

            if any(np.isnan(ou_sim)):
                raise ValueError(f"{sim_params_i}, {i}/{_n_procs} had NAs. Failing")

            sim_list.append(ou_sim)

        return np.asarray(sim_list).T

    def _get_n_procs(
        self,
        stoch_params: Union[Stochastic_Params_Base, List[Stochastic_Params_Base]],
        n_procs: Optional[int],
    ) -> int:
        if isinstance(stoch_params, list):
            return len(stoch_params)
        elif n_procs is None:
            raise ValueError("If stoch_params is not tuple, n_procs must be specified")
        return n_procs


class OU_Process(Stocashtic_Process_Base):
    def create_sim(
        self,
        intervals: int,
        OU_params: Stochastic_Params_Base,
        dW: np.array,
        X_0: Optional[float] = None,
    ) -> np.array:
        interval_arr = np.arange(intervals, dtype=np.longdouble)
        exp_alpha_t = np.exp(-OU_params.mean_reversion * interval_arr)

        integral_W = OU_Process._get_integal_W(interval_arr, dW, OU_params)
        _X_0 = OU_Process._select_X_0(X_0, OU_params)

        return (
            _X_0 * exp_alpha_t
            + OU_params.asymptotic_mean * (1 - exp_alpha_t)
            + OU_params.std_dev * exp_alpha_t * integral_W
        )

    def estimate_params(self, X_t: np.ndarray) -> Stochastic_Params_Base:
        y = np.diff(X_t)
        X = X_t[:-1].reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)

        alpha = -reg.coef_[0]
        gamma = reg.intercept_ / alpha

        y_hat = reg.predict(X)
        beta = np.std(y - y_hat)

        return Stochastic_Params_Base(
            mean_reversion=alpha, asymptotic_mean=gamma, std_dev=beta
        )

    def _select_X_0(
        X_0_in: Optional[float], OU_params: Stochastic_Params_Base
    ) -> float:
        if X_0_in is not None:
            return X_0_in
        return OU_params.asymptotic_mean

    def _get_integal_W(
        intervals: np.ndarray, dW: np.ndarray, OU_params: Stochastic_Params_Base
    ) -> np.ndarray:
        exp_alpha_s = np.exp(OU_params.mean_reversion * intervals)
        integral_W = np.cumsum(exp_alpha_s * dW)
        return integral_W


# TODO should probably make OU_Params class for consistency instead of just using Stocastic_Params_Base
@dataclass
class CIR_Params(Stochastic_Params_Base):
    # NOTE super fun, haven't seen post_init before!
    def __post_init__(self) -> Optional[NoReturn]:
        if 2 * self.mean_reversion * self.asymptotic_mean < self.std_dev**2:
            raise ValueError("2ab has to be less than or equal to c^2.")
        return None


class CIR_Process(Stocashtic_Process_Base):
    def _validate_not_nan(self, dsigma_t: Any) -> Optional[NoReturn]:
        if np.isnan(dsigma_t):
            raise ValueError(
                "CIR process simulation crashed, check your CIR_params. "
                + "Maybe choose a smaller c value."
            )
        return None

    def create_sim(
        self,
        intervals: int,
        CIR_params: CIR_Params,
        dW: np.array,
        sigma_0: Optional[float] = None,
    ) -> np.ndarray:
        return self._generate_CIR_process(dW, CIR_params, sigma_0)

    def _generate_CIR_process(
        self, dW: np.ndarray, CIR_params: CIR_Params, sigma_0: Optional[float] = None
    ) -> np.ndarray:
        if sigma_0 is None:
            sigma_0 = CIR_params.asymptotic_mean

        sigma_t = [sigma_0]
        for t in range(1, len(dW)):
            dsigma_t = (
                CIR_params.mean_reversion
                * (CIR_params.asymptotic_mean - sigma_t[t - 1])
                + CIR_params.std_dev * np.sqrt(sigma_t[t - 1]) * dW[t]
            )

            self._validate_not_nan(dsigma_t)
            sigma_t.append(sigma_t[t - 1] + dsigma_t)

        return np.asarray(sigma_t)

    def estimate_params(self, sigma_t: np.ndarray) -> CIR_Params:
        sigma_sqrt = np.sqrt(sigma_t[:-1])
        y = np.diff(sigma_t) / sigma_sqrt
        x1 = 1.0 / sigma_sqrt
        x2 = sigma_sqrt
        X = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)

        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)

        ab = reg.coef_[0]
        a = -reg.coef_[1]
        b = ab / a

        y_hat = reg.predict(X)
        c = np.std(y - y_hat)
        return CIR_Params(mean_reversion=a, asymptotic_mean=b, std_dev=c)


class Constant_Processes:
    def __init__(
        self,
        intervals: int,
        constants: Union[float, tuple[float, ...]],
        n_procs: Optional[int] = None,
    ):
        self.intervals = intervals
        self.constants = constants
        self._n_procs = n_procs
        self._n_procs_ = self._get_n_procs()

    @property
    def n_procs(self) -> int:
        return self._n_procs_

    # TODO random state was in the example as argument, but not used
    def get_proc(self, random_state: Optional[int] = None) -> np.ndarray:
        if isinstance(self.constants, List):
            return (
                np.repeat(self.constants, self.intervals, axis=0)
                .reshape(-1, self.intervals)
                .T
            )
        return self.constants * np.ones((self.intervals, self._n_procs_))

    def _get_n_procs(self) -> int:
        if isinstance(self.constants, tuple):
            if self._n_procs is not None and len(self.constants) != self._n_procs:
                raise ValueError(
                    "If constants is tuple, n_procs must match tuple length."
                )
            else:
                return len(self.constants)

        elif self._n_procs is None:
            raise ValueError("If constants is not tuple, n_procs cannot be None.")
        return self._n_procs


class Constant_Drift(Constant_Processes):
    def __init__(
        self,
        intervals: int,
        mu_constants: Union[float, tuple[float, ...]],
        n_procs: Optional[int] = None,
    ) -> None:
        super().__init__(intervals, mu_constants, n_procs)
        self.mu_constants = mu_constants
        self.intevals = intervals
        self._n_procs = n_procs

    @property
    def sample_size(self) -> int:
        return self.intervals

    @property
    def n_procs(self) -> int:
        return super().n_procs

    def get_mu(self, random_state: Optional[int] = None) -> np.ndarray:
        return super().get_proc(random_state)


class Constant_Sigma(Constant_Processes):
    def __init__(
        self,
        intervals: int,
        sigma_constants: Union[float, tuple[float, ...]],
        n_procs: Optional[int] = None,
    ) -> None:
        super().__init__(intervals, sigma_constants, n_procs)
        self.sigma_constants = sigma_constants
        self.intervals = intervals
        self._n_procs = n_procs

    @property
    def sample_size(self) -> int:
        return self.intervals

    @property
    def n_procs(self) -> int:
        return super().n_procs

    def get_sigma(self, random_state: Optional[int] = None) -> np.ndarray:
        return super().get_proc(random_state)


class Random_Init_P:
    def __init__(self, lower_bound: float, upper_bound: float, n_procs: int) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._n_procs = n_procs

        self._validate_bounds()

    @property
    def n_procs(self) -> int:
        return self._n_procs

    def get_P_0(self, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)

        random_vec = rng.random(self._n_procs)

        return (self.upper_bound - self.lower_bound) * random_vec + self.lower_bound

    def _validate_bounds(self) -> Optional[NoReturn]:
        if self.lower_bound <= 0 or self.upper_bound <= 0:
            raise ValueError("bounds have to be strictly positive.")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("upper bound has to be larger than lower_bound.")
        return None


class Data_Init_P:
    def __init__(self, P_data: np.ndarray, last_P: bool = True) -> None:
        self.P_data = P_data
        self.last_P = last_P

    @property
    def n_procs(self) -> int:
        shape_val = len(self.P_data.shape)
        return self.P_data.shape[1] if shape_val > 1 else 1

    def get_P_0(self, random_state: Optional[int] = None) -> np.ndarray:
        if self.last_P:
            row_idx = -1
        else:
            row_idx = 0
        return self.P_data[row_idx, :] if self.n_procs > 1 else self.P_data[row_idx]


class Generic_Geometric_Brownian_Motion:
    def __init__(
        self,
        drift: Drift,
        sigma: Sigma,
        init_P: Init_P,
        rho: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.drift = drift
        self.sigma = sigma
        self.init_P = init_P
        self.rho = rho
        # TODO change property name to seed or random_state everywhere
        self.random_state = random_state

        self._validate_drift_sigma_init_P()
        self.intervals, self.n_procs = self.drift.sample_size, self.drift.n_procs
        self.brownian_motion = Brownian_Motion(seed=random_state)

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self._rho = rho

    @property
    def random_state(self) -> Union[float, int]:
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Union[float, int]) -> None:
        self._random_state = random_state

    def get_P(self) -> np.ndarray:
        # TODO / NOTE random state gets set on sigma, drift, init b/c they all have their own BM
        sigmas = self.sigma.get_sigma()
        P_0s = self.init_P.get_P_0()

        time_integrals = self._get_time_integrals(sigmas, self.random_state)
        # NOTE this eventually calls base BM, which expects random_state as class property
        # So shouldn't _get_time_integrals just use the class property?
        W_integrals = self._get_W_integrals(sigmas)

        return P_0s[None, :] * np.exp(time_integrals + W_integrals)

    def _get_time_integrals(
        self, sigmas: np.ndarray, random_state: Optional[int]
    ) -> np.ndarray:
        mus = self.drift.get_mu(random_state)
        integrals = np.cumsum(mus - sigmas**2 / 2, axis=0)
        return np.insert(integrals, 0, np.zeros(mus.shape[1]), axis=0)[:-1]

    def _get_W_integrals(self, sigmas: np.ndarray) -> np.ndarray:
        dWs = self.brownian_motion.get_corr_dW_matrix(
            self.intervals, self.n_procs, self.rho
        )
        integrals = np.cumsum(sigmas * dWs, axis=0)
        return np.insert(integrals, 0, np.zeros(dWs.shape[1]), axis=0)[:-1]

    def _validate_drift_sigma_init_P(self) -> Optional[NoReturn]:
        if (
            self.drift.n_procs != self.sigma.n_procs
            or self.drift.n_procs != self.init_P.n_procs
        ):
            raise ValueError(
                "n_procs for both drift, sigma and init_P has to be the same!"
            )
        elif self.drift.sample_size != self.sigma.sample_size:
            raise ValueError(
                "sample size T for both drift and sigma has to be the same!"
            )
        return None

    def _estimate_params_base(
        self,
        process_matrix: np.ndarray,
        rolling_window: int,
        stat_func: Callable,
        estimate_params_func: Callable,
    ):
        diffusion_increments = np.diff(process_matrix, axis=0) / process_matrix[:-1, :]

        rolled_increments = np.lib.stride_tricks.sliding_window_view(
            diffusion_increments, rolling_window, axis=0
        )

        rolling_stat = stat_func(rolled_increments, axis=-1)

        return [
            estimate_params_func(rolling_stat[:, i])
            for i in range(rolling_stat.shape[1])
        ]

    def estimate_drift_OU_params(
        self, process_matrix: np.ndarray, rolling_window: int
    ) -> List[Stochastic_Params_Base]:
        return self._estimate_params_base(
            process_matrix,
            rolling_window,
            stat_func=np.mean,
            estimate_params_func=self.drift.OU_process.estimate_params,
        )

    def estimate_sigma_CIR_params(
        self, process_matrix: np.ndarray, rolling_window: int
    ) -> List[Stochastic_Params_Base]:
        return self._estimate_params_base(
            process_matrix,
            rolling_window,
            stat_func=np.std,
            estimate_params_func=self.sigma.CIR_process.estimate_params,
        )

    def _estimate_correlation_base(
        self, process_matrix: np.ndarray, rolling_window: int, stat_func: Callable
    ) -> float:
        diffusion_increments = np.diff(process_matrix, axis=0) / process_matrix[:-1, :]

        rolled_increments = np.lib.stride_tricks.sliding_window_view(
            diffusion_increments, rolling_window, axis=0
        )
        rolling_mus = stat_func(rolled_increments, axis=-1)
        corr_mat = np.corrcoef(rolling_mus, rowvar=False)

        np.fill_diagonal(corr_mat, np.nan)
        return float(np.nanmean(corr_mat))

    def estimate_drift_correlation(self, process_matrix: np.ndarray, rolling_window: int):

        return self._estimate_correlation_base(process_matrix, rolling_window, stat_func=np.mean)

    def estimate_sigma_correlation(self, process_matrix: np.ndarray, rolling_window: int):

        return self._estimate_correlation_base(process_matrix, rolling_window, stat_func=np.std)

class OU_Drift:
    def __init__(
        self,
        intervals: int,
        OU_params: Union[Stochastic_Params_Base, List[Stochastic_Params_Base]],
        n_procs: Optional[int] = None,
        rho: Optional[float] = None,
        seed: Optional[float] = None,
    ) -> None:
        self.intervals = intervals
        self._n_procs = n_procs
        self.rho = rho

        self.OU_process = OU_Process(seed, OU_params)

        self._n_procs_ = self._get_n_procs()

    @property
    def sample_size(self) -> int:
        return self.intervals

    @property
    def n_procs(self) -> int:
        return self._n_procs_

    def get_mu(self, random_state: Optional[int] = None) -> np.ndarray:
        return self.OU_process.create_correlated_sims(
            self.intervals, self._n_procs_, self.rho
        )

    def _get_n_procs(self) -> int:
        if isinstance(self.OU_process.model_params, list):
            return len(self.OU_process.model_params)
        elif self._n_procs is None:
            raise ValueError("If OU_params is not list, n_procs cannot be None.")
        return self._n_procs


class CIR_Sigma:
    def __init__(
        self,
        intervals: int,
        CIR_params: Union[CIR_Params, List[CIR_Params]],
        n_procs: Optional[int] = None,
        rho: Optional[float] = None,
        seed: Optional[float] = None,
    ) -> None:
        self.intervals = intervals
        self._n_procs = n_procs
        self.rho = rho

        self.CIR_process = CIR_Process(seed, CIR_params)
        self._n_procs_ = self._get_n_procs()

    @property
    def sample_size(self) -> int:
        return self.intervals

    @property
    def n_procs(self) -> int:
        return self._n_procs_

    def get_sigma(self) -> np.ndarray:
        return self.CIR_process.create_correlated_sims(
            self.intervals, self._n_procs_, self.rho
        )

    def _get_n_procs(self) -> int:
        if isinstance(self.CIR_process.model_params, list):
            return len(self.CIR_process.model_params)
        elif self._n_procs is None:
            raise ValueError("If CIR_params is not list, n_procs cannot be None.")
        return self._n_procs
