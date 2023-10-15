import copy
import numpy as np
import pandas as pd
from pytest_mock import mocker
import pytest
from scipy.stats import pearsonr

from src.stochastic_process_base import (
    Brownian_Motion,
    Stochastic_Params_Base,
    OU_Process,
    CIR_Process,
    CIR_Params,
    Constant_Processes,
    Constant_Drift,
    Constant_Sigma,
    Random_Init_P,
    Data_Init_P,
    Generic_Geometric_Brownian_Motion,
    OU_Drift,
    CIR_Sigma,
)

from src.stocastic_interfaces import Drift, Sigma, Init_P


STOCHASTIC_BASE_PATH = "src.stochastic_process_base"


class Test_Helpers:
    def get_avg_corr(sim_arr):
        return (
            pd.DataFrame(np.corrcoef(np.diff(sim_arr, axis=0), rowvar=False))
            .reset_index(drop=False)
            .melt(id_vars=["index"])
            .groupby("variable")
            .agg({"value": "mean"})["value"]
            .mean()
        )

    def get_avg_variance(corr_matrix):
        return (
            pd.DataFrame(np.cumsum(corr_matrix, axis=0))
            .reset_index(drop=False)
            .melt(id_vars=["index"])
            .groupby(["index"])
            .var()
            .mean()[0]
        )


class Test_Brownian_Motion:
    def test_brownian_motion_change(self):
        stoch_instance = Brownian_Motion(seed=None)

        process = stoch_instance.get_dW(intervals=10000)

        assert len(process) == 10000
        # Abs b/c mean is 0
        assert process.mean() == pytest.approx(0, abs=0.05)
        assert process.std() == pytest.approx(1, rel=0.1)

    def test_brownian_motion_basic(self):
        stoch_instance = Brownian_Motion(seed=None)

        process = stoch_instance.get_W(intervals=100)

        assert len(process) == 100
        assert len(process[np.isnan(process)]) == 0

    def test_brownian_motion_seed(self):
        seed_same = 10
        stoch_instance = Brownian_Motion(seed=seed_same)
        process_1 = stoch_instance.get_W(intervals=100)

        stoch_instance_2 = Brownian_Motion(seed=seed_same)
        process_2 = stoch_instance_2.get_W(intervals=100)

        assert all(process_1 == process_2)

        stoch_instance_3 = Brownian_Motion(seed=seed_same * 2)
        process_3 = stoch_instance_3.get_W(intervals=100)

        assert process_3.sum() != process_1.sum()

    def test_brownian_motion_corr(self):
        seed_same = 10
        stoch_instance = Brownian_Motion(seed=seed_same)
        org_process = stoch_instance.get_dW(intervals=100)

        corr_val = 0.5
        corr_process_1 = stoch_instance._get_correlated_dW(org_process, corr_val)

        assert org_process.mean() != corr_process_1.mean()

        corr_val = 1
        corr_process_2 = stoch_instance._get_correlated_dW(org_process, corr_val)

        assert all(org_process == corr_process_2)

        corr_val = -1
        corr_process_3 = stoch_instance._get_correlated_dW(org_process, corr_val)
        all(org_process * -1 == corr_process_3)

    def test_brownian_multiple_corr(self):
        seed = 123
        stoch_instance = Brownian_Motion(seed=seed)

        corr_matrix_closer = stoch_instance.get_corr_dW_matrix(
            intervals=100, n_procs=50, rho=0.9
        )

        avg_variance_closer = Test_Helpers.get_avg_variance(corr_matrix_closer)

        corr_matrix_farther = stoch_instance.get_corr_dW_matrix(
            intervals=100, n_procs=50, rho=0.1
        )

        avg_variance_farther = Test_Helpers.get_avg_variance(corr_matrix_farther)

        assert avg_variance_closer < avg_variance_farther


class Test_OU_Process:
    def test_get_ou_process(self):
        intervals = 1000
        ou_params = Stochastic_Params_Base(
            mean_reversion=0.07, asymptotic_mean=0.01, std_dev=0.001
        )
        ou_proc = OU_Process(seed=12345, param_obj=ou_params)

        dW = ou_proc.brownian_motion.get_dW(intervals)

        ou_sim = ou_proc.create_sim(intervals, ou_proc.model_params, dW)

        assert len(ou_sim) == intervals

        assert ou_sim[0] == pytest.approx(ou_params.asymptotic_mean, rel=0.2)
        assert ou_sim.mean() == pytest.approx(ou_params.asymptotic_mean, rel=0.1)
        assert ou_sim.std() == pytest.approx(ou_params.std_dev, rel=2)

    def test_get_ou_estimation(self):
        intervals = 1000
        ou_params = Stochastic_Params_Base(
            mean_reversion=0.1, asymptotic_mean=0.2, std_dev=0.05
        )
        ou_proc = OU_Process(seed=6789, param_obj=ou_params)

        dW = ou_proc.brownian_motion.get_dW(intervals)

        ou_sim = ou_proc.create_sim(intervals, ou_proc.model_params, dW)

        ou_params_est = ou_proc.estimate_params(ou_sim)

        assert ou_params_est.mean_reversion == pytest.approx(
            ou_params.mean_reversion, rel=0.25
        )
        assert ou_params_est.asymptotic_mean == pytest.approx(
            ou_params.asymptotic_mean, rel=0.05
        )
        assert ou_params_est.std_dev == pytest.approx(ou_params.std_dev, rel=0.05)

    def test_ou_corr_single(self):
        intervals = 1000
        ou_params = Stochastic_Params_Base(
            mean_reversion=0.4, asymptotic_mean=4, std_dev=3
        )

        ou_proc = OU_Process(seed=91234, param_obj=ou_params)

        corr = 0.9
        n_proc = 10

        ou_sims = ou_proc.create_correlated_sims(intervals, n_proc, corr)

        assert (intervals, n_proc) == ou_sims.shape

        larger_corr = Test_Helpers.get_avg_corr(ou_sims)

        ou_sims_less_corr = ou_proc.create_correlated_sims(intervals, n_proc, corr / 2)

        smaller_corr = Test_Helpers.get_avg_corr(ou_sims_less_corr)

        assert larger_corr > smaller_corr

    def test_ou_corr_multiple(self):
        intervals = 1000

        ou_param_list = [
            Stochastic_Params_Base(mean_reversion=0.1, asymptotic_mean=4, std_dev=3)
        ]
        for i in np.arange(0.05, 0.25, 0.05):
            new_param = copy.deepcopy(ou_param_list[0])

            new_param.mean_reversion += i
            new_param.asymptotic_mean += i
            new_param.std_dev += i

            ou_param_list.append(new_param)

        higher_corr = 0.9
        ou_proc = OU_Process(seed=91234, param_obj=ou_param_list)

        ou_sims = ou_proc.create_correlated_sims(
            intervals, n_procs=None, proc_correlation=higher_corr
        )

        assert (intervals, len(ou_param_list)) == ou_sims.shape

        lower_corr = 0.2
        ou_sims_lower = ou_proc.create_correlated_sims(
            intervals, n_procs=None, proc_correlation=lower_corr
        )
        assert (intervals, len(ou_param_list)) == ou_sims_lower.shape

        higher_corr_est = Test_Helpers.get_avg_corr(ou_sims)
        lower_corr_est = Test_Helpers.get_avg_corr(ou_sims_lower)

        assert higher_corr_est > lower_corr_est

        assert higher_corr == pytest.approx(higher_corr, rel=0.1)
        assert lower_corr == pytest.approx(lower_corr, rel=0.1)


class Test_CIR_Process:
    def test_single_sim(self):
        intervals = 1000
        cir_params = CIR_Params(
            mean_reversion=0.06, asymptotic_mean=0.01, std_dev=0.009
        )
        cir_proc = CIR_Process(seed=12345, param_obj=cir_params)

        dW = cir_proc.brownian_motion.get_dW(intervals)
        cir_sims = cir_proc.create_sim(intervals, cir_proc.model_params, dW)

        assert len(cir_sims) == intervals

        assert cir_sims[0] == pytest.approx(cir_params.asymptotic_mean, rel=0.01)
        assert cir_sims.mean() == pytest.approx(cir_params.asymptotic_mean, rel=0.01)
        assert cir_sims.std() == pytest.approx(cir_params.std_dev, rel=0.75)

    def test_estimate_cir_params(self):
        intervals = 1000
        CIR_params = CIR_Params(mean_reversion=0.05, asymptotic_mean=0.5, std_dev=0.02)
        cir_proc = CIR_Process(seed=12345, param_obj=CIR_params)

        dW = cir_proc.brownian_motion.get_dW(intervals)
        cir_sim = cir_proc.create_sim(intervals, cir_proc.model_params, dW)

        cir_param_est = cir_proc.estimate_params(cir_sim)

        assert cir_param_est.mean_reversion == pytest.approx(
            CIR_params.mean_reversion, rel=0.35
        )
        assert cir_param_est.asymptotic_mean == pytest.approx(
            CIR_params.asymptotic_mean, rel=0.01
        )
        assert cir_param_est.std_dev == pytest.approx(CIR_params.std_dev, rel=0.05)

    def test_corr_cir_process_single(self):
        intervals = 1000
        CIR_params = CIR_Params(
            mean_reversion=0.06, asymptotic_mean=0.01, std_dev=0.009
        )
        cir_proc = CIR_Process(seed=12345, param_obj=CIR_params)

        higher_corr = 0.9
        cir_sims_corr = cir_proc.create_correlated_sims(
            intervals, n_procs=5, proc_correlation=higher_corr
        )

        lower_corr = 0.2
        cir_sims_corr_less = cir_proc.create_correlated_sims(
            intervals, n_procs=5, proc_correlation=lower_corr
        )

        higher_corr_est = Test_Helpers.get_avg_corr(cir_sims_corr)
        lower_corr_est = Test_Helpers.get_avg_corr(cir_sims_corr_less)
        assert higher_corr_est > lower_corr_est

        assert higher_corr == pytest.approx(higher_corr, rel=0.1)
        assert lower_corr == pytest.approx(lower_corr, rel=0.1)

    def test_corr_cir_multiple(self):
        intervals = 1000

        cir_param_list = [
            CIR_Params(mean_reversion=0.06, asymptotic_mean=0.01, std_dev=0.009)
        ]
        for i in np.arange(0.01, 0.05, 0.01):
            new_param = copy.deepcopy(cir_param_list[0])

            new_param.mean_reversion += i
            new_param.std_dev += i / 10

            cir_param_list.append(new_param)

        cir_proc = CIR_Process(seed=91234, param_obj=cir_param_list)
        higher_corr = 0.9

        cir_sims = cir_proc.create_correlated_sims(
            intervals, n_procs=None, proc_correlation=higher_corr
        )

        assert (intervals, len(cir_param_list)) == cir_sims.shape

        lower_corr = 0.2
        cir_sims_lower = cir_proc.create_correlated_sims(
            intervals, n_procs=None, proc_correlation=lower_corr
        )
        assert (intervals, len(cir_param_list)) == cir_sims.shape

        higher_corr_est = Test_Helpers.get_avg_corr(cir_sims)
        lower_corr_est = Test_Helpers.get_avg_corr(cir_sims_lower)

        assert higher_corr_est > lower_corr_est

        assert higher_corr == pytest.approx(higher_corr, rel=0.1)
        assert lower_corr == pytest.approx(lower_corr, rel=0.1)


class Test_Constant_Process:
    def inspect_constant_params(self, proc_random, proc_non_random, intervals, params):
        assert len(proc_non_random) == len(proc_random) == intervals
        assert np.array_equal(proc_non_random, proc_random)

        for proc in range(0, len(params)):
            set(proc_non_random[:, proc]) == set(proc_random[:, proc]) == {params[proc]}

    def test_single_constant(self):
        intervals = 1000
        constants = 1
        n_procs = 3

        const_proc = Constant_Processes(intervals, constants=constants, n_procs=n_procs)
        process_matrix = const_proc.get_proc()

        n_rows, n_columns = process_matrix.shape
        assert n_rows == intervals
        assert n_columns == n_procs

        for col in range(0, n_columns):
            assert set(process_matrix[:, col]) == {constants}

    def test_multiple_constant(self):
        intervals = 1000
        constants = (1.0, 2.0, 3.0)
        n_procs = 3

        const_proc = Constant_Processes(intervals, constants=constants, n_procs=n_procs)
        process_matrix = const_proc.get_proc()

        n_rows, n_columns = process_matrix.shape
        assert n_rows == intervals
        assert n_columns == n_procs

        for col in range(0, n_columns):
            assert set(process_matrix[:, col]) == {constants[col]}

    def test_fail_constants_len(self):
        expected_error = "If constants is tuple, n_procs must match tuple length."

        with pytest.raises(ValueError) as exec_info:
            Constant_Processes(1, constants=(1, 2), n_procs=4)
        assert str(exec_info.value) == expected_error

    def test_random_init_p(self):
        lower_bound = 2_000
        upper_bound = 10_000
        n_procs = 4
        random_init_P = Random_Init_P(lower_bound, upper_bound, n_procs)

        assert random_init_P.upper_bound == upper_bound
        assert random_init_P.lower_bound == lower_bound

        assert random_init_P.n_procs == n_procs

        P_0s = random_init_P.get_P_0()
        assert type(P_0s) == np.ndarray
        assert len(P_0s) == n_procs

        for p_0 in P_0s:
            assert lower_bound < p_0 and upper_bound > p_0

    def test_random_init_p_bad_bounds(self):
        expected_error = "upper bound has to be larger than lower_bound."

        with pytest.raises(ValueError) as exec_info:
            Random_Init_P(100, 50, 1)
        assert str(exec_info.value) == expected_error

    def test_random_init_p_neg_bounds(self):
        expected_error = "bounds have to be strictly positive."

        with pytest.raises(ValueError) as exec_info:
            Random_Init_P(-50, 0, 1)
        assert str(exec_info.value) == expected_error

    @pytest.mark.parametrize("last_P", [(True), (False)])
    def test_data_init_p(self, last_P):
        rand_array = np.random.random(10)
        data_init = Data_Init_P(rand_array, last_P=last_P)
        P_0s = data_init.get_P_0()
        if last_P:
            assert P_0s == rand_array[-1]
        else:
            assert P_0s == rand_array[0]

    def test_constant_sigma(self):
        intervals = 1000
        sigma_constants = (0.01, 0.02, 0.015, 0.025)
        sigma = Constant_Sigma(intervals=intervals, sigma_constants=sigma_constants)

        sigma_random = sigma.get_sigma(random_state=None)
        sigma_non_random = sigma.get_sigma(random_state=1234)

        self.inspect_constant_params(
            sigma_random, sigma_non_random, intervals, sigma_constants
        )

    def test_constant_drift(self):
        intervals = 1000
        mu_constants = (0.00014, 0.00012, -0.0002, -0.00007)
        drift = Constant_Drift(intervals=intervals, mu_constants=mu_constants)

        drift_random = drift.get_mu(random_state=None)
        drift_non_random = drift.get_mu(random_state=1234)

        self.inspect_constant_params(
            drift_random, drift_non_random, intervals, mu_constants
        )


# TODO test Generic_Geometric_Brownian_Motion
class Test_Geometric_Brownian_Motion:
    @pytest.fixture(scope="function")
    def random_state(self):
        yield 3

    @pytest.fixture(scope="function")
    def constant_geo_brownian(self, random_state):
        intervals = 1000
        mu_constants = (0.00014, 0.00012, -0.0002, -0.00007)
        sigma_constants = (0.01, 0.02, 0.015, 0.025)
        init_lower_bound = 3000
        init_upper_bound = 10000
        yield Generic_Geometric_Brownian_Motion(
            drift=Constant_Drift(intervals=intervals, mu_constants=mu_constants),
            sigma=Constant_Sigma(intervals=intervals, sigma_constants=sigma_constants),
            init_P=Random_Init_P(init_lower_bound, init_upper_bound, len(mu_constants)),
            random_state=random_state,
        )

    @pytest.fixture(scope="function")
    def sigmas(self, constant_geo_brownian, random_state):
        yield constant_geo_brownian.sigma.get_sigma(random_state)

    def test_gbm_bad_proc(self):
        expected_error = "n_procs for both drift, sigma and init_P has to be the same!"

        with pytest.raises(ValueError) as exec_info:
            Generic_Geometric_Brownian_Motion(
                drift=Constant_Drift(intervals=2, mu_constants=(1, 2)),
                sigma=Constant_Sigma(intervals=3, sigma_constants=(1, 2, 3)),
                init_P=Random_Init_P(100, 200, 4),
            )

        assert str(exec_info.value) == expected_error

    def test_constant_bm_random(self, constant_geo_brownian):
        non_random_1 = copy.deepcopy(constant_geo_brownian)
        non_random_1.random_state = 1234
        random = copy.deepcopy(constant_geo_brownian)

        non_random_2 = copy.deepcopy(constant_geo_brownian)
        non_random_2.random_state = 1234

        P_matrix_non_random_1 = non_random_1.get_P()
        P_matrix_non_random_2 = non_random_2.get_P()
        P_matrix_random = random.get_P()

        assert np.array_equal(P_matrix_non_random_1, P_matrix_non_random_2)

        assert not np.array_equal(P_matrix_non_random_1, P_matrix_random)

    def test_constant_bm_corr(self, constant_geo_brownian):
        constant_geo_brownian.rho = 0
        P_matrix_no_corr = constant_geo_brownian.get_P()

        no_corr = Test_Helpers.get_avg_corr(P_matrix_no_corr)
        # NOTE a bit curious why this is so far from 0
        assert no_corr == pytest.approx(constant_geo_brownian.rho, abs=0.25)

        constant_geo_brownian.rho = 0.8
        P_matrix_corr = constant_geo_brownian.get_P()

        high_corr = Test_Helpers.get_avg_corr(P_matrix_corr)

        assert high_corr > no_corr
        assert high_corr == pytest.approx(constant_geo_brownian.rho, abs=0.1)

    def test_get_time_integrals(self, constant_geo_brownian, sigmas, random_state):
        time_integrals = constant_geo_brownian._get_time_integrals(sigmas, random_state)

        assert type(time_integrals) == np.ndarray
        assert time_integrals.shape == (
            constant_geo_brownian.intervals,
            constant_geo_brownian.sigma.n_procs,
        )

        time_integrals_df = pd.DataFrame(time_integrals)

        for col in range(0, constant_geo_brownian.sigma.n_procs):
            constant_integral = round(time_integrals_df[col][1], 5)
            time_integrals_df[f"shift_{col}"] = time_integrals_df[col].shift(1)

            constant_change = round(
                (time_integrals_df[col] - time_integrals_df[f"shift_{col}"]), 5
            ).drop_duplicates()
            np.testing.assert_allclose(
                constant_change, [np.nan, constant_integral], equal_nan=True
            )

    def test_get_W_integrals(self, constant_geo_brownian, sigmas):
        w_integrals = constant_geo_brownian._get_W_integrals(sigmas)

        assert type(w_integrals) == np.ndarray
        assert w_integrals.shape == (
            constant_geo_brownian.intervals,
            constant_geo_brownian.sigma.n_procs,
        )

        assert not np.isnan(w_integrals).any()

    def test_get_P(self, constant_geo_brownian, sigmas, random_state, mocker):
        constant_geo_brownian.random_state = random_state

        time_integrals = constant_geo_brownian._get_time_integrals(sigmas, random_state)
        w_integrals = constant_geo_brownian._get_W_integrals(sigmas)

        P_0s = constant_geo_brownian.init_P.get_P_0(random_state)

        mocker.patch(
            f"{STOCHASTIC_BASE_PATH}.Generic_Geometric_Brownian_Motion._get_W_integrals",
            return_value=w_integrals,
        )

        processes = constant_geo_brownian.get_P()

        assert type(processes) == np.ndarray
        assert processes.shape == (
            constant_geo_brownian.intervals,
            constant_geo_brownian.sigma.n_procs,
        )

        expected_arr = np.round((time_integrals + w_integrals), 5)
        derived_arr = np.round(np.log(processes / P_0s), 5)
        np.array_equal(derived_arr, expected_arr)


class Test_Generic_Brownian_Motion:
    @pytest.fixture()
    def OU_drift(self):
        OU_params = [
            Stochastic_Params_Base(
                mean_reversion=0.0097, asymptotic_mean=0.00014, std_dev=0.00028
            ),
            Stochastic_Params_Base(
                mean_reversion=0.008, asymptotic_mean=-0.0002, std_dev=0.0003
            ),
            Stochastic_Params_Base(
                mean_reversion=0.013, asymptotic_mean=0.0, std_dev=0.00015
            ),
            Stochastic_Params_Base(
                mean_reversion=0.007, asymptotic_mean=0.0, std_dev=0.0001
            ),
        ]

        intervals = 1_000
        OU_rho = 0.6

        yield OU_Drift(intervals, OU_params, rho=OU_rho)

    @pytest.fixture()
    def CIR_sigma(self):
        CIR_params = [
            CIR_Params(mean_reversion=0.012, asymptotic_mean=0.019, std_dev=0.0025),
            CIR_Params(mean_reversion=0.013, asymptotic_mean=0.017, std_dev=0.0021),
            CIR_Params(mean_reversion=0.015, asymptotic_mean=0.021, std_dev=0.0017),
            CIR_Params(mean_reversion=0.01, asymptotic_mean=0.027, std_dev=0.0029),
        ]

        intervals = 1_000
        CIR_rho = 0.7
        yield CIR_Sigma(intervals, CIR_params, rho=CIR_rho)

    @pytest.fixture()
    def random_init_P(self, CIR_sigma):
        lower_bound = 3_000
        upper_bound = 10_000

        yield Random_Init_P(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            n_procs=CIR_sigma._n_procs_,
        )

    @pytest.fixture()
    def generic_geo_brownian(self, OU_drift, CIR_sigma, random_init_P):
        geo_rho = 0.8
        yield Generic_Geometric_Brownian_Motion(
            drift=OU_drift,
            sigma=CIR_sigma,
            init_P=random_init_P,
            rho=geo_rho,
        )

    # TODO should test multiple params
    def test_ou_drift_mu(self):
        OU_params = Stochastic_Params_Base(
            mean_reversion=0.7, asymptotic_mean=10, std_dev=0.5
        )

        # TODO for some reason I can't go over 1k intervals here?
        intervals = 1_000
        n_proc = 5
        rho = 0.7
        ou_drift = OU_Drift(
            intervals=intervals,
            OU_params=OU_params,
            n_procs=n_proc,
            rho=rho,
            seed=12345,
        )

        ou_sims = ou_drift.get_mu()

        assert (intervals, n_proc) == ou_sims.shape

        corr = Test_Helpers.get_avg_corr(ou_sims)
        assert corr == pytest.approx(rho, 0.1)

    # TODO test single param
    def test_cir_sigma(self, CIR_sigma):
        cir_sims = CIR_sigma.get_sigma()

        assert (CIR_sigma.intervals, CIR_sigma._n_procs_) == cir_sims.shape

        corr = Test_Helpers.get_avg_corr(cir_sims)
        assert corr == pytest.approx(CIR_sigma.rho, 0.1)

    def test_generic_full(self, OU_drift, generic_geo_brownian):
        P_matrix = generic_geo_brownian.get_P()

        assert (OU_drift.intervals, OU_drift._n_procs_) == P_matrix.shape

        corr = Test_Helpers.get_avg_corr(P_matrix)
        assert corr == pytest.approx(generic_geo_brownian.rho, 0.2)

    def test_estimate_drift_OU_params(self, generic_geo_brownian, OU_drift):
        P_matrix = generic_geo_brownian.get_P()

        OU_estimation = generic_geo_brownian.estimate_drift_OU_params(
            P_matrix, rolling_window=250
        )

        assert type(OU_estimation) == list
        assert len(OU_estimation) == len(OU_drift.OU_process.model_params)

        for i in range(0, len(OU_estimation)):
            assert type(OU_estimation[i]) == Stochastic_Params_Base

    def test_estimate_sigma_CIR_params(self, generic_geo_brownian, CIR_sigma):
        P_matrix = generic_geo_brownian.get_P()

        CIR_estimation = generic_geo_brownian.estimate_sigma_CIR_params(
            P_matrix, rolling_window=250
        )

        assert type(CIR_estimation) == list
        assert len(CIR_estimation) == len(CIR_sigma.CIR_process.model_params)

        for i in range(0, len(CIR_estimation)):
            assert type(CIR_estimation[i]) == CIR_Params

    def test_estimate_OU_corr(self, generic_geo_brownian, OU_drift):
        P_matrix = generic_geo_brownian.get_P()

        OU_corr_estimation = generic_geo_brownian.estimate_drift_correlation(
            P_matrix, rolling_window=100
        )

        assert type(OU_corr_estimation) == float

        assert OU_corr_estimation == pytest.approx(OU_drift.rho, 0.5)

    def test_estimate_CIR_corr(self, generic_geo_brownian, CIR_sigma):
        P_matrix = generic_geo_brownian.get_P()

        CIR_corr_estimation = generic_geo_brownian.estimate_sigma_correlation(
            P_matrix, rolling_window=100
        )

        assert type(CIR_corr_estimation) == float

        assert CIR_corr_estimation == pytest.approx(CIR_sigma.rho, 0.5)
