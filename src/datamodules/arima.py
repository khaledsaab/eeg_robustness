from bisect import bisect
from types import SimpleNamespace
from pytz import all_timezones_set
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional


from functools import partial

from statsmodels.tsa.arima_process import arma_generate_sample


TIME_FEATURES = {
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year",
}

FREQS = {
    "S": "second",
    "T": "minute",
    "H": "hour",
    "D": "day",
    "W": "week",
    "M": "month",
    "Q": "quarter",
    "Y": "year",
}


class TimeSeriesHelper:
    @staticmethod
    def generate_timestamps(n_timestamps, freq):
        # Generate timestamps with freq in pandas
        times = pd.date_range(start=0, periods=n_timestamps, freq=freq)
        return times

    @staticmethod
    def timestamp_to_features(timestamp, pad=0):
        if pad == 0:
            return {
                k: torch.tensor(np.array(getattr(timestamp, k)), dtype=torch.long) + 1
                for k in TIME_FEATURES
            }
        else:
            return {
                k: torch.cat(
                    [
                        torch.zeros(pad, dtype=torch.long) - 1,
                        torch.tensor(np.array(getattr(timestamp, k)), dtype=torch.long)
                        + 1,
                    ]
                )
                for k in self.TIME_FEATURES
            }

    @staticmethod
    def train_test_split(ts, lag, horizon, n_test=1, gap=0, ts_times=None):
        """
        Split the time series into train and test series.
        The test set will contain data for n_test time series.

        Args:
            ts: time series to split
            lag: lag of the time series
            horizon: forecast horizon
            n_test: number of time series in test
            gap: gap between train and test
            ts_times: timestamps of the time series
        """
        assert len(ts) == len(
            ts_times
        ), "Time series and timestamps must have the same length."
        assert lag + horizon <= len(
            ts
        ), "Lag + horizon must be smaller than the length of the time series."
        assert n_test > 0, "Number of time series in test must be greater than 0."

        cutoff = horizon + n_test + lag - 1
        assert cutoff <= len(
            ts
        ), "Cutoff must be smaller than the length of the time series. Check the values of lag, horizon, and n_test."

        train_ts, test_ts = ts[: -cutoff - gap], ts[-cutoff:]

        train_ts_times = None
        test_ts_times = None
        if ts_times is not None:
            train_ts_times = ts_times[: -cutoff - gap]
            test_ts_times = ts_times[-cutoff:]

        return SimpleNamespace(
            train_ts=train_ts,
            test_ts=test_ts,
            train_ts_times=train_ts_times,
            test_ts_times=test_ts_times,
        )

    @staticmethod
    def train_test_split_all(ts, lag, horizon, n_test=1, gap=0, ts_times=None):
        if ts_times is None:
            ts_times = [None] * len(ts)

        train_ts = []
        test_ts = []
        train_ts_times = []
        test_ts_times = []

        for ts_, ts_times_ in zip(ts, ts_times):
            splits = TimeSeriesHelper.train_test_split(
                ts_, lag, horizon, n_test, gap, ts_times_
            )
            train_ts.append(splits.train_ts)
            test_ts.append(splits.test_ts)
            train_ts_times.append(splits.train_ts_times)
            test_ts_times.append(splits.test_ts_times)

        return SimpleNamespace(
            train_ts=train_ts,
            test_ts=test_ts,
            train_ts_times=train_ts_times,
            test_ts_times=test_ts_times,
        )

    @staticmethod
    def split_x_y(ts, lag, horizon, ts_times=None):
        """
        Split the time series into x and y.
        x is the time series with lag.
        y is the time series with horizon.

        Args:
            ts: time series to split
            lag: lag of the time series
            horizon: forecast horizon
            ts_times: timestamps of the time series
        """

        x, y = ts[:-horizon], ts[lag:]
        x_times, y_times = None, None
        if ts_times is not None:
            x_times, y_times = ts_times[:-horizon], ts_times[lag:]

        return SimpleNamespace(
            x=x,
            y=y,
            x_times=x_times,
            y_times=y_times,
        )


class ForecastingSynthetic(Dataset):
    def __init__(self, n_ts, nobs_per_ts, seed):
        super().__init__()

        self.n_ts = n_ts
        self.nobs_per_ts = nobs_per_ts
        self.seed = seed

        np.random.seed(seed)
        self._setup_process()
        self.ts = self.generate()

    def _setup_process(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return self.n_ts

    def __getitem__(self, idx):
        return self.ts[idx]


class ARIMASynthetic(ForecastingSynthetic):
    """
    Generate data from an ARIMA(p, d, q) process.
    Internally, generarates data using an ARMA(p + d, q) process with d unit roots.
    """

    def __init__(
        self,
        p,
        d,
        q,
        n_ts,
        nobs_per_ts,
        c=0,
        scale=1.0,
        seed=42,
        a=1.0,
    ):
        # print("Init called")
        self.p = p
        self.d = d
        self.q = q
        self.c = c  # constant offset term in the ARIMA equation
        self.a = a
        self.scale = scale

        super().__init__(n_ts, nobs_per_ts, seed=seed)

    @staticmethod
    def _sample_complex_unit_circle(n, a=1):
        r = np.sqrt(np.random.rand(n)) * a
        theta = np.random.rand(n) * 2 * np.pi * a
        return r * np.cos(theta) + 1j * r * np.sin(theta)

    def _setup_process(self):
        np.random.seed(self.seed)

        # Construct complex-conjugate roots inside the unit circle for the ARIMA process
        # Both the AR / MA characteristic polynomials should satisfy this
        ar_roots = self._sample_complex_unit_circle(self.p // 2, a=self.a)
        ma_roots = self._sample_complex_unit_circle(self.q // 2)

        # Add unit roots as ARIMA(p, d, q) = ARMA(p + d, q)
        unit_roots = [1.0] * self.d

        # print("Constructing ARIMA(%d, %d, %d) process..." % (self.p, self.d, self.q))
        # print("AR roots:", ar_roots)
        # print("MA roots:", ma_roots)
        # print("Unit roots (multiplicity):", self.d)

        if self.p % 2 == 0:
            # Just keep the complex roots and add in the unit roots
            ar_roots = np.r_[ar_roots, ar_roots.conj(), unit_roots]
        else:
            # Add a real root to the p - 1 complex roots, as well as the unit roots
            ar_roots = np.r_[
                ar_roots, ar_roots.conj(), 2 * np.random.rand(1) - 1, unit_roots
            ]

        if self.q % 2 == 0:
            ma_roots = np.r_[ma_roots, ma_roots.conj()]
        else:
            # Add a real root to the q - 1 complex roots
            ma_roots = np.r_[ma_roots, ma_roots.conj(), np.random.rand(1)]

        # Construct the polynomial coefficients from the roots
        # Coefficients of c[0] * z^n + c[1] * z^(n-1) + ... + c[n]
        # with c[0] always equal to 1.
        ar_coeffs = np.poly(ar_roots)
        ma_coeffs = np.poly(ma_roots)

        # ar_coeffs = np.array([1, -2*np.cos(2*np.pi*0.1), 1])
        # ma_coeffs = np.array([1, -np.cos(2*np.pi*0.1)])

        self.arparams = np.r_[ar_coeffs]
        self.maparams = np.r_[ma_coeffs]

        # print("AR coefficients:", self.arparams)
        # print("MA coefficients:", self.maparams)

    def generate(self):
        ts = []
        for _ in range(self.n_ts):
            y = arma_generate_sample(
                self.arparams,
                self.maparams,
                self.nobs_per_ts,
                scale=1,
                distrvs=partial(np.random.normal, loc=self.c, scale=self.scale),
            )
            ts.append(y)
        #ts = np.array(ts)

        return ts


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        ts,
        ts_times,
        lag,
        horizon,
        n_ts,
        standardization=None,
        normalize=True,
        classification=False,
    ):
        self.ts = ts
        self.ts_times = ts_times
        self.n_ts = n_ts
        self.lag = self.min_lag = lag
        self.horizon = self.forecast_horizon = horizon
        self.standardization = standardization
        self.normalize = normalize
        self.classification = classification

        self.init()

    def init(self):
        # Split all the time series
        self.x, self.y, self.x_times, self.y_times, self.labels = [], [], [], [], []
        for idx, (ts, ts_times) in enumerate(zip(self.ts, self.ts_times)):
            _split_ts = TimeSeriesHelper.split_x_y(ts, self.lag, self.horizon, ts_times)
            self.x.append(_split_ts.x)
            self.y.append(_split_ts.y)
            self.x_times.append(_split_ts.x_times)
            self.y_times.append(_split_ts.y_times)
            # each TS class has n_ts time series, so for n_ts belong to class 0, etc.
            self.labels.append(int(idx/self.n_ts)) 

        self.n_examples = [max(0, len(ts) - self.horizon + 1) for ts in self.y]
        self._cume = np.cumsum([0] + self.n_examples[:-1])

        if self.standardization is None:
            if self.normalize:
                means = [np.mean(ts) for ts in self.ts]
                stds = [np.std(ts) for ts in self.ts]
            else:
                means = [0 for ts in self.ts]
                stds = [1 for ts in self.ts]

            self.standardization = dict(means=means, stds=stds)

    def __len__(self):
        return sum(self.n_examples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        # Bisect to find the time series to pull out
        loc = bisect(self._cume, idx) - 1

        context = self.x[loc][
            max(0, idx - self._cume[loc] - (self.lag - self.min_lag)) : idx
            - self._cume[loc]
            + self.min_lag
        ]
        context_time = self.x_times[loc][
            max(0, idx - self._cume[loc] - (self.lag - self.min_lag)) : idx
            - self._cume[loc]
            + self.min_lag
        ]

        target = self.y[loc][
            idx - self._cume[loc] : idx - self._cume[loc] + self.horizon
        ]
        target_time = self.y_times[loc][
            idx - self._cume[loc] : idx - self._cume[loc] + self.horizon
        ]

        # Normalize
        context = (context - self.standardization["means"][loc]) / self.standardization[
            "stds"
        ][loc]
        target = (target - self.standardization["means"][loc]) / self.standardization[
            "stds"
        ][loc]

        # To tensors
        context, target = torch.tensor(context, dtype=torch.float).unsqueeze(
            1
        ), torch.tensor(target, dtype=torch.float).unsqueeze(1)
        actual_lag = context.shape[0]

        # Add zeros to the end of the context with length equal to the forecast horizon
        context = torch.cat(
            [
                torch.zeros(self.lag - actual_lag, 1),
                context,
                torch.zeros(self.horizon, 1),
            ],
            dim=0,
        )
        time = context_time.union(target_time)

        if self.classification:
            #x = torch.cat([context,target],dim=0)
            x=context
            y = self.labels[loc]
            return x, y

        # return (
        #     context,
        #     target,
        #     TimeSeriesHelper.timestamp_to_features(time, pad=(self.lag - actual_lag)),
        #     loc,
        # )
        return context, target


class ARIMADataModule(LightningDataModule):
    def __init__(
        self,
        p=1,
        d=0,
        q=1,
        n_ts=1,
        nobs_per_ts=100,
        horizon=1,
        lag=1,
        val_gap=0,
        test_gap=0,
        seeds=[42],
        c=0,
        a=1.0, # controls range of AR poles
        scale=1.0,
        seasonal=None,
        classification=False,
        normalize=True,
        batch_size=10,
        num_workers=4,
        pin_memory=False,
        val_ratio=0.2, 
        test_ratio=0.2
    ):
        super().__init__()

        self.p = p
        self.d = d
        self.q = q
        self.n_ts = n_ts
        self.nobs_per_ts = nobs_per_ts
        self.horizon = horizon
        self.lag = lag
        self.val_gap = val_gap
        self.test_gap = test_gap
        self.seeds = seeds
        self.c = c
        self.a = a
        self.scale = scale
        self.seasonal = seasonal
        self.classification = classification
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def _process_seasonality(self, ts, ts_times):
        freqs = self.seasonal.keys()
        assert all([freq in FREQS for freq in freqs]), "Invalid frequency"

        for freq in freqs:
            seasonal_process = ARIMASynthetic(
                p=self.seasonal[freq]["p"],
                d=self.seasonal[freq]["d"],
                q=self.seasonal[freq]["q"],
                n_ts=self.n_ts,
                nobs_per_ts=self.nobs_per_ts,
                seed=self.seasonal[freq]["seed"],
                c=self.seasonal[freq]["c"],
                a=self.a,
                scale=self.seasonal[freq]["scale"],
            )
            seasonal_ts = seasonal_process.generate()
            seasonal_times = [
                TimeSeriesHelper.generate_timestamps(self.nobs_per_ts, freq)
                for _ in range(self.n_ts)
            ]

            # Add values from the seasonal_ts to ts using the timestamps
            for i in range(self.n_ts):
                for j, timestamp in enumerate(ts_times[i]):
                    try:
                        ts[i][j] += seasonal_ts[i][
                            np.where(seasonal_times[i] == timestamp)[0][0]
                        ]
                    except IndexError:
                        pass

        return ts, ts_times

    def setup(self,stage: Optional[str] = None):
        # Generate synthetic data from ARIMA(p, d, q)
        ts = []
        ts_times = []
        for seed in self.seeds:
            process = ARIMASynthetic(
                p=self.p,
                d=self.d,
                q=self.q,
                n_ts=self.n_ts,
                nobs_per_ts=self.nobs_per_ts,
                seed=seed,
                c=self.c,
                a=self.a,
                scale=self.scale,
            )
            self.process = process
            ts_ = process.generate()
            ts_times_ = [
                TimeSeriesHelper.generate_timestamps(self.nobs_per_ts, freq="D")
            ] * self.n_ts

            ts.extend(ts_)
            ts_times.extend(ts_times_)
        ts = np.array(ts)

        # Add seasonal component
        if self.seasonal is not None:
            ts, ts_times = self._process_seasonality(ts, ts_times)

        self.ts = ts

        train_ratio = 1.0 - self.val_ratio - self.test_ratio
        n_val = int(np.round(self.nobs_per_ts * self.val_ratio))
        n_test = int(np.round(self.nobs_per_ts * self.test_ratio))

        self.splits = TimeSeriesHelper.train_test_split_all(
            ts,
            self.lag,
            self.horizon,
            ts_times=ts_times,
            gap=self.test_gap,
            n_test=n_test,
        )
        self.splits_val = TimeSeriesHelper.train_test_split_all(
            self.splits.train_ts,
            self.lag,
            self.horizon,
            ts_times=self.splits.train_ts_times,
            gap=self.val_gap,
            n_test=n_val,
        )

        # Wrap the time series and their timestamps in a dataset
        self.data_train = TimeSeriesDataset(
            self.splits_val.train_ts,
            self.splits_val.train_ts_times,
            self.lag,
            self.horizon,
            self.n_ts,
            normalize=self.normalize,
            classification=self.classification
        )
        self.data_val = TimeSeriesDataset(
            self.splits_val.test_ts,
            self.splits_val.test_ts_times,
            self.lag,
            self.horizon,
            self.n_ts,
            standardization=self.data_train.standardization,
            normalize=self.normalize,
            classification=self.classification
        )
        self.data_test = TimeSeriesDataset(
            self.splits.test_ts,
            self.splits.test_ts_times,
            self.lag,
            self.horizon,
            self.n_ts,
            standardization=self.data_train.standardization,
            normalize=self.normalize,
            classification=self.classification
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # model = ExponentialSmoothingHoltWinters(
    #     np.random.rand(100),
    # )

    if False:
        dataset = ARIMASynthetic(1, 0, 10, 100)
        for i in range(10):
            plt.plot(dataset[i])
            plt.savefig(f"/home/workspace/arima_10_{i}.png")
            plt.close()

        dataset = ARIMASynthetic(2, 0, 10, 100)
        for i in range(10):
            plt.plot(dataset[i])
            plt.savefig(f"/home/workspace/arima_20_{i}.png")
            plt.close()

        dataset = ARIMASynthetic(2, 2, 10, 100)
        for i in range(10):
            plt.plot(dataset[i])
            plt.savefig(f"/home/workspace/arima_22_{i}.png")
            plt.close()

    # dataset = ARIMASynthetic(1, 0, 10, 100)
    # tsh = TimeSeriesHelper()
    # tsh.train_test_split(dataset[0], lag=1, horizon=1, n_test=1, gap=0)
    # tsh._generate_timestamps(ts=dataset[0], freq='D')

    # _test_synthetic_seasonal()
    # breakpoint()

    seasonal_params = {"M": {"p": 0, "d": 1, "q": 0, "c": 1, "seed": 42, "scale": 0.1}}
    # (p, d, q, c) params
    params = [
        # (0, 0, 0, 0.0), # white noise
        # (0, 1, 0, 0.0), # random walk
        # (0, 1, 1, 0.0), # exponential smoothing
        # (0, 2, 2, 0.0), # double exponential smoothing
        # (0, 1, 2, 0.0), # damped Holt's model
        (1, 0, 0, 1.0),
        (2, 0, 1, 1.0),
        (3, 0, 2, 1.0),
        (1, 1, 0, 1.0),
        (2, 1, 1, 1.0),
        (3, 1, 2, 1.0),
    ]

    for p, d, q, c in params:
        print("Running:")
        print("p: {}, d: {}, q: {}, c: {}, seasonal: {}".format(p, d, q, c, "none"))
        all_test_rmse = []
        for seed in range(3):
            dataset = ARIMASyntheticDataset(
                "synthetic-arima",
                p=p,
                d=d,
                q=q,
                n_ts=20,
                nobs_per_ts=1000,
                horizon=1,
                lag=64,  # max(p + d, q),
                c=c,
                seed=seed,
                scale=0.1,
                # seasonal=seasonal_params
            )
            dataset.setup()

            from statsmodels.tsa.arima.model import ARIMA

            def rmse(y_true, y_pred):
                return np.sqrt(np.mean((y_true - y_pred) ** 2))

            val_rmses = []
            test_rmses = []
            for j in range(1):
                arima_mod = ARIMA(
                    dataset.splits_val.train_ts[j], order=(p, d, q), trend="n"
                )
                breakpoint()
                # arima_mod = ARIMA(dataset.splits_val.train_ts[j], order=(2, 0, 0), trend="n")
                arima_res = arima_mod.fit()

                # Validation forecasts
                val_pred = arima_res.apply(dataset.dataset_val.x[j]).forecast(
                    dataset.horizon
                )
                test_pred = arima_res.apply(dataset.dataset_test.x[j]).forecast(
                    dataset.horizon
                )
                # print(dataset.dataset_val.x[j], dataset.dataset_val.y[j])
                # print(dataset.dataset_test.x[j], dataset.dataset_test.y[j])

                val_rmses.append(rmse(dataset.dataset_val.y[j], val_pred))
                test_rmses.append(rmse(dataset.dataset_test.y[j], test_pred))

            # print(f'p={p}, d={d}, q={q}, c={c}')
            # print(f'Validation RMSE: {np.mean(val_rmses)}')
            # print(f'Test RMSE: {np.mean(test_rmses)}')
            all_test_rmse.append(np.mean(test_rmses))

        print("average test RMSE: {}".format(np.mean(all_test_rmse)))

        # for i in range(1):
        #     plt.plot(dataset.ts[i])
        #     plt.savefig(f'/home/ksaab/hippo_dev/hippo/scratch/khaled/arima_forecasts/arima_{p}_{d}_{q}_{c}_{i}.png')
        #     plt.close()

    # for i in range(10):
    #     plt.plot(dataset.dataset_val[i][0])
    #     plt.savefig(f'/home/workspace/arima_val_{i}.png')
    #     plt.close()

    # for i in range(10):
    #     plt.plot(dataset.dataset_test[i][0])
    #     plt.savefig(f'/home/workspace/arima_test_{i}.png')
    #     plt.close()

    # breakpoint()
