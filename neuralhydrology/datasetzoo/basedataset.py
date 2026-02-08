import logging
import pickle
import re
import sys
import warnings
from collections import defaultdict
from typing import List, Dict, Union, Optional, Set

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import torch
import xarray
from numba import NumbaPendingDeprecationWarning
from numba import njit, prange
from ruamel.yaml import YAML
from torch.utils.data import Dataset
from tqdm import tqdm

from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoTrainDataError, NoEvaluationDataError
from neuralhydrology.utils import samplingutils

LOGGER = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base data set class to load and preprocess data."""

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.is_train = is_train

        if period not in ["train", "validation", "test"]:
            raise ValueError("'period' must be one of 'train', 'validation' or 'test' ")
        else:
            self.period = period

        if period in ["validation", "test"]:
            if not scaler:
                raise ValueError("During evaluation of validation or test period, scaler dictionary has to be passed")

            if cfg.use_basin_id_encoding and not id_to_int:
                raise ValueError("For basin id embedding, the id_to_int dictionary has to be passed anything but train")

        if self.cfg.timestep_counter:
            if not self.cfg.forecast_inputs_flattened:
                raise ValueError('Timestep counter only works for forecast data.')
            if cfg.forecast_overlap:
                overlap_zeros = torch.zeros((cfg.forecast_overlap, 1))
                forecast_counter = torch.Tensor(
                    range(1, cfg.forecast_seq_length - cfg.forecast_overlap + 1)
                ).unsqueeze(-1)
                self.forecast_counter = torch.concatenate([overlap_zeros, forecast_counter], dim=0)
                self.hindcast_counter = torch.zeros(
                    (cfg.seq_length - cfg.forecast_seq_length + cfg.forecast_overlap, 1)
                )
            else:
                self.forecast_counter = torch.Tensor(range(1, cfg.forecast_seq_length + 1)).unsqueeze(-1)
                self.hindcast_counter = torch.zeros((cfg.seq_length - cfg.forecast_seq_length, 1))

        if basin is None:
            self.basins = utils.load_basin_file(getattr(cfg, f"{period}_basin_file"))
        else:
            self.basins = [basin]

        # Map basin id -> integer index (used for persistent LSTM state resets)
        self._basin_to_idx = {b: i for i, b in enumerate(self.basins)}

        self.additional_features = additional_features
        self.id_to_int = id_to_int
        self.scaler = scaler

        # don't compute scale when finetuning
        if is_train and not scaler:
            self._compute_scaler = True
        else:
            self._compute_scaler = False

        # check and extract frequency information from config
        self.frequencies = []
        self.seq_len = None
        self._predict_last_n = None
        self._initialize_frequency_configuration()

        # ------------------------------------------------------------------
        # Persistent / subsampling options (e.g. for persistent LSTM)
        # Defaults keep the original behavior: use every valid sample (stride=1)
        # ------------------------------------------------------------------
        self.seq_stride = getattr(cfg, "seq_stride", 1)
        self.non_overlapping_sequences = getattr(cfg, "non_overlapping_sequences", False)

        # during training we log data processing with progress bars, but not during validation/testing
        self._disable_pbar = cfg.verbose == 0 or not self.is_train

        # NEW: desired run frequency (single-frequency mode)
        self.run_frequency: Optional[str] = getattr(cfg, "run_frequency", None)  # e.g., "1H" or "15min"
        # NEW: variables that must be treated as accumulated (sum-type) when resampling.
        # Example: precipitation totals, discharge depth-per-step totals, etc.
        self.sum_variables: Set[str] = set(getattr(cfg, "sum_variables", []) or [])

        # initialize class attributes that are filled in the data loading functions
        self._x_d = {}
        self._x_s = {}
        self._attributes = {}
        self._y = {}
        self._per_basin_target_stds = {}
        self._dates = {}
        self.start_and_end_dates = {}
        self.num_samples = 0
        self.period_starts = {}  # needed for restoring date index during evaluation

        # get the start and end date periods for each basin
        self._get_start_and_end_dates()

        # if additional features files are passed in the config, load those files
        if (not additional_features) and cfg.additional_feature_files:
            self._load_additional_features()

        if cfg.use_basin_id_encoding:
            if self.is_train:
                self._create_id_to_int()

        # load and preprocess data
        self._load_data()

        if self.is_train:
            self._dump_scaler()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        basin, indices = self.lookup_table[item]

        sample = {}
        # Integer basin index so the trainer can reset state per basin (persistent LSTM)
        sample["basin_idx"] = torch.tensor(self._basin_to_idx[basin], dtype=torch.long)

        for freq, seq_len, idx in zip(self.frequencies, self.seq_len, indices):
            # if there's just one frequency, don't use suffixes.
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            hindcast_start_idx = idx + 1 - seq_len
            global_end_idx = idx + 1

            if self.cfg.forecast_seq_length:
                hindcast_end_idx = idx + 1 - self.cfg.forecast_seq_length
                forecast_start_idx = idx + 1 - self.cfg.forecast_seq_length
                if self.cfg.forecast_overlap and self.cfg.forecast_overlap > 0:
                    hindcast_end_idx += self.cfg.forecast_overlap
            else:
                hindcast_end_idx = None
                forecast_start_idx = None

            x_d_key = f'x_d{freq_suffix}'
            sample[x_d_key] = {}
            sample[f'{x_d_key}_hindcast'] = {}
            sample[f'{x_d_key}_forecast'] = {}

            for k, v in self._x_d[basin][freq].items():
                if k in self.cfg.hindcast_inputs_flattened:
                    sample[f'{x_d_key}_hindcast'][k] = v[hindcast_start_idx:hindcast_end_idx]
                if k in self.cfg.forecast_inputs_flattened:
                    sample[f'{x_d_key}_forecast'][k] = v[forecast_start_idx:global_end_idx]
                if not self.cfg.hindcast_inputs_flattened:
                    sample[x_d_key][k] = v[hindcast_start_idx:global_end_idx]

            if self.is_train and (self.cfg.nan_step_probability or self.cfg.nan_sequence_probability):
                if self.cfg.hindcast_inputs_flattened:
                    sample[f'{x_d_key}_hindcast'] = self._add_nan_streaks(
                        sample[f'{x_d_key}_hindcast'], groups=self.cfg.hindcast_inputs
                    )
                    sample[f'{x_d_key}_forecast'] = self._add_nan_streaks(
                        sample[f'{x_d_key}_forecast'], groups=self.cfg.forecast_inputs
                    )
                else:
                    sample[x_d_key] = self._add_nan_streaks(sample[x_d_key], groups=self.cfg.dynamic_inputs)

            sample[f'y{freq_suffix}'] = self._y[basin][freq][hindcast_start_idx:global_end_idx]
            sample[f'date{freq_suffix}'] = self._dates[basin][freq][hindcast_start_idx:global_end_idx]

            # check for static inputs
            static_inputs = []
            if self._attributes:
                static_inputs.append(self._attributes[basin])
            if self._x_s:
                static_inputs.append(self._x_s[basin][freq][idx])
            if static_inputs:
                sample[f'x_s{freq_suffix}'] = torch.cat(static_inputs, dim=-1)

            if self.cfg.timestep_counter:
                sample[f'x_d{freq_suffix}']['hindcast_counter'] = self.hindcast_counter
                sample[f'x_d{freq_suffix}']['forecast_counter'] = self.forecast_counter

        if self._per_basin_target_stds:
            sample['per_basin_target_stds'] = self._per_basin_target_stds[basin]
        if self.id_to_int:
            sample['x_one_hot'] = torch.nn.functional.one_hot(
                torch.tensor(self.id_to_int[basin]), num_classes=len(self.id_to_int)
            ).to(torch.float32)

        return sample

    def _add_nan_streaks(self, x_d: dict[str, torch.Tensor], groups: list[list[str]]) -> dict[str, torch.Tensor]:
        """Samples NaN streaks for each feature group."""
        if not groups or not isinstance(groups[0], list):
            raise ValueError('For dropout streaks, dynamic_inputs must be a list of lists.')
        seq_length = x_d[groups[0][0]].shape[0]
        drop_masks = np.zeros((len(groups), seq_length, 1), dtype=bool)
        drop_sequences = np.random.choice(
            [True, False],
            p=[self.cfg.nan_sequence_probability, 1 - self.cfg.nan_sequence_probability],
            size=len(groups)
        )
        if drop_sequences.all():
            drop_sequences[np.random.choice(len(groups))] = False
        for i in range(len(groups)):
            drop_steps = np.random.choice(
                [True, False],
                p=[self.cfg.nan_step_probability, 1 - self.cfg.nan_step_probability],
                size=(seq_length, 1)
            )
            drop_masks[i] = drop_sequences[i] | drop_steps
        drop_masks = torch.from_numpy(drop_masks)
        for i, group in enumerate(groups):
            for feature in group:
                x_d[feature] = torch.where(drop_masks[i], torch.nan, x_d[feature])
        return x_d

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """This function has to return the data for the specified basin as a time-indexed pandas DataFrame"""
        raise NotImplementedError

    def _load_attributes(self) -> pd.DataFrame:
        """This function has to return the attributes in a basin-indexed DataFrame."""
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # NEW: Robust resampling helper (hourly <-> 15min) controlled by YAML
    # This is the FIX for your error: factor=0 when converting 1H -> 15min.
    # ---------------------------------------------------------------------
    def _infer_native_frequency_safe(self, index: pd.DatetimeIndex) -> str:
        try:
            freq = utils.infer_frequency(index)
        except Exception:
            freq = None
        if not freq:
            freq = pd.infer_freq(index)
        if not freq:
            raise ValueError("Could not infer native frequency from basin time index.")
        return str(freq)

    def _ensure_complete_time_index(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        full = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        return df.reindex(pd.DatetimeIndex(full, name=df.index.name))

    def _resample_basin_df_to_run_frequency(self, df: pd.DataFrame, run_freq: Optional[str]) -> pd.DataFrame:
        """Convert basin dataframe df to run_freq ("1H" or "15min", etc.).

        Rules:
        - sum_variables: treated as accumulated amounts over the time step:
            * upsample (e.g., 1H -> 15min): forward-fill then divide by factor
            * downsample (e.g., 15min -> 1H): sum
        - all other variables: treated as mean/state:
            * upsample: forward-fill
            * downsample: mean
        """
        if run_freq is None or str(run_freq).lower() in ["none", "null", ""]:
            return df
        if df.empty:
            return df

        native_freq = self._infer_native_frequency_safe(df.index)

        # already same resolution
        try:
            if to_offset(native_freq) == to_offset(run_freq):
                return df
        except Exception:
            pass

        # regularize native grid (avoids weird resample artifacts)
        df = self._ensure_complete_time_index(df, native_freq)

        native_dt = to_offset(native_freq).delta
        run_dt = to_offset(run_freq).delta
        if native_dt is None or run_dt is None:
            raise ValueError(f"Unsupported frequency conversion: {native_freq} -> {run_freq}")

        # ratio = native step / run step
        ratio = native_dt / run_dt
        if ratio <= 0:
            raise ValueError(f"Invalid frequency ratio from {native_freq} to {run_freq}: {ratio}")

        sum_cols = [c for c in df.columns if c in self.sum_variables]

        # Upsample (e.g., 1H -> 15min): ratio = 4
        if ratio > 1:
            k = int(round(float(ratio)))
            if abs(float(ratio) - k) > 1e-6:
                raise ValueError(f"Non-integer upsample ratio {native_freq} -> {run_freq}: {ratio}")

            df_up = df.resample(run_freq).ffill()
            if sum_cols:
                df_up[sum_cols] = df_up[sum_cols] / k
            return df_up

        # Downsample (e.g., 15min -> 1H): ratio = 0.25 => inv = 4
        inv = 1.0 / float(ratio)
        k = int(round(inv))
        if abs(inv - k) > 1e-6:
            raise ValueError(f"Non-integer downsample ratio {native_freq} -> {run_freq}: {inv}")

        agg = {c: ("sum" if c in self.sum_variables else "mean") for c in df.columns}
        df_dn = df.resample(run_freq).agg(agg)
        return df_dn

    def _create_id_to_int(self):
        self.id_to_int = {str(b): i for i, b in enumerate(np.random.permutation(self.basins))}

        file_path = self.cfg.train_dir / "id_to_int.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(self.id_to_int, fp)

    def _dump_scaler(self):
        scaler = defaultdict(dict)
        for key, value in self.scaler.items():
            if isinstance(value, pd.Series) or isinstance(value, xarray.Dataset):
                scaler[key] = value.to_dict()
            else:
                raise RuntimeError(f"Unknown datatype for scaler: {key}. Supported are pd.Series and xarray.Dataset")
        file_path = self.cfg.train_dir / "train_data_scaler.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(dict(scaler), fp)

    def _get_start_and_end_dates(self):
        if getattr(self.cfg, f"per_basin_{self.period}_periods_file") is None:
            if isinstance(getattr(self.cfg, f'{self.period}_start_date'), list):
                if self.period != "train":
                    raise ValueError("Evaluation on split periods currently not supported")
                start_dates = getattr(self.cfg, f'{self.period}_start_date')
            else:
                start_dates = [getattr(self.cfg, f'{self.period}_start_date')]

            if isinstance(getattr(self.cfg, f'{self.period}_end_date'), list):
                end_dates = getattr(self.cfg, f'{self.period}_end_date')
            else:
                end_dates = [getattr(self.cfg, f'{self.period}_end_date')]

            self.start_and_end_dates = {b: {'start_dates': start_dates, 'end_dates': end_dates} for b in self.basins}
        else:
            with open(getattr(self.cfg, f"per_basin_{self.period}_periods_file"), 'rb') as fp:
                self.start_and_end_dates = pickle.load(fp)

    def _load_additional_features(self):
        for file in self.cfg.additional_feature_files:
            with open(file, "rb") as fp:
                self.additional_features.append(pickle.load(fp))

    def _duplicate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature, n_duplicates in self.cfg.duplicate_features.items():
            for n in range(1, n_duplicates + 1):
                df[f"{feature}_copy{n}"] = df[feature]
        return df

    def _add_missing_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        for var in self.cfg.target_variables:
            if var not in df.columns:
                df[var] = np.nan
        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_autoregressive_inputs()

        for feature, shift in self.cfg.lagged_features.items():
            if isinstance(shift, list):
                for s in set(shift):
                    df[f"{feature}_shift{s}"] = df[feature].shift(periods=s, freq="infer")
            elif isinstance(shift, int):
                df[f"{feature}_shift{shift}"] = df[feature].shift(periods=shift, freq="infer")
            else:
                raise ValueError("The value of the 'lagged_features' arg must be either an int or a list of ints")
        return df

    def _check_autoregressive_inputs(self):
        for input in self.cfg.autoregressive_inputs:
            capture = re.compile(r'^(.*)_shift(\d+)$').search(input)
            if not capture:
                raise ValueError('Autoregressive inputs must be a shifted variable with form <variable>_shift<lag> '
                                 f'where <lag> is an integer. Instead got: {input}.')
            if capture[1] not in self.cfg.lagged_features or int(capture[2]) not in self.cfg.lagged_features[capture[1]]:
                raise ValueError('Autoregressive inputs must be in the list of "lagged_inputs".')
        return

    def _load_or_create_xarray_dataset(self) -> xarray.Dataset:
        if (self.cfg.train_data_file is None) or (not self.is_train):
            data_list = []

            keep_cols = self.cfg.target_variables + self.cfg.evolving_attributes + self.cfg.mass_inputs + self.cfg.autoregressive_inputs

            if isinstance(self.cfg.dynamic_inputs, list):
                keep_cols += self.cfg.dynamic_inputs_flattened
            else:
                keep_cols += [i for inputs in self.cfg.dynamic_inputs.values() for i in inputs]

            keep_cols += self.cfg.dynamic_conceptual_inputs
            keep_cols = list(sorted(set(keep_cols)))

            if not self._disable_pbar:
                LOGGER.info("Loading basin data into xarray data set.")
            for basin in tqdm(self.basins, disable=self._disable_pbar, file=sys.stdout):
                df = self._load_basin_data(basin)

                df = pd.concat([df, *[d[basin] for d in self.additional_features]], axis=1)

                if not self.is_train:
                    df = self._add_missing_targets(df)

                df = self._duplicate_features(df)
                df = self._add_lagged_features(df)

                try:
                    df = df[keep_cols]
                except KeyError:
                    not_available_columns = [x for x in keep_cols if x not in df.columns]
                    msg = [
                        f"The following features are not available in the data: {not_available_columns}. ",
                        f"These are the available features: {df.columns.tolist()}"
                    ]
                    raise KeyError("".join(msg))

                for holdout_variable, holdout_dict in self.cfg.random_holdout_from_dynamic_features.items():
                    df[holdout_variable] = samplingutils.bernoulli_subseries_sampler(
                        data=df[holdout_variable].values,
                        missing_fraction=holdout_dict['missing_fraction'],
                        mean_missing_length=holdout_dict['mean_missing_length'],
                    )

                # NEW: enforce requested run frequency (hourly or 15min)
                if self.run_frequency is not None:
                    df = self._resample_basin_df_to_run_frequency(df, self.run_frequency)

                # Make end_date the last second of the specified day
                start_dates = self.start_and_end_dates[basin]["start_dates"]
                end_dates = [
                    date + pd.Timedelta(days=1, seconds=-1) for date in self.start_and_end_dates[basin]["end_dates"]
                ]

                native_frequency = utils.infer_frequency(df.index)
                if not self.frequencies:
                    self.frequencies = [native_frequency]

                try:
                    freq_vs_native = [utils.compare_frequencies(freq, native_frequency) for freq in self.frequencies]
                except ValueError:
                    LOGGER.warning('Cannot compare provided frequencies with native frequency. '
                                   'Make sure the frequencies are not higher than the native frequency.')
                    freq_vs_native = []
                if any(comparison > 1 for comparison in freq_vs_native):
                    raise ValueError(f'Frequency is higher than native data frequency {native_frequency}.')

                offsets = [(self.seq_len[i] - self._predict_last_n[i]) * to_offset(freq)
                           for i, freq in enumerate(self.frequencies)]

                basin_data_list = []
                for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
                    if not all(to_offset(freq).is_on_offset(start_date) for freq in self.frequencies):
                        misaligned = [freq for freq in self.frequencies if not to_offset(freq).is_on_offset(start_date)]
                        raise ValueError(f'start date {start_date} is not aligned with frequencies {misaligned}.')

                    warmup_start_date = min(start_date - offset for offset in offsets)
                    df_sub = df[warmup_start_date:end_date]

                    full_range = pd.date_range(start=warmup_start_date, end=end_date, freq=native_frequency)
                    df_sub = df_sub.reindex(pd.DatetimeIndex(full_range, name=df_sub.index.name))

                    df_sub.loc[df_sub.index < start_date, self.cfg.target_variables] = np.nan
                    basin_data_list.append(df_sub)

                if not basin_data_list:
                    continue

                df = pd.concat(basin_data_list, axis=0)

                df_non_duplicated = df[~df.index.duplicated(keep=False)]
                df_duplicated = df[df.index.duplicated(keep=False)]

                filtered_duplicates = []
                for _, grp in df_duplicated.groupby('date'):
                    mask = ~grp[self.cfg.target_variables].isna().any(axis=1)
                    if not mask.any():
                        filtered_duplicates.append(grp.head(1))
                    else:
                        filtered_duplicates.append(grp[mask].head(1))

                if filtered_duplicates:
                    df_filtered_duplicates = pd.concat(filtered_duplicates, axis=0)
                    df = pd.concat([df_non_duplicated, df_filtered_duplicates], axis=0)
                else:
                    df = df_non_duplicated

                df = df.sort_index(axis=0, ascending=True)
                df = df.reindex(
                    pd.DatetimeIndex(
                        data=pd.date_range(df.index[0], df.index[-1], freq=native_frequency),
                        name=df.index.name
                    )
                )

                xr = xarray.Dataset.from_dataframe(df.astype(np.float32))
                xr = xr.assign_coords({'basin': basin})
                data_list.append(xr)

            if not data_list:
                if self.is_train:
                    raise NoTrainDataError
                else:
                    raise NoEvaluationDataError

            xr = xarray.concat(data_list, dim="basin")

            if self.is_train and self.cfg.save_train_data:
                self._save_xarray_dataset(xr)

        else:
            with self.cfg.train_data_file.open("rb") as fp:
                d = pickle.load(fp)
            xr = xarray.Dataset.from_dict(d)
            if not self.frequencies:
                native_frequency = utils.infer_frequency(xr["date"].values)
                self.frequencies = [native_frequency]

        return xr

    def _save_xarray_dataset(self, xr: xarray.Dataset):
        file_path = self.cfg.train_dir / "train_data.p"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as fp:
            pickle.dump(xr.to_dict(), fp)

    def _calculate_per_basin_std(self, xr: xarray.Dataset):
        if not self._disable_pbar:
            LOGGER.info("Calculating target variable stds per basin")
        nan_basins = []
        for basin in tqdm(self.basins, file=sys.stdout, disable=self._disable_pbar):
            obs = xr.sel(basin=basin)[self.cfg.target_variables].to_array().values
            if np.sum(~np.isnan(obs)) > 1:
                per_basin_target_stds = torch.tensor(np.expand_dims(np.nanstd(obs, axis=1), 0), dtype=torch.float32)
            else:
                nan_basins.append(basin)
                per_basin_target_stds = torch.full((1, obs.shape[0]), np.nan, dtype=torch.float32)

            self._per_basin_target_stds[basin] = per_basin_target_stds

        if len(nan_basins) > 0:
            LOGGER.warning("The following basins had not enough valid target values to calculate a standard deviation: "
                           f"{', '.join(nan_basins)}. NSE loss values for this basin will be NaN.")

    def _create_lookup_table(self, xr: xarray.Dataset):
        lookup = []
        if not self._disable_pbar:
            LOGGER.info("Create lookup table and convert to pytorch tensor")

        basins_without_samples = []
        basin_coordinates = xr["basin"].values.tolist()
        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):

            x_d, x_s, y, dates = {}, {}, {}, {}
            frequency_maps = {}
            lowest_freq = utils.sort_frequencies(self.frequencies)[0]

            df_native = xr.sel(basin=basin).to_dataframe()

            for freq in self.frequencies:
                if isinstance(self.cfg.dynamic_inputs, list):
                    dynamic_cols = self.cfg.mass_inputs + self.cfg.dynamic_inputs_flattened
                else:
                    dynamic_cols = self.cfg.mass_inputs + self.cfg.dynamic_inputs[freq]

                dynamic_cols += self.cfg.dynamic_conceptual_inputs

                all_cols = dynamic_cols + self.cfg.target_variables + self.cfg.evolving_attributes + self.cfg.autoregressive_inputs

                # per-column aggregation (sum for sum_variables, mean for others)
                agg = {c: "mean" for c in all_cols}
                for c in self.sum_variables:
                    if c in agg:
                        agg[c] = "sum"

                df_resampled = df_native[all_cols].resample(freq).agg(agg)

                x_d[freq] = {col: df_resampled[[col]].values for col in dynamic_cols}
                y[freq] = df_resampled[self.cfg.target_variables].values
                if self.cfg.evolving_attributes:
                    x_s[freq] = df_resampled[self.cfg.evolving_attributes].values

                dates[freq] = df_resampled.index.to_numpy()

                frequency_factor = int(utils.get_frequency_factor(lowest_freq, freq))
                if len(df_resampled) % frequency_factor != 0:
                    raise ValueError(f'The length of the dataframe at frequency {freq} is {len(df_resampled)} '
                                     f'(including warmup), which is not a multiple of {frequency_factor} (i.e., the '
                                     f'factor between the lowest frequency {lowest_freq} and the frequency {freq}. '
                                     f'To fix this, adjust the {self.period} start or end date such that the period '
                                     f'(including warmup) has a length that is divisible by {frequency_factor}.')
                frequency_maps[freq] = np.arange(len(df_resampled) // frequency_factor) \
                                       * frequency_factor + (frequency_factor - 1)

            if not self.is_train:
                self.period_starts[basin] = pd.to_datetime(xr.sel(basin=basin)["date"].values[0])

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
                if self.is_train:
                    x_d_validate = [np.concatenate([v for v in x_d[freq].values()], axis=-1)
                                    for freq in self.frequencies]
                else:
                    x_d_validate = None
                flag = _validate_samples(
                    x_d=x_d_validate,
                    x_s=[x_s[freq] for freq in self.frequencies] if self.is_train and x_s else None,
                    y=[y[freq] for freq in self.frequencies] if self.is_train else None,
                    frequency_maps=[frequency_maps[freq] for freq in self.frequencies],
                    seq_length=self.seq_len,
                    predict_last_n=self._predict_last_n
                )

            if self.cfg.autoregressive_inputs:
                if len(self.frequencies) > 1:
                    raise ValueError('Autoregressive inputs are not supported for datasets with multiple frequencies.')
                x_d[self.frequencies[0]].update({col: df_resampled[[col]].values
                                                 for col in self.cfg.autoregressive_inputs})

            valid_samples = np.argwhere(flag == 1).flatten()

            if self.non_overlapping_sequences:
                if len(self.frequencies) != 1:
                    raise ValueError("non_overlapping_sequences is currently only supported for single-frequency datasets.")
                stride = int(self.seq_stride)
                if stride <= 0:
                    raise ValueError("seq_stride must be a positive integer.")
                if stride == 1:
                    stride = int(self.seq_len[0])
                valid_samples = valid_samples[::stride]
            elif self.seq_stride > 1:
                stride = int(self.seq_stride)
                if stride <= 0:
                    raise ValueError("seq_stride must be a positive integer.")
                valid_samples = valid_samples[::stride]

            for f in valid_samples:
                lookup.append((basin, [frequency_maps[freq][int(f)] for freq in self.frequencies]))

            if valid_samples.size > 0:
                if self.cfg.forecast_inputs_flattened and not self.cfg.hindcast_inputs_flattened:
                    raise ValueError('Hindcast inputs must be provided if forecast inputs are provided.')
                self._x_d[basin] = {freq: {k: torch.from_numpy(v.astype(np.float32))
                                           for k, v in _x_d.items()}
                                    for freq, _x_d in x_d.items()}
                self._y[basin] = {freq: torch.from_numpy(_y.astype(np.float32)) for freq, _y in y.items()}
                if x_s:
                    self._x_s[basin] = {freq: torch.from_numpy(_x_s.astype(np.float32)) for freq, _x_s in x_s.items()}
                self._dates[basin] = dates
            else:
                basins_without_samples.append(basin)

        if basins_without_samples:
            LOGGER.info(f"These basins do not have a single valid sample in the {self.period} period: {basins_without_samples}")

        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)

        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError

    def _load_hydroatlas_attributes(self):
        df = utils.load_hydroatlas_attributes(self.cfg.data_dir, basins=self.basins)
        drop_cols = [c for c in df.columns if c not in self.cfg.hydroatlas_attributes]
        df = df.drop(drop_cols, axis=1)
        if self.is_train:
            utils.attributes_sanity_check(df=df)
        return df

    def _load_combined_attributes(self):
        dfs = []
        if self.cfg.static_attributes:
            df = self._load_attributes()
            missing_attrs = [attr for attr in self.cfg.static_attributes if attr not in df.columns]
            if len(missing_attrs) > 0:
                raise ValueError(f'Static attributes {missing_attrs} are missing.')
            df = df[self.cfg.static_attributes]
            if self._compute_scaler:
                utils.attributes_sanity_check(df=df)
            dfs.append(df)

        if self.cfg.hydroatlas_attributes:
            dfs.append(self._load_hydroatlas_attributes())

        if dfs:
            df = pd.concat(dfs, axis=1)
            combined_attributes = self.cfg.static_attributes + self.cfg.hydroatlas_attributes
            missing_columns = [attr for attr in combined_attributes if attr not in df.columns]
            if missing_columns:
                raise ValueError(f"The following attributes are not available in the dataset: {missing_columns}")

            df = df.sort_index(axis=1)

            if self._compute_scaler:
                self.scaler["attribute_means"] = df.mean()
                self.scaler["attribute_stds"] = df.std()

            if any([k.startswith("camels_attr") for k in self.scaler.keys()]):
                LOGGER.warning("Deprecation warning: Using old scaler files won't be supported in the upcoming release.")
                df = (df - self.scaler['camels_attr_means']) / self.scaler["camels_attr_stds"]
            else:
                df = (df - self.scaler['attribute_means']) / self.scaler["attribute_stds"]

            for basin in self.basins:
                attributes = df.loc[df.index == basin].values.flatten()
                self._attributes[basin] = torch.from_numpy(attributes.astype(np.float32))

    def _load_data(self):
        self._load_combined_attributes()
        xr = self._load_or_create_xarray_dataset()

        if self.cfg.loss.lower() in ['nse', 'weightednse']:
            self._calculate_per_basin_std(xr)

        if self._compute_scaler:
            self._setup_normalization(xr)

        xr = (xr - self.scaler["xarray_feature_center"]) / self.scaler["xarray_feature_scale"]
        self._create_lookup_table(xr)

    def _setup_normalization(self, xr: xarray.Dataset):
        self.scaler["xarray_feature_scale"] = xr.std(skipna=True)
        self.scaler["xarray_feature_center"] = xr.mean(skipna=True)

        for feature, feature_specs in self.cfg.custom_normalization.items():
            for key, val in feature_specs.items():
                if key == "centering":
                    if (val is None) or (val.lower() == "none"):
                        self.scaler["xarray_feature_center"][feature] = np.float32(0.0)
                    elif val.lower() == "median":
                        self.scaler["xarray_feature_center"][feature] = xr[feature].median(skipna=True)
                    elif val.lower() == "min":
                        self.scaler["xarray_feature_center"][feature] = xr[feature].min(skipna=True)
                    elif val.lower() == "mean":
                        pass
                    else:
                        raise ValueError(f"Unknown centering method {val}")

                elif key == "scaling":
                    if (val is None) or (val.lower() == "none"):
                        self.scaler["xarray_feature_scale"][feature] = np.float32(1.0)
                    elif val == "minmax":
                        self.scaler["xarray_feature_scale"][feature] = xr[feature].max(skipna=True) - xr[feature].min(skipna=True)
                    elif val == "std":
                        pass
                    else:
                        raise ValueError(f"Unknown scaling method {val}")
                else:
                    raise ValueError("Unknown dict key. Use 'centering' and/or 'scaling' for each feature.")

    def get_period_start(self, basin: str) -> pd.Timestamp:
        return self.period_starts[basin]

    def _initialize_frequency_configuration(self):
        self.frequencies = self.cfg.use_frequencies
        self.seq_len = self.cfg.seq_length
        self._predict_last_n = self.cfg.predict_last_n
        if not self.frequencies:
            if not isinstance(self.seq_len, int) or not isinstance(self._predict_last_n, int):
                raise ValueError('seq_length and predict_last_n must be integers if use_frequencies is not provided.')
            self.seq_len = [self.seq_len]
            self._predict_last_n = [self._predict_last_n]
        else:
            if not isinstance(self.seq_len, dict) or not isinstance(self._predict_last_n, dict) \
                    or any([freq not in self.seq_len for freq in self.frequencies]) \
                    or any([freq not in self._predict_last_n for freq in self.frequencies]):
                raise ValueError('seq_length and predict_last_n must be dictionaries with one key per frequency.')
            self.seq_len = [self.seq_len[freq] for freq in self.frequencies]
            self._predict_last_n = [self._predict_last_n[freq] for freq in self.frequencies]

    @staticmethod
    def collate_fn(samples: List[Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]]]
                   ) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]]:
        batch = {}
        if not samples:
            return batch
        features = list(samples[0].keys())
        for feature in features:
            if feature.startswith('date'):
                batch[feature] = np.stack([sample[feature] for sample in samples], axis=0)
            elif feature.startswith('x_d'):
                batch[feature] = {k: torch.stack([sample[feature][k] for sample in samples], dim=0)
                                  for k in samples[0][feature]}
            else:
                batch[feature] = torch.stack([sample[feature] for sample in samples], dim=0)
        return batch


@njit()
def _validate_samples(x_d: List[np.ndarray], x_s: List[np.ndarray], y: List[np.ndarray], seq_length: List[int],
                      predict_last_n: List[int], frequency_maps: List[np.ndarray]) -> np.ndarray:
    n_samples = len(frequency_maps[0])
    flag = np.ones(n_samples)
    for i in range(len(frequency_maps)):
        for j in prange(n_samples):
            last_sample_of_freq = frequency_maps[i][j]
            if last_sample_of_freq < seq_length[i] - 1:
                flag[j] = 0
                continue

            if x_d is not None:
                _x_d = x_d[i][last_sample_of_freq - seq_length[i] + 1:last_sample_of_freq + 1]
                if np.any(np.isnan(_x_d)):
                    flag[j] = 0
                    continue

            if y is not None:
                _y = y[i][last_sample_of_freq - predict_last_n[i] + 1:last_sample_of_freq + 1]
                if np.prod(np.array(_y.shape)) > 0 and np.all(np.isnan(_y)):
                    flag[j] = 0
                    continue

            if x_s is not None:
                _x_s = x_s[i][last_sample_of_freq]
                if np.any(np.isnan(_x_s)):
                    flag[j] = 0

    return flag
