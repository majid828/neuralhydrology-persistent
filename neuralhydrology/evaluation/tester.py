import logging
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.utils import get_frequency_factor, load_basin_file, load_scaler, sort_frequencies
from neuralhydrology.evaluation import plots
from neuralhydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from neuralhydrology.evaluation.utils import load_basin_id_encoding, metrics_to_dataframe
from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.training import get_loss_obj, get_regularization_obj
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import AllNaNError, NoEvaluationDataError

LOGGER = logging.getLogger(__name__)


class BaseTester(object):
    """Base class to run inference on a model."""

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        self.cfg = cfg
        self.run_dir = run_dir
        self.init_model = init_model

        if period in ["train", "validation", "test"]:
            self.period = period
        else:
            raise ValueError(f'Invalid period {period}. Must be one of ["train", "validation", "test"]')

        # determine device
        self._set_device()

        if self.init_model:
            self.model = get_model(cfg).to(self.device)

        self._disable_pbar = cfg.verbose == 0

        # pre-initialize variables, defined in class methods
        self.basins = None
        self.scaler = None
        self.id_to_int = {}
        self.additional_features = []

        # placeholder to store cached validation data
        self.cached_datasets = {}

        # initialize loss object to compute the loss of the evaluation data
        self.loss_obj = get_loss_obj(cfg)
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

        # ------------------------------
        # Persistent eval state (mirror training)
        # ------------------------------
        self._persistent_hidden_eval = None
        self._current_basin_idx_eval = None

        # Default behavior inside tester (NO YAML key needed):
        # - "per_basin" is NH-safe (no cross-basin leakage even if evaluate loop changes)
        # - If you want strict "start-only reset", change this constant to "start".
        self._persistent_eval_reset_mode = "per_basin"  # {"per_basin", "start"}

        self._load_run_data()

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(":")[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
                self.device = torch.device(self.cfg.device)
            elif self.cfg.device == "mps":
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    raise RuntimeError("MPS device is not available.")
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def _load_run_data(self):
        """Load run specific data from run directory"""
        self.basins = load_basin_file(getattr(self.cfg, f"{self.period}_basin_file"))
        self.scaler = load_scaler(self.run_dir)

        # check for old scaler files, where the center/scale parameters had still old names
        if "xarray_means" in self.scaler.keys():
            self.scaler["xarray_feature_center"] = self.scaler.pop("xarray_means")
        if "xarray_stds" in self.scaler.keys():
            self.scaler["xarray_feature_scale"] = self.scaler.pop("xarray_stds")

        if self.cfg.use_basin_id_encoding:
            self.id_to_int = load_basin_id_encoding(self.run_dir)

        for file in self.cfg.additional_feature_files:
            with open(file, "rb") as fp:
                self.additional_features.append(pickle.load(fp))

    def _get_weight_file(self, epoch: int):
        if epoch is None:
            weight_file = sorted(list(self.run_dir.glob("model_epoch*.pt")))[-1]
        else:
            weight_file = self.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"
        return weight_file

    def _load_weights(self, epoch: int = None):
        weight_file = self._get_weight_file(epoch)
        LOGGER.info(f"Using the model weights from {weight_file}")
        self.model.load_state_dict(torch.load(weight_file, map_location=self.device))

    def _get_dataset(self, basin: str) -> BaseDataset:
        ds = get_dataset(
            cfg=self.cfg,
            is_train=False,
            period=self.period,
            basin=basin,
            additional_features=self.additional_features,
            id_to_int=self.id_to_int,
            scaler=self.scaler,
        )
        return ds

    # ================= MOD: persistent-continuous detection helper =================
    def _is_persistent_continuous_eval(self, frequencies: List[str]) -> bool:
        """
        True when we are in the specific case:
        - persistent_state enabled
        - non_overlapping_sequences enabled
        - single frequency
        - seq_stride is 1 or equals seq_length (common for non-overlap)
        This is the case where NH's (date,time_step) logic is wrong.
        """
        if not getattr(self.cfg, "persistent_state", False):
            return False
        if not getattr(self.cfg, "non_overlapping_sequences", False):
            return False
        if len(frequencies) != 1:
            return False

        # seq_stride may not exist in older configs; default to 1
        seq_stride = int(getattr(self.cfg, "seq_stride", 1))

        # seq_length may be int or dict
        seq_length = self.cfg.seq_length
        if isinstance(seq_length, dict):
            # single-frequency dict
            freq0 = frequencies[0]
            seq_length = int(seq_length[freq0])
        else:
            seq_length = int(seq_length)

        return seq_stride in (1, seq_length)

    def evaluate(
        self,
        epoch: int = None,
        save_results: bool = True,
        save_all_output: bool = False,
        metrics: Union[list, dict] = [],
        model: torch.nn.Module = None,
        experiment_logger: Logger = None,
    ) -> dict:
        """Evaluate the model."""
        if model is None:
            if self.init_model:
                self._load_weights(epoch=epoch)
                model = self.model
            else:
                raise RuntimeError("No model was initialized for the evaluation")

        basins = self.basins
        if self.period == "validation":
            if len(basins) > self.cfg.validate_n_random_basins:
                random.shuffle(basins)
                basins = basins[: self.cfg.validate_n_random_basins]

        if self.cfg.mc_dropout:
            model.train()
        else:
            model.eval()

        # STRICT reset option: reset once at start of evaluation
        if self._persistent_eval_reset_mode == "start":
            self._persistent_hidden_eval = None
            self._current_basin_idx_eval = None

        results = defaultdict(dict)
        all_output = {basin: None for basin in basins}

        pbar = tqdm(basins, file=sys.stdout, disable=self._disable_pbar)
        pbar.set_description("# Validation" if self.period == "validation" else "# Evaluation")

        for basin in pbar:
            # NH-safe option: reset at basin boundaries
            if self._persistent_eval_reset_mode == "per_basin":
                self._persistent_hidden_eval = None
                self._current_basin_idx_eval = None

            if self.cfg.cache_validation_data and basin in self.cached_datasets.keys():
                ds = self.cached_datasets[basin]
            else:
                try:
                    ds = self._get_dataset(basin)
                except NoEvaluationDataError:
                    continue
                if self.cfg.cache_validation_data and self.period == "validation":
                    self.cached_datasets[basin] = ds

            loader = DataLoader(ds, batch_size=self.cfg.batch_size, num_workers=0, collate_fn=ds.collate_fn)

            y_hat, y, dates, mean_losses, all_output[basin] = self._evaluate(
                model, loader, ds.frequencies, save_all_output
            )

            if experiment_logger is not None:
                experiment_logger.log_step(**{k: (v, len(loader)) for k, v in mean_losses.items()})

            predict_last_n = self.cfg.predict_last_n
            seq_length = self.cfg.seq_length
            if isinstance(predict_last_n, int):
                predict_last_n = {ds.frequencies[0]: predict_last_n}
            if isinstance(seq_length, int):
                seq_length = {ds.frequencies[0]: seq_length}

            lowest_freq = sort_frequencies(ds.frequencies)[0]

            # ================= MOD: decide plotting/metric mode =================
            persistent_continuous = self._is_persistent_continuous_eval(ds.frequencies)

            for freq in ds.frequencies:
                if predict_last_n[freq] == 0:
                    continue

                results[basin][freq] = {}

                feature_scaler = self.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values
                feature_center = self.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values
                y_freq = y[freq] * feature_scaler + feature_center

                # scale y_hat (supports regression [N,L,T] and uncertainty [N,L,T,S])
                if y_hat[freq].ndim == 3 or (len(feature_scaler) == 1):
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                elif y_hat[freq].ndim == 4:
                    feature_scaler_ = np.expand_dims(feature_scaler, (0, 1, 3))
                    feature_center_ = np.expand_dims(feature_center, (0, 1, 3))
                    y_hat_freq = y_hat[freq] * feature_scaler_ + feature_center_
                else:
                    raise RuntimeError(f"Simulations have {y_hat[freq].ndim} dimension. Only 3 and 4 are supported.")

                # ================= MOD: persistent-continuous xarray =================
                if persistent_continuous:
                    # dates[freq] is the truth; flatten directly (no (date,time_step) grid)
                    dt_1d = pd.to_datetime(dates[freq].reshape(-1))
                    y_obs_1d = y_freq.reshape(-1, y_freq.shape[-1])

                    if y_hat_freq.ndim == 3:
                        y_sim_1d = y_hat_freq.reshape(-1, y_hat_freq.shape[-1])
                        data_vars = {}
                        for i, var in enumerate(self.cfg.target_variables):
                            data_vars[f"{var}_obs"] = (("datetime",), y_obs_1d[:, i])
                            data_vars[f"{var}_sim"] = (("datetime",), y_sim_1d[:, i])
                    else:
                        # y_hat_freq is [N_seq, L, n_targets, n_samples] -> [T, n_targets, n_samples]
                        y_sim_1d = y_hat_freq.reshape(-1, y_hat_freq.shape[2], y_hat_freq.shape[3])
                        data_vars = {}
                        for i, var in enumerate(self.cfg.target_variables):
                            data_vars[f"{var}_obs"] = (("datetime",), y_obs_1d[:, i])
                            data_vars[f"{var}_sim"] = (("datetime", "samples"), y_sim_1d[:, i, :])

                    xr = xarray.Dataset(data_vars=data_vars, coords={"datetime": dt_1d})
                    results[basin][freq]["xr"] = xr

                    # ================= MOD: metrics from true datetime series =================
                    if metrics:
                        for target_variable in self.cfg.target_variables:
                            obs_ = xr[f"{target_variable}_obs"]
                            sim_ = xr[f"{target_variable}_sim"]

                            if target_variable in self.cfg.clip_targets_to_zero:
                                sim_ = xarray.where(sim_ < 0, 0, sim_)

                            if "samples" in sim_.dims:
                                sim_ = sim_.mean(dim="samples")

                            var_metrics = metrics if isinstance(metrics, list) else metrics[target_variable]
                            if "all" in var_metrics:
                                var_metrics = get_available_metrics()

                            try:
                                values = calculate_metrics(obs_, sim_, metrics=var_metrics, resolution=freq)
                            except AllNaNError as err:
                                msg = (
                                    f"Basin {basin} "
                                    + (f"{target_variable} " if len(self.cfg.target_variables) > 1 else "")
                                    + (f"{freq} " if len(ds.frequencies) > 1 else "")
                                    + str(err)
                                )
                                LOGGER.warning(msg)
                                values = {metric: np.nan for metric in var_metrics}

                            if len(self.cfg.target_variables) > 1:
                                values = {f"{target_variable}_{key}": val for key, val in values.items()}
                            if len(ds.frequencies) > 1:
                                values = {f"{key}_{freq}": val for key, val in values.items()}

                            if experiment_logger is not None:
                                experiment_logger.log_step(**values)

                            for k, v in values.items():
                                results[basin][freq][k] = v

                    # Done for this freq
                    continue

                # ---------------- ORIGINAL NH BEHAVIOR (unchanged) ----------------
                data_vars = self._create_xarray_data_vars(y_hat_freq, y_freq)
                frequency_factor = int(get_frequency_factor(lowest_freq, freq))

                coords = {
                    "date": dates[lowest_freq][:, -1],
                    "time_step": ((dates[freq][0, :] - dates[freq][0, -1]) / pd.Timedelta(freq)).astype(np.int64)
                    + frequency_factor
                    - 1,
                }
                xr = xarray.Dataset(data_vars=data_vars, coords=coords)
                xr = xr.reindex(
                    {
                        "date": pd.DatetimeIndex(
                            pd.date_range(xr["date"].values[0], xr["date"].values[-1], freq=lowest_freq),
                            name="date",
                        )
                    }
                )
                results[basin][freq]["xr"] = xr

                freq_date_range = pd.date_range(start=dates[lowest_freq][0, -1], end=dates[freq][-1, -1], freq=freq)
                mask = np.ones(frequency_factor).astype(bool)
                mask[:-predict_last_n[freq]] = False
                freq_date_range = freq_date_range[np.tile(mask, len(xr["date"]))]

                if frequency_factor < predict_last_n[freq] and basin == basins[0]:
                    tqdm.write(
                        f"Metrics for {freq} are calculated over last {frequency_factor} elements only. "
                        f"Ignoring {predict_last_n[freq] - frequency_factor} predictions per sequence."
                    )

                if metrics:
                    for target_variable in self.cfg.target_variables:
                        obs_ = (
                            xr.isel(time_step=slice(-frequency_factor, None))
                            .stack(datetime=["date", "time_step"])
                            .drop_vars({"datetime", "date", "time_step"})[f"{target_variable}_obs"]
                        )
                        obs_["datetime"] = freq_date_range

                        if not all(obs_.isnull()):
                            sim_ = (
                                xr.isel(time_step=slice(-frequency_factor, None))
                                .stack(datetime=["date", "time_step"])
                                .drop_vars({"datetime", "date", "time_step"})[f"{target_variable}_sim"]
                            )
                            sim_["datetime"] = freq_date_range

                            if target_variable in self.cfg.clip_targets_to_zero:
                                sim_ = xarray.where(sim_ < 0, 0, sim_)

                            if "samples" in sim_.dims:
                                sim_ = sim_.mean(dim="samples")

                            var_metrics = metrics if isinstance(metrics, list) else metrics[target_variable]
                            if "all" in var_metrics:
                                var_metrics = get_available_metrics()

                            try:
                                values = calculate_metrics(obs_, sim_, metrics=var_metrics, resolution=freq)
                            except AllNaNError as err:
                                msg = (
                                    f"Basin {basin} "
                                    + (f"{target_variable} " if len(self.cfg.target_variables) > 1 else "")
                                    + (f"{freq} " if len(ds.frequencies) > 1 else "")
                                    + str(err)
                                )
                                LOGGER.warning(msg)
                                values = {metric: np.nan for metric in var_metrics}

                            if len(self.cfg.target_variables) > 1:
                                values = {f"{target_variable}_{key}": val for key, val in values.items()}
                            if len(ds.frequencies) > 1:
                                values = {f"{key}_{freq}": val for key, val in values.items()}

                            if experiment_logger is not None:
                                experiment_logger.log_step(**values)

                            for k, v in values.items():
                                results[basin][freq][k] = v

        results = dict(results)

        if (self.period == "validation") and (self.cfg.log_n_figures > 0) and (experiment_logger is not None) and results:
            self._create_and_log_figures(results, experiment_logger, epoch)

        results_to_save = results if save_results else None
        states_to_save = all_output if save_all_output else None
        if save_results or save_all_output:
            self._save_results(results=results_to_save, states=states_to_save, epoch=epoch)

        return results

    def _create_and_log_figures(self, results: dict, experiment_logger: Logger, epoch: int):
        basins = list(results.keys())
        random.shuffle(basins)
        for target_var in self.cfg.target_variables:
            max_figures = min(self.cfg.validate_n_random_basins, self.cfg.log_n_figures, len(basins))
            for freq in results[basins[0]].keys():
                figures = []
                for i in range(max_figures):
                    xr = results[basins[i]][freq]["xr"]

                    # NOTE: For persistent-continuous, obs/sim are 1D.
                    # plots.regression_plot / uncertainty_plot can handle 1D arrays.
                    obs = xr[f"{target_var}_obs"].values
                    sim = xr[f"{target_var}_sim"].values

                    if target_var in self.cfg.clip_targets_to_zero:
                        sim = xarray.where(sim < 0, 0, sim)

                    figures.append(
                        self._get_plots(
                            obs, sim, title=f"{target_var} - Basin {basins[i]} - Epoch {epoch} - Frequency {freq}"
                        )[0]
                    )
                experiment_logger.log_figures(figures, freq, preamble=re.sub(r"[^A-Za-z0-9\._\-]+", "", target_var))

    def _save_results(self, results: Optional[dict], states: Optional[dict] = None, epoch: int = None):
        weight_file = self._get_weight_file(epoch=epoch)
        parent_directory = self.run_dir / self.period / weight_file.stem
        parent_directory.mkdir(parents=True, exist_ok=True)

        if self.cfg.metrics and results is not None:
            metrics_list = self.cfg.metrics
            if isinstance(metrics_list, dict):
                metrics_list = list(set(metrics_list.values()))
            if "all" in metrics_list:
                metrics_list = get_available_metrics()
            df = metrics_to_dataframe(results, metrics_list, self.cfg.target_variables)
            metrics_file = parent_directory / f"{self.period}_metrics.csv"
            df.to_csv(metrics_file)
            LOGGER.info(f"Stored metrics at {metrics_file}")

        if results is not None:
            result_file = parent_directory / f"{self.period}_results.p"
            with result_file.open("wb") as fp:
                pickle.dump(results, fp)
            LOGGER.info(f"Stored results at {result_file}")

        if states is not None:
            result_file = parent_directory / f"{self.period}_all_output.p"
            with result_file.open("wb") as fp:
                pickle.dump(states, fp)
            LOGGER.info(f"Stored states at {result_file}")

    # ------------------------------------------------------------------
    # Persistent helpers (mirror BaseTrainer)
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_time_batch(t: torch.Tensor) -> torch.Tensor:
        """Flatten [B, L, ...] → [1, B*L, ...]"""
        if (not torch.is_tensor(t)) or t.dim() < 2:
            return t
        b, l = t.shape[0], t.shape[1]
        return t.reshape(1, b * l, *t.shape[2:])

    def _slice_time_segment(self, data_dict: Dict[str, object], segment_slice: slice) -> Dict[str, object]:
        """Slice a contiguous segment from reshaped data (expects time-like tensors as [1, L_new, ...])."""
        seg = {}
        for key, val in data_dict.items():
            if key.startswith("x_d"):
                seg[key] = {feat: v[:, segment_slice, ...] for feat, v in val.items()}
            elif key.startswith("x_s"):
                seg[key] = val
            elif key.startswith("y") or key.startswith("basin_idx"):
                if torch.is_tensor(val):
                    if val.dim() == 1:
                        seg[key] = val[segment_slice]
                    else:
                        seg[key] = val[:, segment_slice, ...]
                else:
                    seg[key] = val
            elif key.startswith("date"):
                seg[key] = val  # keep as-is
            elif torch.is_tensor(val) and val.dim() >= 2:
                seg[key] = val[:, segment_slice, ...]
            else:
                seg[key] = val
        return seg

    def _is_persistent_mirror_eval(self, frequencies: List[str]) -> bool:
        # Keep this strict to avoid breaking multi-frequency models
        return (
            getattr(self.cfg, "persistent_state", False)
            and self.cfg.model.lower() == "persistentlstm"
            and len(frequencies) == 1
        )

    # ------------------------------------------------------------------
    # EVALUATION CORE
    # ------------------------------------------------------------------
    def _evaluate(self, model: BaseModel, loader: DataLoader, frequencies: List[str], save_all_output: bool = False):
        """Evaluate model."""
        if self._is_persistent_mirror_eval(frequencies):
            return self._evaluate_persistent_mirror_training(model, loader, frequencies, save_all_output)

        # ---------------- ORIGINAL (non-persistent) BEHAVIOR ----------------
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            predict_last_n = {frequencies[0]: predict_last_n}

        preds, obs, dates, all_output = {}, {}, {}, {}
        losses = []

        with torch.no_grad():
            for data in loader:
                for key in data:
                    if key.startswith("x_d"):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith("date"):
                        data[key] = data[key].to(self.device)

                data = model.pre_model_hook(data, is_train=False)
                predictions, loss = self._get_predictions_and_loss(model, data)

                if all_output:
                    for key, value in predictions.items():
                        if value is not None and type(value) != dict:
                            all_output[key].append(value.detach().cpu().numpy())
                elif save_all_output:
                    all_output = {
                        key: [value.detach().cpu().numpy()]
                        for key, value in predictions.items()
                        if value is not None and type(value) != dict
                    }

                for freq in frequencies:
                    if predict_last_n[freq] == 0:
                        continue
                    freq_key = "" if len(frequencies) == 1 else f"_{freq}"
                    y_hat_sub, y_sub = self._subset_targets(model, data, predictions, predict_last_n[freq], freq_key)
                    date_sub = data[f"date{freq_key}"][:, -predict_last_n[freq] :]

                    if freq not in preds:
                        preds[freq] = y_hat_sub.detach().cpu()
                        obs[freq] = y_sub.cpu()
                        dates[freq] = date_sub
                    else:
                        preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                        obs[freq] = torch.cat((obs[freq], y_sub.detach().cpu()), 0)
                        dates[freq] = np.concatenate((dates[freq], date_sub), axis=0)

                losses.append(loss)

            for freq in preds.keys():
                preds[freq] = preds[freq].numpy()
                obs[freq] = obs[freq].numpy()

        for key, list_of_data in all_output.items():
            all_output[key] = np.concatenate(list_of_data, 0)

        mean_losses = {}
        if len(losses) == 0:
            mean_losses["loss"] = np.nan
        else:
            for loss_name in losses[0].keys():
                loss_values = [loss[loss_name] for loss in losses]
                mean_losses[loss_name] = np.nanmean(loss_values) if not np.all(np.isnan(loss_values)) else np.nan

        return preds, obs, dates, mean_losses, all_output

    # ---------------- PERSISTENT MIRROR EVAL ----------------
    def _evaluate_persistent_mirror_training(
        self, model: BaseModel, loader: DataLoader, frequencies: List[str], save_all_output: bool = False
    ):
        """Mirror BaseTrainer persistent logic 1:1, stitch outputs back to [B, L, ...],
        and compute global timestep-weighted mean losses across the entire loader (NaN-safe).
        """
        freq = frequencies[0]
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            predict_last_n = {freq: predict_last_n}

        preds, obs, dates, all_output = {}, {}, {}, {}

        # Global timestep-weighted loss accumulators across loader
        loss_sum: Dict[str, float] = {}
        loss_w: Dict[str, int] = {}

        with torch.no_grad():
            for data in loader:
                # move to device
                for key in data.keys():
                    if key.startswith("x_d"):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith("date"):
                        data[key] = data[key].to(self.device)

                data = model.pre_model_hook(data, is_train=False)

                # infer B, L from any x_d tensor
                any_xd = None
                for k, v in data.items():
                    if k.startswith("x_d") and isinstance(v, dict) and len(v) > 0:
                        any_xd = next(iter(v.values()))
                        break
                if any_xd is None or any_xd.dim() < 2:
                    raise RuntimeError("Persistent eval: could not infer [B, L] from x_d.")
                B, L = int(any_xd.shape[0]), int(any_xd.shape[1])

                # ----- Step 1: Build reshaped_data with flattened time dimension -----
                reshaped_data: Dict[str, object] = {}
                for key, val in data.items():
                    if key.startswith("x_d"):
                        reshaped_data[key] = {feat: self._flatten_time_batch(v) for feat, v in val.items()}
                    elif key.startswith("x_s"):
                        reshaped_data[key] = val
                    elif key.startswith("y"):
                        reshaped_data[key] = self._flatten_time_batch(val)
                    elif key.startswith("basin_idx"):
                        if torch.is_tensor(val) and val.dim() == 1:
                            basin_2d = val.view(B, 1).expand(B, L)
                            reshaped_data[key] = self._flatten_time_batch(basin_2d)
                        else:
                            reshaped_data[key] = self._flatten_time_batch(val)
                    else:
                        if torch.is_tensor(val):
                            if val.dim() >= 2:
                                reshaped_data[key] = self._flatten_time_batch(val)
                            else:
                                reshaped_data[key] = val.view(1, -1) if val.dim() == 1 else val
                        else:
                            reshaped_data[key] = val

                basin_indices_flat = reshaped_data["basin_idx"].view(-1)
                L_new = basin_indices_flat.size(0)
                if L_new == 0:
                    continue

                # ----- Step 2: Split flattened sequence by basin changes -----
                diff = basin_indices_flat[1:] - basin_indices_flat[:-1]
                change_points = torch.where(diff != 0)[0] + 1

                segment_slices = []
                start = 0
                for cp in change_points:
                    cp_int = int(cp.item())
                    segment_slices.append(slice(start, cp_int))
                    start = cp_int
                segment_slices.append(slice(start, L_new))

                # ----- Step 3: loop over segments (exact reset logic as trainer) -----
                hidden = self._persistent_hidden_eval
                prev_basin_id = None

                # stitched time outputs (flat, then reshape back to [B, L, ...])
                stitched: Dict[str, List[torch.Tensor]] = {}

                for i_seg, seg_slice in enumerate(segment_slices):
                    seg_data = self._slice_time_segment(reshaped_data, seg_slice)
                    seg_basin_ids = seg_data["basin_idx"].view(-1)
                    basin_id_this_seg = int(seg_basin_ids[0].item())
                    seg_len = int(seg_basin_ids.numel())

                    # EXACT reset logic from trainer:
                    if i_seg == 0:
                        if self._current_basin_idx_eval is None or basin_id_this_seg != self._current_basin_idx_eval:
                            hidden = None
                    else:
                        if prev_basin_id is not None and basin_id_this_seg != prev_basin_id:
                            hidden = None

                    seg_preds = model(seg_data, hidden_state=hidden)

                    # update hidden (detach)
                    new_hidden = seg_preds.get("hidden_state", None)
                    if new_hidden is not None:
                        h, c = new_hidden
                        hidden = (h.detach(), c.detach())
                    else:
                        hidden = None

                    # loss for this segment
                    _, seg_all_losses = self.loss_obj(seg_preds, seg_data)

                    # Global timestep-weighted mean across loader (NaN-safe)
                    for k, v in seg_all_losses.items():
                        if torch.is_tensor(v):
                            v_item = float(v.detach().cpu().item())
                        else:
                            v_item = float(v)
                        if np.isfinite(v_item):
                            loss_sum[k] = loss_sum.get(k, 0.0) + v_item * seg_len
                            loss_w[k] = loss_w.get(k, 0) + seg_len

                    # Stitch time-like outputs.
                    # RULE: only stitch tensors that clearly have a time axis = seg_len
                    for k, v in seg_preds.items():
                        if v is None or isinstance(v, dict):
                            continue
                        if k == "hidden_state":
                            continue

                        if not torch.is_tensor(v):
                            continue

                        # Normalize shapes so we can concatenate on "time axis"
                        # Accept:
                        #   [1, seg_len, ...]   (preferred)
                        #   [seg_len, ...]      (no batch dim)
                        # Reject:
                        #   [1, ...] or [layers, batch, hidden] etc. (e.g. h_n/c_n)
                        v_norm = None
                        if v.dim() >= 2 and v.shape[0] == 1 and v.shape[1] == seg_len:
                            v_norm = v
                        elif v.dim() >= 1 and v.shape[0] == seg_len:
                            v_norm = v.unsqueeze(0)  # [1, seg_len, ...]
                        else:
                            # Not a time-series tensor; skip
                            continue

                        stitched.setdefault(k, []).append(v_norm.detach())

                    prev_basin_id = basin_id_this_seg

                # persist last hidden state across batches (same as trainer)
                self._persistent_hidden_eval = hidden
                self._current_basin_idx_eval = prev_basin_id

                # Build a full-batch predictions dict (time outputs reshaped back to [B, L, ...])
                full_predictions: Dict[str, torch.Tensor] = {}
                for k, parts in stitched.items():
                    flat = torch.cat(parts, dim=1)  # [1, B*L, ...]
                    if flat.shape[1] != B * L:
                        raise RuntimeError(
                            f"Persistent eval stitch mismatch for key={k}: got {flat.shape[1]} steps, expected {B*L}."
                        )
                    new_shape = (B, L) + tuple(flat.shape[2:])
                    full_predictions[k] = flat.reshape(*new_shape)

                # Save all_output if requested
                if all_output:
                    for key, value in full_predictions.items():
                        all_output[key].append(value.detach().cpu().numpy())
                elif save_all_output:
                    all_output = {key: [value.detach().cpu().numpy()] for key, value in full_predictions.items()}

                # Collect preds/obs/dates using original logic
                if predict_last_n[freq] == 0:
                    continue

                freq_key = ""  # single freq
                y_hat_sub, y_sub = self._subset_targets(model, data, full_predictions, predict_last_n[freq], freq_key)
                date_sub = data[f"date{freq_key}"][:, -predict_last_n[freq] :]

                if freq not in preds:
                    preds[freq] = y_hat_sub.detach().cpu()
                    obs[freq] = y_sub.detach().cpu()
                    dates[freq] = date_sub
                else:
                    preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                    obs[freq] = torch.cat((obs[freq], y_sub.detach().cpu()), 0)
                    dates[freq] = np.concatenate((dates[freq], date_sub), axis=0)

        # finalize arrays
        for f in preds.keys():
            preds[f] = preds[f].numpy()
            obs[f] = obs[f].numpy()

        for key, list_of_data in all_output.items():
            all_output[key] = np.concatenate(list_of_data, 0)

        # Global timestep-weighted mean losses over entire loader
        mean_losses = {}
        if len(loss_sum) == 0:
            mean_losses["loss"] = np.nan
        else:
            for k in loss_sum.keys():
                if loss_w.get(k, 0) > 0:
                    mean_losses[k] = loss_sum[k] / loss_w[k]
                else:
                    mean_losses[k] = np.nan

        return preds, obs, dates, mean_losses, all_output

    def _get_predictions_and_loss(self, model: BaseModel, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        predictions = model(data)
        _, all_losses = self.loss_obj(predictions, data)
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(
        self, model: BaseModel, data: Dict[str, torch.Tensor], predictions: dict, predict_last_n: int, freq: str
    ):
        raise NotImplementedError

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        raise NotImplementedError


class RegressionTester(BaseTester):
    """Tester class to run inference on a regression model."""

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        super(RegressionTester, self).__init__(cfg, run_dir, period, init_model)

    def _subset_targets(
        self, model: BaseModel, data: Dict[str, torch.Tensor], predictions: dict, predict_last_n: int, freq: str
    ):
        # freq is "" for single-frequency, or "_<freq>" for multi-frequency
        y_hat_sub = predictions[f"y_hat{freq}"][:, -predict_last_n:, :]
        y_sub = data[f"y{freq}"][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f"{var}_obs"] = (("date", "time_step"), y[:, :, i])
            data[f"{var}_sim"] = (("date", "time_step"), y_hat[:, :, i])
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.regression_plot(qobs, qsim, title)


class UncertaintyTester(BaseTester):
    """Tester class to run inference on an uncertainty model."""

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        super(UncertaintyTester, self).__init__(cfg, run_dir, period, init_model)

    def _get_predictions_and_loss(self, model: BaseModel, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        outputs = model(data)
        _, all_losses = self.loss_obj(outputs, data)
        predictions = model.sample(data, self.cfg.n_samples)
        model.eval()
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(
        self, model: BaseModel, data: Dict[str, torch.Tensor], predictions: dict, predict_last_n: int, freq: str = None
    ):
        y_hat_sub = predictions[f"y_hat{freq}"][:, -predict_last_n:, :]
        y_sub = data[f"y{freq}"][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f"{var}_obs"] = (("date", "time_step"), y[:, :, i])
            data[f"{var}_sim"] = (("date", "time_step", "samples"), y_hat[:, :, i, :])
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.uncertainty_plot(qobs, qsim, title)
