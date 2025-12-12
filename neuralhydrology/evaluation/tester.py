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
from neuralhydrology.datautils.utils import (
    get_frequency_factor,
    load_basin_file,
    load_scaler,
    sort_frequencies,
)
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
    """Base class to run inference on a model.

    Use subclasses of this class to evaluate a trained model on its train, test, or validation period.
    For regression settings, `RegressionTester` is used; for uncertainty prediction, `UncertaintyTester`.
    """

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

        self._load_run_data()

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(":")[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
                else:
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
        # get list of basins
        self.basins = load_basin_file(getattr(self.cfg, f"{self.period}_basin_file"))

        # load feature scaler
        self.scaler = load_scaler(self.run_dir)

        # check for old scaler files, where the center/scale parameters had still old names
        if "xarray_means" in self.scaler.keys():
            self.scaler["xarray_feature_center"] = self.scaler.pop("xarray_means")
        if "xarray_stds" in self.scaler.keys():
            self.scaler["xarray_feature_scale"] = self.scaler.pop("xarray_stds")

        # load basin_id to integer dictionary for one-hot-encoding
        if self.cfg.use_basin_id_encoding:
            self.id_to_int = load_basin_id_encoding(self.run_dir)

        for file in self.cfg.additional_feature_files:
            with open(file, "rb") as fp:
                self.additional_features.append(pickle.load(fp))

    def _get_weight_file(self, epoch: int):
        """Get file path to weight file"""
        if epoch is None:
            weight_file = sorted(list(self.run_dir.glob("model_epoch*.pt")))[-1]
        else:
            weight_file = self.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"
        return weight_file

    def _load_weights(self, epoch: int = None):
        """Load weights of a certain (or the last) epoch into the model."""
        weight_file = self._get_weight_file(epoch)
        LOGGER.info(f"Using the model weights from {weight_file}")
        self.model.load_state_dict(torch.load(weight_file, map_location=self.device))

    def _get_dataset(self, basin: str) -> BaseDataset:
        """Get dataset for a single basin."""
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

        # during validation, depending on settings, only evaluate on a random subset of basins
        basins = self.basins
        if self.period == "validation":
            if len(basins) > self.cfg.validate_n_random_basins:
                random.shuffle(basins)
                basins = basins[: self.cfg.validate_n_random_basins]

        # force model to train-mode when doing mc-dropout evaluation
        if self.cfg.mc_dropout:
            model.train()
        else:
            model.eval()

        results = defaultdict(dict)
        all_output = {basin: None for basin in basins}

        pbar = tqdm(basins, file=sys.stdout, disable=self._disable_pbar)
        pbar.set_description("# Validation" if self.period == "validation" else "# Evaluation")

        for basin in pbar:

            if self.cfg.cache_validation_data and basin in self.cached_datasets.keys():
                ds = self.cached_datasets[basin]
            else:
                try:
                    ds = self._get_dataset(basin)
                except NoEvaluationDataError:
                    continue
                if self.cfg.cache_validation_data and self.period == "validation":
                    self.cached_datasets[basin] = ds

            loader = DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                num_workers=0,
                collate_fn=ds.collate_fn,
            )

            # --- evaluation (persistent or normal) ---
            y_hat, y, dates, all_losses, all_output[basin] = self._evaluate(
                model, loader, ds.frequencies, save_all_output
            )

            # log loss of this basin plus number of samples in the logger to compute epoch aggregates later
            if experiment_logger is not None:
                experiment_logger.log_step(**{k: (v, len(loader)) for k, v in all_losses.items()})

            predict_last_n = self.cfg.predict_last_n
            seq_length = self.cfg.seq_length
            if isinstance(predict_last_n, int):
                predict_last_n = {ds.frequencies[0]: predict_last_n}
            if isinstance(seq_length, int):
                seq_length = {ds.frequencies[0]: seq_length}

            lowest_freq = sort_frequencies(ds.frequencies)[0]
            for freq in ds.frequencies:
                if predict_last_n[freq] == 0:
                    continue
                results[basin][freq] = {}

                # rescale observations
                feature_scaler = self.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values
                feature_center = self.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values
                y_freq = y[freq] * feature_scaler + feature_center

                # rescale predictions
                if y_hat[freq].ndim == 3 or (len(feature_scaler) == 1):
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                elif y_hat[freq].ndim == 4:
                    feature_scaler_ = np.expand_dims(feature_scaler, (0, 1, 3))
                    feature_center_ = np.expand_dims(feature_center, (0, 1, 3))
                    y_hat_freq = y_hat[freq] * feature_scaler_ + feature_center_
                else:
                    raise RuntimeError(
                        f"Simulations have {y_hat[freq].ndim} dimension. Only 3 and 4 are supported."
                    )

                data_vars = self._create_xarray_data_vars(y_hat_freq, y_freq)

                frequency_factor = int(get_frequency_factor(lowest_freq, freq))
                coords = {
                    "date": dates[lowest_freq][:, -1],
                    "time_step": (
                        (dates[freq][0, :] - dates[freq][0, -1]) / pd.Timedelta(freq)
                    ).astype(np.int64)
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

                # ------------------------------------------------------------------
                # Special handling for Persistent LSTM (single-frequency, persistent_state=True)
                # ------------------------------------------------------------------
                is_persistent = (
                    getattr(self.cfg, "persistent_state", False)
                    and self.cfg.model.lower() == "persistentlstm"
                    and len(ds.frequencies) == 1
                )

                if is_persistent:
                    if basin == basins[0]:
                        tqdm.write(
                            f"PersistentLSTM: metrics for {freq} use all "
                            f"{predict_last_n[freq]} time steps per sequence."
                        )

                    if metrics:
                        for target_variable in self.cfg.target_variables:
                            obs = xr[f"{target_variable}_obs"].stack(datetime=["date", "time_step"])
                            sim = xr[f"{target_variable}_sim"].stack(datetime=["date", "time_step"])

                            valid_mask = ~np.isnan(obs)
                            obs = obs[valid_mask]
                            sim = sim[valid_mask]

                            if target_variable in self.cfg.clip_targets_to_zero:
                                sim = xarray.where(sim < 0, 0, sim)

                            if "samples" in sim.dims:
                                sim = sim.mean(dim="samples")

                            var_metrics = metrics if isinstance(metrics, list) else metrics[target_variable]
                            if "all" in var_metrics:
                                var_metrics = get_available_metrics()

                            try:
                                values = calculate_metrics(obs, sim, metrics=var_metrics, resolution=freq)
                            except AllNaNError as err:
                                msg = (
                                    f"Basin {basin} "
                                    + (f"{target_variable} " if len(self.cfg.target_variables) > 1 else "")
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

                else:
                    # ---------------- ORIGINAL behaviour for ALL OTHER MODELS ----------------
                    freq_date_range = pd.date_range(
                        start=dates[lowest_freq][0, -1], end=dates[freq][-1, -1], freq=freq
                    )
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
                            obs = (
                                xr.isel(time_step=slice(-frequency_factor, None))
                                .stack(datetime=["date", "time_step"])
                                .drop_vars({"datetime", "date", "time_step"})[f"{target_variable}_obs"]
                            )
                            obs["datetime"] = freq_date_range

                            if not all(obs.isnull()):
                                sim = (
                                    xr.isel(time_step=slice(-frequency_factor, None))
                                    .stack(datetime=["date", "time_step"])
                                    .drop_vars({"datetime", "date", "time_step"})[f"{target_variable}_sim"]
                                )
                                sim["datetime"] = freq_date_range

                                if target_variable in self.cfg.clip_targets_to_zero:
                                    sim = xarray.where(sim < 0, 0, sim)

                                if "samples" in sim.dims:
                                    sim = sim.mean(dim="samples")

                                var_metrics = metrics if isinstance(metrics, list) else metrics[target_variable]
                                if "all" in var_metrics:
                                    var_metrics = get_available_metrics()

                                try:
                                    values = calculate_metrics(obs, sim, metrics=var_metrics, resolution=freq)
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
                    obs = xr[f"{target_var}_obs"].values
                    sim = xr[f"{target_var}_sim"].values
                    if target_var in self.cfg.clip_targets_to_zero:
                        sim = xarray.where(sim < 0, 0, sim)
                    figures.append(
                        self._get_plots(
                            obs,
                            sim,
                            title=f"{target_var} - Basin {basins[i]} - Epoch {epoch} - Frequency {freq}",
                        )[0]
                    )
                experiment_logger.log_figures(
                    figures,
                    freq,
                    preamble=re.sub(r"[^A-Za-z0-9\._\-]+", "", target_var),
                )

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
    # EVALUATION CORE
    # ------------------------------------------------------------------
    def _evaluate(
        self, model: BaseModel, loader: DataLoader, frequencies: List[str], save_all_output: bool = False
    ):
        """Evaluate model.

        For PersistentLSTM with persistent_state=True and a single frequency,
        this dispatches to `_evaluate_persistent`.
        """
        is_persistent = (
            getattr(self.cfg, "persistent_state", False)
            and self.cfg.model.lower() == "persistentlstm"
            and len(frequencies) == 1
        )

        if is_persistent:
            return self._evaluate_persistent(model, loader, frequencies, save_all_output)

        # ---------------- ORIGINAL (non-persistent) BEHAVIOUR ----------------
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
                        obs[freq] = torch.cat((obs[freq], y_sub.cpu()), 0)
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

    def _evaluate_persistent(
        self, model: BaseModel, loader: DataLoader, frequencies: List[str], save_all_output: bool = False
    ):
        """Persistent evaluation for PersistentLSTM.

        Processes each sequence one-by-one (within each batch) and carries hidden state.
        """
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            predict_last_n = {frequencies[0]: predict_last_n}

        preds, obs, dates, all_output = {}, {}, {}, {}
        losses = []

        persistent_hidden = None
        prev_basin_idx: Optional[int] = None

        with torch.no_grad():
            for data in loader:
                # move batch to device
                for key in data:
                    if key.startswith("x_d"):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith("date"):
                        data[key] = data[key].to(self.device)

                data = model.pre_model_hook(data, is_train=False)

                # ---- FIX #1: robust batch_size (do NOT depend on dict ordering) ----
                if "basin_idx" in data and torch.is_tensor(data["basin_idx"]):
                    batch_size = int(data["basin_idx"].shape[0])
                else:
                    # fallback: find any tensor with batch dimension
                    batch_size = None
                    for v in data.values():
                        if torch.is_tensor(v) and v.dim() >= 1:
                            batch_size = int(v.shape[0])
                            break
                        if isinstance(v, dict):
                            any_t = next(iter(v.values()))
                            if torch.is_tensor(any_t):
                                batch_size = int(any_t.shape[0])
                                break
                    if batch_size is None:
                        raise RuntimeError("Could not infer batch_size during persistent evaluation.")

                for b in range(batch_size):
                    # ---- build a single-sample view 'sample_b' ----
                    sample_b: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}
                    for key, val in data.items():
                        if key.startswith("date"):
                            # dates are numpy arrays
                            sample_b[key] = val[b : b + 1]
                        elif key.startswith("x_d"):
                            sample_b[key] = {feat: ten[b : b + 1] for feat, ten in val.items()}
                        else:
                            sample_b[key] = val[b : b + 1]

                    # ---- FIX #2: reset hidden if basin_idx changes (safety/future-proof) ----
                    if "basin_idx" in sample_b and torch.is_tensor(sample_b["basin_idx"]):
                        basin_now = int(sample_b["basin_idx"].item())
                        if prev_basin_idx is None:
                            prev_basin_idx = basin_now
                        elif basin_now != prev_basin_idx:
                            persistent_hidden = None
                            prev_basin_idx = basin_now

                    predictions_b = model(sample_b, hidden_state=persistent_hidden)

                    new_hidden = predictions_b.get("hidden_state", None)
                    if new_hidden is not None:
                        h, c = new_hidden
                        persistent_hidden = (h.detach(), c.detach())

                    _, all_losses_b = self.loss_obj(predictions_b, sample_b)
                    losses.append({k: v.item() for k, v in all_losses_b.items()})

                    if all_output:
                        for key, value in predictions_b.items():
                            if value is not None and type(value) != dict:
                                all_output[key].append(value.detach().cpu().numpy())
                    elif save_all_output:
                        all_output = {
                            key: [value.detach().cpu().numpy()]
                            for key, value in predictions_b.items()
                            if value is not None and type(value) != dict
                        }

                    for freq in frequencies:
                        if predict_last_n[freq] == 0:
                            continue
                        freq_key = "" if len(frequencies) == 1 else f"_{freq}"
                        y_hat_sub, y_sub = self._subset_targets(
                            model, sample_b, predictions_b, predict_last_n[freq], freq_key
                        )
                        date_sub = sample_b[f"date{freq_key}"][:, -predict_last_n[freq] :]

                        if freq not in preds:
                            preds[freq] = y_hat_sub.detach().cpu()
                            obs[freq] = y_sub.cpu()
                            dates[freq] = date_sub
                        else:
                            preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                            obs[freq] = torch.cat((obs[freq], y_sub.cpu()), 0)
                            dates[freq] = np.concatenate((dates[freq], date_sub), axis=0)

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

    def _get_predictions_and_loss(
        self, model: BaseModel, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        predictions = model(data)
        _, all_losses = self.loss_obj(predictions, data)
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(
        self,
        model: BaseModel,
        data: Dict[str, torch.Tensor],
        predictions: np.ndarray,
        predict_last_n: int,
        freq: str,
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
        self,
        model: BaseModel,
        data: Dict[str, torch.Tensor],
        predictions: np.ndarray,
        predict_last_n: np.ndarray,
        freq: str,
    ):
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

    def _get_predictions_and_loss(
        self, model: BaseModel, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        outputs = model(data)
        _, all_losses = self.loss_obj(outputs, data)
        predictions = model.sample(data, self.cfg.n_samples)
        model.eval()
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(
        self,
        model: BaseModel,
        data: Dict[str, torch.Tensor],
        predictions: np.ndarray,
        predict_last_n: int,
        freq: str = None,
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
