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

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}, optional
        The period to evaluate, by default 'test'.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
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
        """Evaluate the model.

        Parameters
        ----------
        epoch : int, optional
            Define a specific epoch to evaluate. By default, the weights of the last epoch are used.
        save_results : bool, optional
            If True, stores the evaluation results in the run directory. By default, True.
        save_all_output : bool, optional
            If True, stores all of the model output in the run directory. By default, False.
        metrics : Union[list, dict], optional
            List of metrics to compute during evaluation. Can also be a dict that specifies per-target metrics
        model : torch.nn.Module, optional
            If a model is passed, this is used for validation.
        experiment_logger : Logger, optional
            Logger can be passed during training to log metrics

        Returns
        -------
        dict
            A dictionary containing one xarray per basin with the evaluation results.
        """
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
                    # skip basin
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
            # if predict_last_n/seq_length are int, there's only one frequency
            if isinstance(predict_last_n, int):
                predict_last_n = {ds.frequencies[0]: predict_last_n}
            if isinstance(seq_length, int):
                seq_length = {ds.frequencies[0]: seq_length}
            lowest_freq = sort_frequencies(ds.frequencies)[0]

            for freq in ds.frequencies:
                if predict_last_n[freq] == 0:
                    continue  # this frequency is not being predicted
                results[basin][freq] = {}

                # rescale observations
                feature_scaler = self.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values
                feature_center = self.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values
                y_freq = y[freq] * feature_scaler + feature_center
                # rescale predictions
                if y_hat[freq].ndim == 3 or (len(feature_scaler) == 1):
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                elif y_hat[freq].ndim == 4:
                    # if y_hat has 4 dim and we have multiple features we expand the dimensions for scaling
                    feature_scaler = np.expand_dims(feature_scaler, (0, 1, 3))
                    feature_center = np.expand_dims(feature_center, (0, 1, 3))
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                else:
                    raise RuntimeError(
                        f"Simulations have {y_hat[freq].ndim} dimension. Only 3 and 4 are supported."
                    )

                # Create data_vars dictionary for the xarray.Dataset
                data_vars = self._create_xarray_data_vars(y_hat_freq, y_freq)

                # freq_range are the steps of the current frequency at each lowest-frequency step
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
                    # For Persistent LSTM we use ALL predict_last_n steps per sequence.
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
                    # ------------------------------------------------------------------
                    # ORIGINAL behaviour (unchanged) for ALL OTHER MODELS
                    # ------------------------------------------------------------------
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

        results_to_save = None
        states_to_save = None
        if save_results:
            results_to_save = results
        if save_all_output:
            states_to_save = all_output
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
        """Store results in various formats to disk."""
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
        this dispatches to a persistent evaluator.

        Two modes:
          - "simple": sequence-by-sequence in each batch, carry hidden across sequences
          - "mirror": mirrors BaseTrainer flatten+segment logic ([B,L]->[1,B*L] and basin segments)

        Set via cfg.persistent_eval_mode in {"simple","mirror"}.
        Default is "simple" (backward-safe).
        """
        is_persistent = (
            getattr(self.cfg, "persistent_state", False)
            and self.cfg.model.lower() == "persistentlstm"
            and len(frequencies) == 1
        )

        if is_persistent:
            mode = getattr(self.cfg, "persistent_eval_mode", "simple")
            mode = str(mode).lower()
            if mode not in ["simple", "mirror"]:
                LOGGER.warning(f"Unknown persistent_eval_mode={mode}; falling back to 'simple'.")
                mode = "simple"
            if mode == "mirror":
                return self._evaluate_persistent_mirror(model, loader, frequencies, save_all_output)
            return self._evaluate_persistent_simple(model, loader, frequencies, save_all_output)

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

    def _get_batch_size_from_data(self, data: Dict[str, object]) -> int:
        """Robustly infer batch size B from NH batch dict."""
        # Prefer any x_d tensor
        for k, v in data.items():
            if k.startswith("x_d") and isinstance(v, dict) and len(v) > 0:
                any_feat = next(iter(v.values()))
                if torch.is_tensor(any_feat) and any_feat.dim() >= 2:
                    return int(any_feat.shape[0])
        # Fallback: any tensor with dim>=1
        for _, v in data.items():
            if torch.is_tensor(v) and v.dim() >= 1:
                return int(v.shape[0])
        raise RuntimeError("Could not infer batch size from evaluation batch.")

    # ---------------- Persistent evaluation: SIMPLE ----------------
    def _evaluate_persistent_simple(
        self, model: BaseModel, loader: DataLoader, frequencies: List[str], save_all_output: bool = False
    ):
        """Persistent evaluation for PersistentLSTM (simple mode).

        Processes sequences one-by-one within each batch and carries hidden state across sequences.
        This is stable and backward-safe, but does NOT mirror the trainer's flatten+segment logic.
        """
        freq = frequencies[0]
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            predict_last_n = {freq: predict_last_n}

        preds, obs, dates, all_output = {}, {}, {}, {}
        losses = []

        persistent_hidden = None

        with torch.no_grad():
            for data in loader:
                for key in data:
                    if key.startswith("x_d"):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith("date"):
                        data[key] = data[key].to(self.device)

                data = model.pre_model_hook(data, is_train=False)
                batch_size = self._get_batch_size_from_data(data)

                for b in range(batch_size):
                    sample_b: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}
                    for key, val in data.items():
                        if key.startswith("date"):
                            sample_b[key] = val[b : b + 1]
                        elif key.startswith("x_d"):
                            sample_b[key] = {feat: ten[b : b + 1] for feat, ten in val.items()}
                        else:
                            # basin_idx, y, x_s, ...
                            sample_b[key] = val[b : b + 1]

                    predictions_b = model(sample_b, hidden_state=persistent_hidden)

                    new_hidden = predictions_b.get("hidden_state", None)
                    if new_hidden is not None:
                        h, c = new_hidden
                        persistent_hidden = (h.detach(), c.detach())
                    else:
                        persistent_hidden = None

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

                    # collect predictions/targets/dates
                    if predict_last_n[freq] == 0:
                        continue
                    freq_key = ""  # single frequency
                    y_hat_sub, y_sub = self._subset_targets(model, sample_b, predictions_b, predict_last_n[freq], freq_key)
                    date_sub = sample_b[f"date{freq_key}"][:, -predict_last_n[freq] :]

                    if freq not in preds:
                        preds[freq] = y_hat_sub.detach().cpu()
                        obs[freq] = y_sub.detach().cpu()
                        dates[freq] = date_sub
                    else:
                        preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                        obs[freq] = torch.cat((obs[freq], y_sub.detach().cpu()), 0)
                        dates[freq] = np.concatenate((dates[freq], date_sub), axis=0)

        for f in preds.keys():
            preds[f] = preds[f].numpy()
            obs[f] = obs[f].numpy()

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

    # ---------------- Persistent evaluation: MIRROR TRAINING ----------------
    def _evaluate_persistent_mirror(
        self, model: BaseModel, loader: DataLoader, frequencies: List[str], save_all_output: bool = False
    ):
        """Persistent evaluation that mirrors BaseTrainer flatten+segment logic.

        - Flatten [B,L,...] -> [1,B*L,...]
        - Segment by basin_idx change points in the flattened stream
        - Reset hidden when basin changes (intra-batch and inter-batch)
        """
        freq = frequencies[0]
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            predict_last_n = {freq: predict_last_n}

        preds, obs, dates, all_output = {}, {}, {}, {}
        losses = []

        hidden = None
        prev_basin_id = None

        def _flatten_time_batch(t: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(t) or t.dim() < 2:
                return t
            b, l = t.shape[0], t.shape[1]
            return t.reshape(1, b * l, *t.shape[2:])

        def _slice_time_segment(data_dict: Dict[str, object], segment_slice: slice) -> Dict[str, object]:
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
                    seg[key] = val  # handled separately (numpy)
                elif torch.is_tensor(val) and val.dim() >= 2:
                    seg[key] = val[:, segment_slice, ...]
                else:
                    seg[key] = val
            return seg

        with torch.no_grad():
            for data in loader:
                for key in data:
                    if key.startswith("x_d"):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith("date"):
                        data[key] = data[key].to(self.device)

                data = model.pre_model_hook(data, is_train=False)

                # ---- reshape ----
                reshaped_data: Dict[str, object] = {}
                for key, val in data.items():
                    if key.startswith("x_d"):
                        reshaped_data[key] = {feat: _flatten_time_batch(v) for feat, v in val.items()}
                    elif key.startswith("x_s"):
                        reshaped_data[key] = val
                    elif key.startswith("y"):
                        reshaped_data[key] = _flatten_time_batch(val)
                    elif key.startswith("basin_idx"):
                        if torch.is_tensor(val) and val.dim() == 1:
                            # infer L from any x_d tensor
                            L_inferred = None
                            for dk, dv in data.items():
                                if dk.startswith("x_d"):
                                    any_feat = next(iter(dv.values()))
                                    L_inferred = any_feat.shape[1]
                                    break
                            if L_inferred is None:
                                raise RuntimeError("Could not infer sequence length L to expand basin_idx.")
                            B = val.shape[0]
                            basin_2d = val.view(B, 1).expand(B, L_inferred)
                            reshaped_data[key] = _flatten_time_batch(basin_2d)
                        else:
                            reshaped_data[key] = _flatten_time_batch(val)
                    elif key.startswith("date"):
                        reshaped_data[key] = val  # numpy
                    else:
                        if torch.is_tensor(val) and val.dim() >= 2:
                            reshaped_data[key] = _flatten_time_batch(val)
                        else:
                            reshaped_data[key] = val

                basin_indices_flat = reshaped_data["basin_idx"].view(-1)  # [B*L]
                L_new = basin_indices_flat.numel()
                if L_new == 0:
                    continue

                diff = basin_indices_flat[1:] - basin_indices_flat[:-1]
                change_points = torch.where(diff != 0)[0] + 1

                segment_slices = []
                start = 0
                for cp in change_points:
                    cp_int = int(cp.item())
                    segment_slices.append(slice(start, cp_int))
                    start = cp_int
                segment_slices.append(slice(start, L_new))

                # flatten dates for slicing (keep numpy)
                date_key = "date"  # single frequency -> no suffix
                date_block = reshaped_data.get(date_key, None)
                date_flat = None
                if date_block is not None and hasattr(date_block, "reshape"):
                    date_flat = date_block.reshape(-1, *date_block.shape[2:])

                for seg_slice in segment_slices:
                    seg_data = _slice_time_segment(reshaped_data, seg_slice)

                    seg_basin_ids = seg_data["basin_idx"].view(-1)
                    basin_id_this_seg = int(seg_basin_ids[0].item())

                    if prev_basin_id is None or basin_id_this_seg != prev_basin_id:
                        hidden = None

                    preds_seg = model(seg_data, hidden_state=hidden)

                    new_hidden = preds_seg.get("hidden_state", None)
                    if new_hidden is not None:
                        h, c = new_hidden
                        hidden = (h.detach(), c.detach())
                    else:
                        hidden = None

                    _, all_losses_seg = self.loss_obj(preds_seg, seg_data)
                    losses.append({k: v.item() for k, v in all_losses_seg.items()})

                    if all_output:
                        for key, value in preds_seg.items():
                            if value is not None and type(value) != dict:
                                all_output[key].append(value.detach().cpu().numpy())
                    elif save_all_output:
                        all_output = {
                            key: [value.detach().cpu().numpy()]
                            for key, value in preds_seg.items()
                            if value is not None and type(value) != dict
                        }

                    if predict_last_n[freq] == 0:
                        prev_basin_id = basin_id_this_seg
                        continue

                    freq_key = ""
                    y_hat_sub, y_sub = self._subset_targets(model, seg_data, preds_seg, predict_last_n[freq], freq_key)

                    # slice dates in the same segment order
                    if date_flat is None:
                        # fallback: use model/data-provided dates if available
                        date_sub = seg_data.get(f"date{freq_key}", None)
                        if date_sub is None:
                            raise RuntimeError("Dates are missing; cannot build evaluation outputs.")
                        date_sub = date_sub[:, -predict_last_n[freq] :]
                    else:
                        date_seg = date_flat[seg_slice]
                        date_last = date_seg[-predict_last_n[freq] :]
                        date_sub = date_last.reshape(1, *date_last.shape)

                    if freq not in preds:
                        preds[freq] = y_hat_sub.detach().cpu()
                        obs[freq] = y_sub.detach().cpu()
                        dates[freq] = date_sub
                    else:
                        preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                        obs[freq] = torch.cat((obs[freq], y_sub.detach().cpu()), 0)
                        dates[freq] = np.concatenate((dates[freq], date_sub), axis=0)

                    prev_basin_id = basin_id_this_seg

        for f in preds.keys():
            preds[f] = preds[f].numpy()
            obs[f] = obs[f].numpy()

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
