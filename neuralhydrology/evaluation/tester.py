import logging
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

        # persistent hidden for evaluation (reset per basin)
        self._persistent_hidden_eval: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

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

    # ------------------------------------------------------------------
    # Chronological loader inside a basin (for persistent eval)
    # IMPORTANT: drop_last=True to match your basetrainer sampler.
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.array(x)

    def _infer_sample_start_time(self, sample: Dict[str, object]) -> Optional[np.int64]:
        if "date" in sample:
            try:
                d = self._safe_to_numpy(sample["date"]).reshape(-1)[0]
                if np.issubdtype(np.array(d).dtype, np.datetime64):
                    return np.array(d).astype("datetime64[s]").astype(np.int64)
                return np.int64(d)
            except Exception:
                return None
        return None

    def _get_chrono_loader(self, ds: BaseDataset) -> DataLoader:
        n = len(ds)
        if n == 0:
            return DataLoader(ds, batch_size=self.cfg.batch_size, num_workers=0, collate_fn=ds.collate_fn)

        items: List[Tuple[int, Union[int, np.int64]]] = []
        for i in range(n):
            sample = ds[i]
            t0 = self._infer_sample_start_time(sample)
            if t0 is None:
                t0 = np.int64(i)
            items.append((i, t0))

        items_sorted = sorted(items, key=lambda x: x[1])
        sorted_indices = [i for i, _ in items_sorted]

        batch_size = int(self.cfg.batch_size)
        batches = [sorted_indices[i: i + batch_size] for i in range(0, len(sorted_indices), batch_size)]
        # MATCH TRAINING: drop_last=True
        batches = [b for b in batches if len(b) == batch_size]

        class _ListBatchSampler(torch.utils.data.Sampler):
            def __init__(self, batches_):
                self.batches_ = batches_

            def __iter__(self):
                yield from self.batches_

            def __len__(self):
                return len(self.batches_)

        return DataLoader(ds, batch_sampler=_ListBatchSampler(batches), num_workers=0, collate_fn=ds.collate_fn)

    # ------------------------------------------------------------------
    # Persistent switches
    # ------------------------------------------------------------------
    def _is_persistent_mirror_eval(self, frequencies: List[str]) -> bool:
        return (
            getattr(self.cfg, "persistent_state", False)
            and self.cfg.model.lower() == "persistentlstm"
            and len(frequencies) == 1
        )

    def _is_persistent_continuous_eval(self, frequencies: List[str]) -> bool:
        if not getattr(self.cfg, "persistent_state", False):
            return False
        if not getattr(self.cfg, "non_overlapping_sequences", False):
            return False
        if len(frequencies) != 1:
            return False

        seq_stride = int(getattr(self.cfg, "seq_stride", 1))

        seq_length = self.cfg.seq_length
        if isinstance(seq_length, dict):
            freq0 = frequencies[0]
            seq_length = int(seq_length[freq0])
        else:
            seq_length = int(seq_length)

        return seq_stride in (1, seq_length)

    # ------------------------------------------------------------------
    # SAME flatten utilities as your BaseTrainer
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_time_batch(t: torch.Tensor) -> torch.Tensor:
        """Flatten [B, L, ...] -> [1, B*L, ...]."""
        if not torch.is_tensor(t) or t.dim() < 2:
            return t
        t = t.contiguous()
        b, l = t.shape[0], t.shape[1]
        return t.view(1, b * l, *t.shape[2:])

    def _slice_time_segment(self, data_dict: Dict[str, object], segment_slice: slice) -> Dict[str, object]:
        """Slice time segment from already reshaped tensors (expects [1, T, ...])."""
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
                seg[key] = val
            elif torch.is_tensor(val) and val.dim() >= 2:
                seg[key] = val[:, segment_slice, ...]
            else:
                seg[key] = val
        return seg

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        epoch: int = None,
        save_results: bool = True,
        save_all_output: bool = False,
        metrics: Union[list, dict] = [],
        model: torch.nn.Module = None,
        experiment_logger: Logger = None,
    ) -> dict:
        if model is None:
            if self.init_model:
                self._load_weights(epoch=epoch)
                model = self.model
            else:
                raise RuntimeError("No model was initialized for the evaluation")

        basins = list(self.basins)
        if self.period == "validation":
            if len(basins) > self.cfg.validate_n_random_basins:
                random.shuffle(basins)
                basins = basins[: self.cfg.validate_n_random_basins]

        if self.cfg.mc_dropout:
            model.train()
        else:
            model.eval()

        results = defaultdict(dict)
        all_output = {basin: None for basin in basins}

        pbar = tqdm(basins, file=sys.stdout, disable=self._disable_pbar)
        pbar.set_description("# Validation" if self.period == "validation" else "# Evaluation")

        for basin in pbar:
            # reset hidden per basin (no leakage across basins)
            self._persistent_hidden_eval = None

            if self.cfg.cache_validation_data and basin in self.cached_datasets.keys():
                ds = self.cached_datasets[basin]
            else:
                try:
                    ds = self._get_dataset(basin)
                except NoEvaluationDataError:
                    continue
                if self.cfg.cache_validation_data and self.period == "validation":
                    self.cached_datasets[basin] = ds

            loader = self._get_chrono_loader(ds)

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
            persistent_continuous = self._is_persistent_continuous_eval(ds.frequencies)

            for freq in ds.frequencies:
                if predict_last_n[freq] == 0:
                    continue

                results[basin][freq] = {}

                feature_scaler = self.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values
                feature_center = self.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values
                y_freq = y[freq] * feature_scaler + feature_center

                if y_hat[freq].ndim == 3 or (len(feature_scaler) == 1):
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                elif y_hat[freq].ndim == 4:
                    feature_scaler_ = np.expand_dims(feature_scaler, (0, 1, 3))
                    feature_center_ = np.expand_dims(feature_center, (0, 1, 3))
                    y_hat_freq = y_hat[freq] * feature_scaler_ + feature_center_
                else:
                    raise RuntimeError(f"Simulations have {y_hat[freq].ndim} dimension. Only 3 and 4 are supported.")

                if persistent_continuous:
                    dt_1d = pd.to_datetime(dates[freq].reshape(-1))
                    y_obs_1d = y_freq.reshape(-1, y_freq.shape[-1])

                    if y_hat_freq.ndim == 3:
                        y_sim_1d = y_hat_freq.reshape(-1, y_hat_freq.shape[-1])
                        data_vars = {}
                        for i, var in enumerate(self.cfg.target_variables):
                            data_vars[f"{var}_obs"] = (("datetime",), y_obs_1d[:, i])
                            data_vars[f"{var}_sim"] = (("datetime",), y_sim_1d[:, i])
                    else:
                        y_sim_1d = y_hat_freq.reshape(-1, y_hat_freq.shape[2], y_hat_freq.shape[3])
                        data_vars = {}
                        for i, var in enumerate(self.cfg.target_variables):
                            data_vars[f"{var}_obs"] = (("datetime",), y_obs_1d[:, i])
                            data_vars[f"{var}_sim"] = (("datetime", "samples"), y_sim_1d[:, i, :])

                    xr = xarray.Dataset(data_vars=data_vars, coords={"datetime": dt_1d})
                    results[basin][freq]["xr"] = xr

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

                    continue

                # original NH xarray path
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

    # ------------------------------------------------------------------
    # Core _evaluate()
    # ------------------------------------------------------------------
    def _evaluate(self, model: BaseModel, loader: DataLoader, frequencies: List[str], save_all_output: bool = False):
        if self._is_persistent_mirror_eval(frequencies):
            return self._evaluate_persistent_match_training_with_flatten(model, loader, frequencies, save_all_output)

        # original NH non-persistent path
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
                predictions = model(data)
                _, all_losses = self.loss_obj(predictions, data)
                losses.append({k: v.item() for k, v in all_losses.items()})

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
                    date_sub = data[f"date{freq_key}"][:, -predict_last_n[freq]:]

                    if freq not in preds:
                        preds[freq] = y_hat_sub.detach().cpu()
                        obs[freq] = y_sub.cpu()
                        dates[freq] = date_sub
                    else:
                        preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                        obs[freq] = torch.cat((obs[freq], y_sub.detach().cpu()), 0)
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
                mean_losses[loss_name] = float(np.nanmean([d[loss_name] for d in losses]))

        return preds, obs, dates, mean_losses, all_output

    # ------------------------------------------------------------------
    # Persistent eval that MATCHES your BaseTrainer flatten logic
    # ------------------------------------------------------------------
    def _evaluate_persistent_match_training_with_flatten(
        self, model: BaseModel, loader: DataLoader, frequencies: List[str], save_all_output: bool = False
    ):
        freq = frequencies[0]
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            predict_last_n = {freq: predict_last_n}

        preds, obs, dates, all_output = {}, {}, {}, {}
        losses = []

        hidden = self._persistent_hidden_eval  # starts None per basin (set/reset in evaluate())

        with torch.no_grad():
            for data in loader:
                # move tensors
                for key in data.keys():
                    if key.startswith("x_d"):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith("date"):
                        data[key] = data[key].to(self.device)

                data = model.pre_model_hook(data, is_train=False)

                # infer B,L from x_d
                if "x_d" not in data or not isinstance(data["x_d"], dict) or len(data["x_d"]) == 0:
                    raise RuntimeError("Persistent eval: expected data['x_d'] dict with dynamic inputs.")
                any_xd = next(iter(data["x_d"].values()))
                if any_xd.dim() < 2:
                    raise RuntimeError("Persistent eval: expected x_d tensors with shape [B, L, ...].")
                B_inferred, L_inferred = int(any_xd.shape[0]), int(any_xd.shape[1])

                # flatten like trainer -> batch=1, time=B*L
                reshaped: Dict[str, object] = {}
                for key, val in data.items():
                    if key.startswith("x_d"):
                        reshaped[key] = {feat: self._flatten_time_batch(v) for feat, v in val.items()}
                    elif key.startswith("x_s"):
                        reshaped[key] = val
                    elif key.startswith("y") and torch.is_tensor(val):
                        reshaped[key] = self._flatten_time_batch(val)
                    elif key == "basin_idx" and torch.is_tensor(val):
                        if val.dim() == 1:
                            basin_2d = val.view(B_inferred, 1).expand(B_inferred, L_inferred)
                        else:
                            basin_2d = val
                        reshaped[key] = self._flatten_time_batch(basin_2d)
                    else:
                        if torch.is_tensor(val) and val.dim() >= 2:
                            reshaped[key] = self._flatten_time_batch(val)
                        else:
                            reshaped[key] = val

                seg_len = B_inferred * L_inferred
                seg_data = self._slice_time_segment(reshaped, slice(0, seg_len))

                # force x_s to batch=1 (same as trainer)
                if "x_s" in data:
                    if isinstance(data["x_s"], dict):
                        seg_data["x_s"] = {feat: v[0:1, ...] for feat, v in data["x_s"].items()}
                    else:
                        seg_data["x_s"] = data["x_s"][0:1, ...]

                # forward with hidden
                pred = model(seg_data, hidden_state=hidden)

                # update hidden (shape should remain [layers, 1, H] always)
                new_hidden = pred.get("hidden_state", None)
                if new_hidden is not None:
                    h, c = new_hidden
                    hidden = (h.detach(), c.detach())
                else:
                    hidden = None

                # losses for logging
                _, all_losses = self.loss_obj(pred, seg_data)
                losses.append({k: float(v.detach().cpu().item()) for k, v in all_losses.items()})

                # build "full_predictions" back to [B,L,...] for metrics code
                full_predictions: Dict[str, torch.Tensor] = {}
                for k, v in pred.items():
                    if v is None or isinstance(v, dict) or k == "hidden_state":
                        continue
                    if not torch.is_tensor(v):
                        continue
                    if v.dim() >= 2 and v.shape[0] == 1 and v.shape[1] == seg_len:
                        full_predictions[k] = v.reshape(B_inferred, L_inferred, *v.shape[2:])

                # optional save_all_output
                if save_all_output:
                    if all_output:
                        for k, v in full_predictions.items():
                            all_output.setdefault(k, []).append(v.detach().cpu().numpy())
                    else:
                        all_output = {k: [v.detach().cpu().numpy()] for k, v in full_predictions.items()}

                # store preds/obs/dates for metrics
                if predict_last_n[freq] == 0:
                    continue
                freq_key = ""
                y_hat_sub, y_sub = self._subset_targets(model, data, full_predictions, predict_last_n[freq], freq_key)
                date_sub = data[f"date{freq_key}"][:, -predict_last_n[freq]:]

                if freq not in preds:
                    preds[freq] = y_hat_sub.detach().cpu()
                    obs[freq] = y_sub.detach().cpu()
                    dates[freq] = date_sub
                else:
                    preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                    obs[freq] = torch.cat((obs[freq], y_sub.detach().cpu()), 0)
                    dates[freq] = np.concatenate((dates[freq], date_sub), axis=0)

        self._persistent_hidden_eval = hidden

        for f in preds.keys():
            preds[f] = preds[f].numpy()
            obs[f] = obs[f].numpy()

        if save_all_output:
            for k, vlist in all_output.items():
                all_output[k] = np.concatenate(vlist, axis=0)

        mean_losses = {}
        if len(losses) == 0:
            mean_losses["loss"] = np.nan
        else:
            for k in losses[0].keys():
                mean_losses[k] = float(np.nanmean([d[k] for d in losses]))

        return preds, obs, dates, mean_losses, all_output

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
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
