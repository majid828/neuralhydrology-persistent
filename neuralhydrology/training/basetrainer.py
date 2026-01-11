import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import neuralhydrology.training.loss as loss
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.tester import BaseTester
from neuralhydrology.modelzoo import get_model
from neuralhydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import setup_logging
from neuralhydrology.training.earlystopper import EarlyStopper

LOGGER = logging.getLogger(__name__)


class BaseTrainer(object):
    """Default class to train a model."""

    def __init__(self, cfg: Config):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.loss_obj = None
        self.experiment_logger = None
        self.loader = None
        self.validator = None
        self.noise_sampler_y = None
        self._target_mean = None
        self._target_std = None
        self._scaler = {}
        self._allow_subsequent_nan_losses = cfg.allow_subsequent_nan_losses
        self._disable_pbar = cfg.verbose == 0
        self._max_updates_per_epoch = cfg.max_updates_per_epoch
        self._early_stopping = cfg.early_stopping
        self._patience_early_stopping = cfg.patience_early_stopping
        self._minimum_epochs_before_early_stopping = cfg.minimum_epochs_before_early_stopping
        self._dynamic_learning_rate = cfg.dynamic_learning_rate
        self._patience_dynamic_learning_rate = cfg.patience_dynamic_learning_rate
        self._factor_dynamic_learning_rate = cfg.factor_dynamic_learning_rate

        # --------------------------------------------------
        # Persistent LSTM support (existing)
        # --------------------------------------------------
        self.persistent_state = getattr(self.cfg, "persistent_state", False)

        # --------------------------------------------------
        
        # --------------------------------------------------
        # If enabled, we save (h,c) per basin to disk and reload it next epoch.
        self.persist_state_across_epochs = getattr(self.cfg, "persist_state_across_epochs", False)
        self.shuffle_basins_each_epoch = getattr(self.cfg, "shuffle_basins_each_epoch", False)

        # Where to store per-basin states
        state_dir_cfg = getattr(self.cfg, "persistent_state_dir", "persistent_states")
        self.persistent_state_dir = None  # set after run_dir exists

        # load train basin list and add number of basins to the config
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # check at which epoch the training starts
        self._epoch = self._get_start_epoch_number()

        self._create_folder_structure()
        setup_logging(str(self.cfg.run_dir / "output.log"))
        LOGGER.info(f"### Folder structure created at {self.cfg.run_dir}")

        # MOD: now that run_dir exists, create persistent state directory (if enabled)
        if self.persistent_state and self.cfg.model.lower() == "persistentlstm" and self.persist_state_across_epochs:
            # relative path → store inside run_dir
            self.persistent_state_dir = (self.cfg.run_dir / state_dir_cfg).resolve()
            self.persistent_state_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"### MOD: persistent per-basin states will be stored in {self.persistent_state_dir}")

        if self.cfg.is_continue_training:
            LOGGER.info(f"### Continue training of run stored in {self.cfg.base_run_dir}")

        if self.cfg.is_finetuning:
            LOGGER.info(f"### Start finetuning with pretrained model stored in {self.cfg.base_run_dir}")

        LOGGER.info(f"### Run configurations for {self.cfg.experiment_name}")
        for key, val in self.cfg.as_dict().items():
            LOGGER.info(f"{key}: {val}")

        self._set_random_seeds()
        self._set_device()

    # ------------------------------------------------------------------
    # Helper methods for persistent LSTM reshaping and segment slicing
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_time_batch(t: torch.Tensor) -> torch.Tensor:
        """Flatten [B, L, ...] → [1, B*L, ...]."""
        if not torch.is_tensor(t) or t.dim() < 2:
            return t
        t = t.contiguous()
        b, l = t.shape[0], t.shape[1]
        return t.view(1, b * l, *t.shape[2:])

    def _slice_time_segment(self, data_dict: Dict[str, object], segment_slice: slice) -> Dict[str, object]:
        """Slice a contiguous time segment from reshaped data (expects [1, L_new, ...])."""
        seg = {}
        for key, val in data_dict.items():
            if key.startswith("x_d"):
                seg[key] = {feat: v[:, segment_slice, ...] for feat, v in val.items()}
            elif key.startswith("x_s"):
                seg[key] = val  # keep, but can overwrite (IMPORTANT: we will overwrite to batch=1 below)
            elif key.startswith("y") or key.startswith("basin_idx"):
                if torch.is_tensor(val):
                    if val.dim() == 1:
                        seg[key] = val[segment_slice]
                    else:
                        seg[key] = val[:, segment_slice, ...]
                else:
                    seg[key] = val
            elif torch.is_tensor(val) and val.dim() >= 2:
                seg[key] = val[:, segment_slice, ...]
            else:
                seg[key] = val
        return seg

    @staticmethod
    def _resolve_batch_row_for_basin(data: Dict[str, object], basin_id: int, B_expected: int) -> int:
        """
        Robust mapping: which original batch-row corresponds to `basin_id`.
        Prefers data['basin_idx'] if it is [B]. Falls back to [B,L] if present.
        """
        if "basin_idx" not in data or (not torch.is_tensor(data["basin_idx"])):
            # If basin_idx is missing, we cannot map; for basin-specific dataset, row 0 is safe.
            return 0

        bidx = data["basin_idx"]

        # common: [B]
        if bidx.dim() == 1 and int(bidx.shape[0]) == int(B_expected):
            matches = (bidx == basin_id).nonzero(as_tuple=True)[0]
            if matches.numel() == 0:
                # For basin-specific dataset, all rows should be same basin; fallback to 0
                return 0
            return int(matches[0].item())

        # sometimes: [B,L]
        if bidx.dim() == 2 and int(bidx.shape[0]) == int(B_expected):
            row_ids = bidx[:, 0]
            matches = (row_ids == basin_id).nonzero(as_tuple=True)[0]
            if matches.numel() == 0:
                return 0
            return int(matches[0].item())

        # unknown shape → safest fallback
        return 0

    def _get_dataset(self, basin: Optional[str] = None) -> BaseDataset:
        # MOD: allow basin-specific dataset for basin-by-basin training
        return get_dataset(cfg=self.cfg, period="train", is_train=True, scaler=self._scaler, basin=basin)

    def _get_model(self) -> torch.nn.Module:
        return get_model(cfg=self.cfg)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(model=self.model, cfg=self.cfg)

    def _get_loss_obj(self) -> loss.BaseLoss:
        return get_loss_obj(cfg=self.cfg)

    def _set_regularization(self):
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

    def _get_tester(self) -> BaseTester:
        return get_tester(cfg=self.cfg, run_dir=self.cfg.run_dir, period="validation", init_model=False)

    def _get_data_loader(self, ds: BaseDataset, shuffle: bool) -> torch.utils.data.DataLoader:
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=ds.collate_fn,
        )

    # ------------------------------------------------------------------
    # MOD: per-basin state file I/O
    # ------------------------------------------------------------------
    def _state_file_for_basin(self, basin: str) -> Path:
        if self.persistent_state_dir is None:
            raise RuntimeError("Persistent state dir is not initialized.")
        safe = basin.replace("/", "_")
        return self.persistent_state_dir / f"{safe}.pt"

    def _load_basin_state(self, basin: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """MOD: load (h,c) for this basin from disk. Returns None if not found."""
        f = self._state_file_for_basin(basin)
        if not f.exists():
            return None
        obj = torch.load(str(f), map_location="cpu")
        h = obj["h"]
        c = obj["c"]
        return (h.to(self.device), c.to(self.device))

    def _save_basin_state(self, basin: str, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        """MOD: save (h,c) for this basin to disk, detached and on CPU."""
        if hidden is None:
            return
        h, c = hidden
        payload = {"h": h.detach().to("cpu"), "c": c.detach().to("cpu")}
        f = self._state_file_for_basin(basin)
        torch.save(payload, str(f))

    def initialize_training(self):
        """Initialize the training class."""
        if self.cfg.is_finetuning:
            self._scaler = load_scaler(self.cfg.base_run_dir)

        ds = self._get_dataset()
        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")

        self.loader = self._get_data_loader(ds=ds, shuffle=True)

        self.model = self._get_model().to(self.device)

        if self.cfg.checkpoint_path is not None:
            LOGGER.info(f"Starting training from Checkpoint {self.cfg.checkpoint_path}")
            self.model.load_state_dict(torch.load(str(self.cfg.checkpoint_path), map_location=self.device))
        elif self.cfg.checkpoint_path is None and self.cfg.is_finetuning:
            checkpoint_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            LOGGER.info(f"Starting training from checkpoint {checkpoint_path}")
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))

        self.optimizer = self._get_optimizer()
        self.loss_obj = self._get_loss_obj().to(self.device)
        self._set_regularization()

        if self.cfg.is_continue_training:
            self._restore_training_state()

        self.experiment_logger = Logger(cfg=self.cfg)
        if self.cfg.log_tensorboard:
            self.experiment_logger.start_tb()

        if self.cfg.is_continue_training:
            self.experiment_logger.epoch = self._epoch
            self.experiment_logger.update = len(self.loader) * self._epoch

        if self.cfg.validate_every is not None:
            if self.cfg.validate_n_random_basins < 1:
                warn_msg = [
                    f"Validation set to validate every {self.cfg.validate_every} epoch(s), but ",
                    "'validate_n_random_basins' not set or set to zero. Will validate on the entire validation set."
                ]
                LOGGER.warning("".join(warn_msg))
                self.cfg.validate_n_random_basins = self.cfg.number_of_basins
            self.validator = self._get_tester()

        if self.cfg.target_noise_std is not None:
            self.noise_sampler_y = torch.distributions.Normal(loc=0, scale=self.cfg.target_noise_std)
            self._target_mean = torch.from_numpy(
                ds.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values
            ).to(self.device)
            self._target_std = torch.from_numpy(
                ds.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values
            ).to(self.device)

    def train_and_validate(self):
        """Train and validate the model."""
        if self._early_stopping:
            if self.cfg.is_continue_training:
                LOGGER.warning("Early stopping state is reset.")
            early_stopper = EarlyStopper(patience=self._patience_early_stopping, min_delta=0.0001)

        if self._dynamic_learning_rate:
            if self.cfg.is_continue_training:
                LOGGER.warning("Scheduler state is reset.")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=self._factor_dynamic_learning_rate,
                patience=self._patience_dynamic_learning_rate
            )

        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            if not self._dynamic_learning_rate:
                if epoch in self.cfg.learning_rate.keys():
                    LOGGER.info(f"Setting learning rate to {self.cfg.learning_rate[epoch]}")
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.cfg.learning_rate[epoch]

            self._train_epoch(epoch=epoch)

            avg_losses = self.experiment_logger.summarise()
            loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
            LOGGER.info(f"Epoch {epoch} average loss: {loss_str}")

            if epoch % self.cfg.save_weights_every == 0:
                self._save_weights_and_optimizer(epoch)

            if (self.validator is not None) and (epoch % self.cfg.validate_every == 0):
                self.validator.evaluate(
                    epoch=epoch,
                    save_results=self.cfg.save_validation_results,
                    save_all_output=self.cfg.save_all_output,
                    metrics=self.cfg.metrics,
                    model=self.model,
                    experiment_logger=self.experiment_logger.valid()
                )

                valid_metrics = self.experiment_logger.summarise()
                print_msg = f"Epoch {epoch} average validation loss: {valid_metrics['avg_total_loss']:.5f}"
                if self.cfg.metrics:
                    print_msg += f" -- Median validation metrics: "
                    print_msg += ", ".join(
                        f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != 'avg_total_loss'
                    )
                    LOGGER.info(print_msg)

                if self._early_stopping and epoch > self._minimum_epochs_before_early_stopping and \
                        early_stopper.check_early_stopping(valid_metrics['avg_total_loss']):
                    LOGGER.info(
                        f"Early stopping triggered at epoch {epoch} with validation loss "
                        f"{valid_metrics['avg_total_loss']:.5f}. Training stopped."
                    )
                    break

                if self._dynamic_learning_rate:
                    scheduler.step(valid_metrics['avg_total_loss'])

        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _get_start_epoch_number(self):
        if self.cfg.is_continue_training:
            if self.cfg.continue_from_epoch is not None:
                epoch = self.cfg.continue_from_epoch
            else:
                weight_path = [x for x in sorted(list(self.cfg.run_dir.glob('model_epoch*.pt')))][-1]
                epoch = weight_path.name[-6:-3]
        else:
            epoch = 0
        return int(epoch)

    def _restore_training_state(self):
        if self.cfg.continue_from_epoch is not None:
            epoch = f"{self.cfg.continue_from_epoch:03d}"
            weight_path = self.cfg.base_run_dir / f"model_epoch{epoch}.pt"
        else:
            weight_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            epoch = weight_path.name[-6:-3]

        optimizer_path = self.cfg.base_run_dir / f"optimizer_state_epoch{epoch}.pt"

        LOGGER.info(f"Continue training from epoch {int(epoch)}")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(str(optimizer_path), map_location=self.device))

    def _save_weights_and_optimizer(self, epoch: int):
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), str(weight_path))

        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.experiment_logger.train()

        # ==================================================
        # MOD: Basin-aware persistent save/load across epochs
        # ==================================================
        if self.persistent_state and self.cfg.model.lower() == "persistentlstm" and self.persist_state_across_epochs:
            basins = list(self.basins)

            if self.shuffle_basins_each_epoch:
                random.shuffle(basins)
                LOGGER.info(f"### MOD: shuffled basin order for epoch {epoch}: {basins}")

            nan_count = 0

            for basin in basins:
                # Epoch 1 starts with None (unless continuing training and files exist)
                if epoch == 1:
                    hidden = None
                else:
                    hidden = self._load_basin_state(basin)

                # basin-specific dataset + loader with NO shuffling (keeps chronology)
                ds_b = self._get_dataset(basin=basin)
                if len(ds_b) == 0:
                    continue
                loader_b = self._get_data_loader(ds=ds_b, shuffle=False)

                pbar = tqdm(loader_b, file=sys.stdout, disable=self._disable_pbar)
                pbar.set_description(f"# Epoch {epoch} | Basin {basin}")

                for i, data in enumerate(pbar):
                    if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                        break

                    # move batch to device
                    for key in data.keys():
                        if key.startswith('x_d'):
                            data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                        elif not key.startswith('date'):
                            data[key] = data[key].to(self.device)

                    data = self.model.pre_model_hook(data, is_train=True)

                    # Infer B,L from one dynamic feature
                    any_xd = None
                    if "x_d" in data and isinstance(data["x_d"], dict) and len(data["x_d"]) > 0:
                        any_xd = next(iter(data["x_d"].values()))
                    else:
                        # fallback: any key that starts with x_d
                        for k, v in data.items():
                            if k.startswith("x_d") and isinstance(v, dict) and len(v) > 0:
                                any_xd = next(iter(v.values()))
                                break
                    if any_xd is None or any_xd.dim() < 2:
                        raise RuntimeError("Persistent training: could not infer [B,L] from x_d.")
                    B_inferred, L_inferred = int(any_xd.shape[0]), int(any_xd.shape[1])

                    # --------------------------------------------------
                    # MOD: flatten [B,L] → [1,B*L] for dynamics/targets/etc.
                    # --------------------------------------------------
                    reshaped_data: Dict[str, object] = {}
                    for key, val in data.items():
                        if key.startswith("x_d"):
                            reshaped_data[key] = {feat: self._flatten_time_batch(v) for feat, v in val.items()}
                        elif key.startswith("x_s"):
                            # keep as-is for now, but MUST be overwritten to batch=1 per segment below
                            reshaped_data[key] = val
                        elif key.startswith("y"):
                            reshaped_data[key] = self._flatten_time_batch(val)
                        elif key.startswith("basin_idx"):
                            if torch.is_tensor(val) and val.dim() == 1:
                                # [B] → [B,L] → flatten
                                basin_2d = val.view(B_inferred, 1).expand(B_inferred, L_inferred)
                                reshaped_data[key] = self._flatten_time_batch(basin_2d)
                            else:
                                reshaped_data[key] = self._flatten_time_batch(val)
                        else:
                            if torch.is_tensor(val) and val.dim() >= 2:
                                reshaped_data[key] = self._flatten_time_batch(val)
                            else:
                                reshaped_data[key] = val

                    # Single segment per batch (same basin), but we still build seg_data
                    basin_idx_flat = reshaped_data.get("basin_idx", None)
                    seg_len = int(basin_idx_flat.numel()) if torch.is_tensor(basin_idx_flat) else (B_inferred * L_inferred)
                    seg_slice = slice(0, seg_len)
                    seg_data = self._slice_time_segment(reshaped_data, seg_slice)

                    # --------------------------------------------------
                    # CRITICAL FIX FOR OUR ERROR:
                    # Ensure static attributes have batch=1 to match flattened dynamics batch=1.
                    # Without this, InputLayer torch.cat() fails with "Expected size 1 but got 64".
                    # --------------------------------------------------
                    if "x_s" in data:
                        # Determine basin_id from seg_data basin_idx if available
                        basin_id_this_seg = None
                        if "basin_idx" in seg_data and torch.is_tensor(seg_data["basin_idx"]):
                            basin_id_this_seg = int(seg_data["basin_idx"].view(-1)[0].item())
                        else:
                            # basin-only dataset → any row works
                            basin_id_this_seg = int(data["basin_idx"][0].item()) if ("basin_idx" in data and torch.is_tensor(data["basin_idx"])) else 0

                        orig_b = self._resolve_batch_row_for_basin(data, basin_id_this_seg, B_inferred)

                        if isinstance(data["x_s"], dict):
                            seg_data["x_s"] = {feat: v[orig_b:orig_b + 1, ...] for feat, v in data["x_s"].items()}
                        else:
                            seg_data["x_s"] = data["x_s"][orig_b:orig_b + 1, ...]

                    preds = self.model(seg_data, hidden_state=hidden)

                    # Update hidden (detach) for continuity INSIDE this basin
                    new_hidden = preds.get("hidden_state", None)
                    if new_hidden is not None:
                        h, c = new_hidden
                        hidden = (h.detach(), c.detach())
                    else:
                        hidden = None

                    loss_val, all_losses = self.loss_obj(preds, seg_data)

                    # NaN handling
                    if torch.isnan(loss_val):
                        nan_count += 1
                        if nan_count > self._allow_subsequent_nan_losses:
                            raise RuntimeError(f"Loss was NaN for {nan_count} times in a row. Stopped training.")
                        LOGGER.warning(f"Loss is Nan; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")
                        continue
                    else:
                        nan_count = 0

                    self.optimizer.zero_grad()
                    loss_val.backward()

                    if self.cfg.clip_gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)

                    self.optimizer.step()

                    pbar.set_postfix_str(f"Loss: {loss_val.item():.4f}")
                    self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})

                # basin finished → SAVE its final (h,c) to disk
                self._save_basin_state(basin, hidden)

            return  # MOD: done (skip original multi-basin loop)

        # ==================================================
        # ORIGINAL training (non-basin-aware persistent)
        # ==================================================
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) \
            if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f'# Epoch {epoch}')

        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            for key in data.keys():
                if key.startswith('x_d'):
                    data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                elif not key.startswith('date'):
                    data[key] = data[key].to(self.device)

            data = self.model.pre_model_hook(data, is_train=True)

            predictions = self.model(data)
            loss_val, all_losses = self.loss_obj(predictions, data)

            if torch.isnan(loss_val):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(f"Loss was NaN for {nan_count} times in a row. Stopped training.")
                LOGGER.warning(f"Loss is Nan; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")
            else:
                nan_count = 0
                self.optimizer.zero_grad()
                loss_val.backward()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)

                self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {loss_val.item():.4f}")
            self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})

    def _set_random_seeds(self):
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.uniform(low=0, high=1e6))
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(':')[-1])
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
        LOGGER.info(f"### Device {self.device} will be used for training")

    def _create_folder_structure(self):
        if self.cfg.is_continue_training:
            folder_name = f"continue_training_from_epoch{self._epoch:03d}"
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.run_dir = self.cfg.run_dir / folder_name
        else:
            now = datetime.now()
            day = f"{now.day}".zfill(2)
            month = f"{now.month}".zfill(2)
            hour = f"{now.hour}".zfill(2)
            minute = f"{now.minute}".zfill(2)
            second = f"{now.second}".zfill(2)
            run_name = f'{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}'

            if self.cfg.run_dir is None:
                self.cfg.run_dir = Path().cwd() / "runs" / run_name
            else:
                self.cfg.run_dir = self.cfg.run_dir / run_name

        if not self.cfg.run_dir.is_dir():
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True)
        else:
            raise RuntimeError(f"There is already a folder at {self.cfg.run_dir}")

        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
            self.cfg.img_log_dir.mkdir(parents=True)
