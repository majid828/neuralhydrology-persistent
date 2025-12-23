import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

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
    """Default class to train a model.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

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
        # Persistent LSTM support (optional, backward-safe)
        # --------------------------------------------------
        # - self.persistent_state: flag from config to enable persistent hidden state
        # - self._persistent_hidden: tuple (h, c) carried across batches
        # - self._current_basin_idx: remembers which basin we are in across batches
        self.persistent_state = getattr(self.cfg, "persistent_state", False)
        self._persistent_hidden = None
        self._current_basin_idx = None

        # load train basin list and add number of basins to the config
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # check at which epoch the training starts
        self._epoch = self._get_start_epoch_number()

        self._create_folder_structure()
        setup_logging(str(self.cfg.run_dir / "output.log"))
        LOGGER.info(f"### Folder structure created at {self.cfg.run_dir}")

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
        """Flatten [B, L, ...] → [1, B*L, ...] along the first two dims.

        IMPORTANT: enforce contiguity to make the flatten mapping deterministic.
        """
        if not torch.is_tensor(t) or t.dim() < 2:
            return t
        t = t.contiguous()
        b, l = t.shape[0], t.shape[1]
        remaining = t.shape[2:]
        new_shape = (1, b * l) + remaining
        return t.view(*new_shape)

    def _slice_time_segment(self, data_dict: Dict[str, object], segment_slice: slice) -> Dict[str, object]:
        """Slice a contiguous time segment from reshaped data.

        Expects time-like tensors to have shape [1, L_new, ...] after `_flatten_time_batch`.
        Static (x_s) and non-tensor entries are kept as-is (but we will overwrite x_s per segment in train loop).
        """
        seg = {}
        for key, val in data_dict.items():
            if key.startswith("x_d"):
                seg[key] = {feat: v[:, segment_slice, ...] for feat, v in val.items()}
            elif key.startswith("x_s"):
                seg[key] = val  # overwritten per segment later
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
        """Find which original batch row corresponds to `basin_id` (robust).
        Prefers data['basin_idx'] if it is [B]. Falls back to mapping via [B,L] if present.
        """
        if "basin_idx" not in data or (not torch.is_tensor(data["basin_idx"])):
            raise RuntimeError("PersistentLSTM: basin_idx missing from batch; cannot map static features.")

        bidx = data["basin_idx"]

        # Common NH case: basin_idx is [B]
        if bidx.dim() == 1 and int(bidx.shape[0]) == int(B_expected):
            matches = (bidx == basin_id).nonzero(as_tuple=True)[0]
            if matches.numel() == 0:
                raise RuntimeError(f"PersistentLSTM: basin {basin_id} not found in batch basin_idx.")
            return int(matches[0].item())

        # Sometimes basin_idx might already be [B,L] (rare but handle)
        if bidx.dim() == 2 and int(bidx.shape[0]) == int(B_expected):
            # Each row should be constant basin id; take first column
            row_ids = bidx[:, 0]
            matches = (row_ids == basin_id).nonzero(as_tuple=True)[0]
            if matches.numel() == 0:
                raise RuntimeError(f"PersistentLSTM: basin {basin_id} not found in batch basin_idx[:,0].")
            return int(matches[0].item())

        raise RuntimeError(
            f"PersistentLSTM: unexpected basin_idx shape {tuple(bidx.shape)}; expected [B] or [B,L]."
        )

    def _get_dataset(self) -> BaseDataset:
        return get_dataset(cfg=self.cfg, period="train", is_train=True, scaler=self._scaler)

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

    def _get_data_loader(self, ds: BaseDataset) -> torch.utils.data.DataLoader:
        # For persistent LSTM we want strict temporal order (no shuffling)
        shuffle = True
        if getattr(self.cfg, "persistent_state", False) and self.cfg.model.lower() == "persistentlstm":
            shuffle = False

        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=ds.collate_fn,
        )

    def _freeze_model_parts(self):
        # freeze all model weights
        for param in self.model.parameters():
            param.requires_grad = False

        unresolved_modules = []

        # unfreeze parameters specified in config as tuneable parameters
        if isinstance(self.cfg.finetune_modules, list):
            for module_part in self.cfg.finetune_modules:
                if module_part in self.model.module_parts:
                    module = getattr(self.model, module_part)
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    unresolved_modules.append(module_part)
        else:
            # if it was no list, it has to be a dictionary
            for module_group, module_parts in self.cfg.finetune_modules.items():
                if module_group in self.model.module_parts:
                    if isinstance(module_parts, str):
                        module_parts = [module_parts]
                    for module_part in module_parts:
                        module = getattr(self.model, module_group)[module_part]
                        for param in module.parameters():
                            param.requires_grad = True
                else:
                    unresolved_modules.append(module_group)
        if unresolved_modules:
            LOGGER.warning(f"Could not resolve the following module parts for finetuning: {unresolved_modules}")

    def initialize_training(self):
        """Initialize the training class."""
        if self.cfg.is_finetuning:
            # Load scaler from pre-trained model.
            self._scaler = load_scaler(self.cfg.base_run_dir)

        # Initialize dataset before the model is loaded.
        ds = self._get_dataset()
        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")
        self.loader = self._get_data_loader(ds=ds)

        self.model = self._get_model().to(self.device)
        if self.cfg.checkpoint_path is not None:
            LOGGER.info(f"Starting training from Checkpoint {self.cfg.checkpoint_path}")
            self.model.load_state_dict(torch.load(str(self.cfg.checkpoint_path), map_location=self.device))
        elif self.cfg.checkpoint_path is None and self.cfg.is_finetuning:
            checkpoint_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            LOGGER.info(f"Starting training from checkpoint {checkpoint_path}")
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))

        # Freeze model parts from pre-trained model.
        if self.cfg.is_finetuning:
            self._freeze_model_parts()

        self.optimizer = self._get_optimizer()
        self.loss_obj = self._get_loss_obj().to(self.device)

        # Add possible regularization terms to the loss function.
        self._set_regularization()

        # restore optimizer and model state if training is continued
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

        # Reset persistent LSTM state at the start of each epoch
        if self.persistent_state and self.cfg.model.lower() == "persistentlstm":
            self._persistent_hidden = None
            self._current_basin_idx = None

        n_iter = min(self._max_updates_per_epoch, len(self.loader)) \
            if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f'# Epoch {epoch}')

        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            # move batch to device
            for key in data.keys():
                if key.startswith('x_d'):
                    data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                elif not key.startswith('date'):
                    data[key] = data[key].to(self.device)

            # pre-processing hook
            data = self.model.pre_model_hook(data, is_train=True)

            # --------------------------------------------------
            # PERSISTENT LSTM branch
            # --------------------------------------------------
            if self.persistent_state and self.cfg.model.lower() == "persistentlstm":

                # infer B and L once
                any_feat = None
                for dk, dv in data.items():
                    if dk.startswith("x_d") and isinstance(dv, dict) and len(dv) > 0:
                        any_feat = next(iter(dv.values()))
                        break
                if any_feat is None or (not torch.is_tensor(any_feat)) or any_feat.dim() < 2:
                    raise RuntimeError("PersistentLSTM: could not infer B,L because no valid x_d tensor found.")
                B_inferred = int(any_feat.shape[0])
                L_inferred = int(any_feat.shape[1])

                # ----- Step 1: Build reshaped_data with flattened time dimension -----
                reshaped_data: Dict[str, object] = {}

                for key, val in data.items():
                    if key.startswith("x_d"):
                        reshaped_data[key] = {feat: self._flatten_time_batch(v) for feat, v in val.items()}

                    elif key.startswith("x_s"):
                        # keep as-is; will slice per segment to [1, ...]
                        reshaped_data[key] = val

                    elif key.startswith("y"):
                        reshaped_data[key] = self._flatten_time_batch(val)

                    elif key.startswith("basin_idx"):
                        # basin_idx should become [1, B*L]
                        if torch.is_tensor(val) and val.dim() == 1:
                            basin_2d = val.view(B_inferred, 1).expand(B_inferred, L_inferred)
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

                # ----- Step 2: Split the flattened sequence into basin segments -----
                basin_indices_flat = reshaped_data["basin_idx"].view(-1)
                L_new = basin_indices_flat.size(0)

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

                # ----- Step 3: Loop over segments, manage hidden state, accumulate loss -----
                total_loss = None
                total_len = 0
                accumulated_losses = None

                hidden = self._persistent_hidden
                prev_basin_id = None

                for i_seg, segment_slice in enumerate(segment_slices):
                    seg_data = self._slice_time_segment(reshaped_data, segment_slice)

                    # Basin id for this segment (constant)
                    seg_basin_ids = seg_data["basin_idx"].view(-1)
                    basin_id_this_seg = int(seg_basin_ids[0].item())

                    # ---------------------------
                    # STATIC ATTRIBUTES:
                    # slice x_s to [1, ...] based on basin_id
                    # ---------------------------
                    if any(k.startswith("x_s") for k in data.keys()):
                        try:
                            orig_b = self._resolve_batch_row_for_basin(data, basin_id_this_seg, B_inferred)
                        except Exception:
                            # fallback mapping (only if basin_idx isn't [B] or unexpected)
                            seg_start = int(segment_slice.start)
                            orig_b = seg_start // L_inferred

                        for sk, sv in data.items():
                            if sk.startswith("x_s"):
                                if isinstance(sv, dict):
                                    seg_data[sk] = {feat: v[orig_b:orig_b + 1, ...] for feat, v in sv.items()}
                                else:
                                    seg_data[sk] = sv[orig_b:orig_b + 1, ...]

                    # Reset logic
                    if i_seg == 0:
                        if self._current_basin_idx is None or basin_id_this_seg != self._current_basin_idx:
                            hidden = None
                    else:
                        if prev_basin_id is not None and basin_id_this_seg != prev_basin_id:
                            hidden = None

                    # Forward pass
                    preds = self.model(seg_data, hidden_state=hidden)

                    # Update hidden (detach)
                    new_hidden = preds.get("hidden_state", None)
                    if new_hidden is not None:
                        h, c = new_hidden
                        hidden = (h.detach(), c.detach())
                    else:
                        hidden = None

                    # Segment loss
                    seg_loss, seg_all_losses = self.loss_obj(preds, seg_data)
                    seg_len = seg_basin_ids.numel()

                    # Weighted accumulation
                    if total_loss is None:
                        total_loss = seg_loss * seg_len
                    else:
                        total_loss = total_loss + seg_loss * seg_len
                    total_len += seg_len

                    if accumulated_losses is None:
                        accumulated_losses = {k: v * seg_len for k, v in seg_all_losses.items()}
                    else:
                        for k, v in seg_all_losses.items():
                            accumulated_losses[k] = accumulated_losses.get(k, 0.0) + v * seg_len

                    prev_basin_id = basin_id_this_seg

                # Final loss averaged over all time steps in this batch
                loss = total_loss / total_len
                all_losses = {k: v / total_len for k, v in accumulated_losses.items()}

                # ----- Step 4: Update persistent hidden state and current basin -----
                self._persistent_hidden = hidden
                self._current_basin_idx = prev_basin_id

            # --------------------------------------------------
            # NORMAL BEHAVIOR (all other models)
            # --------------------------------------------------
            else:
                predictions = self.model(data)
                loss, all_losses = self.loss_obj(predictions, data)

            # early stop training if loss is NaN
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(f"Loss was NaN for {nan_count} times in a row. Stopped training.")
                LOGGER.warning(f"Loss is Nan; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")
            else:
                nan_count = 0

                self.optimizer.zero_grad()
                loss.backward()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)

                self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
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
