import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

import neuralhydrology.training.loss as loss
from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.tester import BaseTester
from neuralhydrology.modelzoo import get_model
from neuralhydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from neuralhydrology.training.earlystopper import EarlyStopper
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


# =============================================================================
# FIXED: BasinChronoInterleaveBatchSampler
# =============================================================================
class BasinChronoInterleaveBatchSampler(Sampler[List[int]]):
    """
    Correct sampler for *within-epoch* persistent hidden state.

    What we want (and what this sampler guarantees):
      - Within each basin: batches are always chronological (never A2 before A1).
      - Across basins: we still "shuffle" by randomly interleaving basins
        (A1, C1, B1, A2, D1, B2, ...).

    Why this is necessary:
      Persistent LSTM carries (h,c) forward in time. If you shuffle all batch-units
      globally, you can accidentally process a later time-chunk before an earlier one
      for the same basin -> hidden-state time misalignment -> training collapses / bad NSE.

    How it works:
      - We precompute, for each basin, the chronological list of dataset indices.
      - We keep a per-basin pointer `ptr[basin]` indicating which batch number comes next.
      - At each step we randomly choose a basin from the still-active basins and yield its NEXT batch.
    """

    def __init__(
        self,
        basin_to_sorted_indices: Dict[int, List[int]],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.basin_to_sorted_indices = basin_to_sorted_indices
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        # Precompute how many batches each basin contributes.
        self._basin_num_batches: Dict[int, int] = {}
        for basin_id, idxs in self.basin_to_sorted_indices.items():
            n = len(idxs)
            if self.drop_last:
                nb = n // self.batch_size
            else:
                nb = (n + self.batch_size - 1) // self.batch_size
            self._basin_num_batches[basin_id] = nb

        self._length = sum(self._basin_num_batches.values())

    def set_epoch(self, epoch: int):
        """Call this once per epoch so interleaving changes deterministically."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)

        # Pointer to "next chronological batch" per basin
        ptr: Dict[int, int] = {b: 0 for b in self._basin_num_batches.keys()}

        # Only basins with at least one batch
        active = [b for b, nb in self._basin_num_batches.items() if nb > 0]

        while active:
            basin_id = rng.choice(active)

            k = ptr[basin_id]  # batch number within this basin
            start = k * self.batch_size
            end = start + self.batch_size

            idxs = self.basin_to_sorted_indices[basin_id]
            batch = idxs[start:end]

            # Safety (should not happen for drop_last=True)
            if self.drop_last and len(batch) < self.batch_size:
                ptr[basin_id] += 1
                if ptr[basin_id] >= self._basin_num_batches[basin_id]:
                    active.remove(basin_id)
                continue

            yield batch

            ptr[basin_id] += 1
            if ptr[basin_id] >= self._basin_num_batches[basin_id]:
                active.remove(basin_id)


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

        # Persistent LSTM flag (your control switch)
        self.persistent_state = getattr(self.cfg, "persistent_state", False)

        # NOTE: You said you do NOT want persistence across epochs anymore.
        # These are kept only so older configs won't crash, but they are NOT used here.
        self.persist_state_across_epochs = getattr(self.cfg, "persist_state_across_epochs", False)
        self.shuffle_basins_each_epoch = getattr(self.cfg, "shuffle_basins_each_epoch", False)

        # Basin list
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # epoch start
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

        # Will be created in initialize_training
        self._basin_batch_sampler: Optional[BasinChronoInterleaveBatchSampler] = None

    # ------------------------------------------------------------------
    # Helpers for persistent LSTM reshaping and slicing
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_time_batch(t: torch.Tensor) -> torch.Tensor:
        """
        Flatten [B, L, ...] -> [1, B*L, ...].

        Why:
          Your persistentlstm forward expects a single "continuous" time axis for state carryover.
          By flattening across the DataLoader batch dimension, we treat the batch as one long segment.
        """
        if not torch.is_tensor(t) or t.dim() < 2:
            return t
        t = t.contiguous()
        b, l = t.shape[0], t.shape[1]
        return t.view(1, b * l, *t.shape[2:])

    def _slice_time_segment(self, data_dict: Dict[str, object], segment_slice: slice) -> Dict[str, object]:
        """Slice a contiguous segment along time axis from already reshaped tensors."""
        seg: Dict[str, object] = {}
        for key, val in data_dict.items():
            if key.startswith("x_d"):
                seg[key] = {feat: v[:, segment_slice, ...] for feat, v in val.items()}
            elif key.startswith("x_s"):
                seg[key] = val  # fixed to batch=1 later
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

    def _get_dataset(self, basin: Optional[str] = None) -> BaseDataset:
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

    # ------------------------------------------------------------------
    # Build basin->chronological indices from the dataset
    # ------------------------------------------------------------------
    def _build_basin_chrono_index(self, ds: BaseDataset) -> Dict[int, List[int]]:
        """
        Build mapping:
            basin_id (int) -> list of dataset indices sorted by start date.

        This is used by BasinChronoInterleaveBatchSampler so that:
            basin A always yields A1, A2, A3... (chronological)
        even while basins are interleaved in random order.
        """
        basin_to_items: Dict[int, List[Tuple[int, Any]]] = {}
        LOGGER.info("### Building basin chronological index (one pass over dataset) ...")

        for idx in range(len(ds)):
            sample = ds[idx]

            if "basin_idx" not in sample:
                raise RuntimeError("Dataset sample does not contain 'basin_idx'. Required for persistent batching.")
            bidx = sample["basin_idx"]
            if torch.is_tensor(bidx):
                basin_id = int(bidx.view(-1)[0].item())
            else:
                basin_id = int(np.array(bidx).reshape(-1)[0])

            # Prefer actual date if present
            if "date" in sample:
                d = sample["date"]
                d_np = np.array(d)
                start_time = d_np.reshape(-1)[0] if d_np.ndim >= 1 else d_np
            else:
                # Fallback: dataset order (still deterministic, but less robust)
                start_time = idx

            basin_to_items.setdefault(basin_id, []).append((idx, start_time))

        basin_to_sorted_indices: Dict[int, List[int]] = {}
        for basin_id, items in basin_to_items.items():
            items_sorted = sorted(items, key=lambda x: x[1])
            basin_to_sorted_indices[basin_id] = [i for i, _ in items_sorted]

        LOGGER.info(f"### Built chrono index for {len(basin_to_sorted_indices)} basins.")
        return basin_to_sorted_indices

    def initialize_training(self):
        """Initialize the training class."""
        if self.cfg.is_finetuning:
            self._scaler = load_scaler(self.cfg.base_run_dir)

        ds = self._get_dataset()
        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")

        # ------------------------------------------------------------------
        # CORRECT persistent batching
        # ------------------------------------------------------------------
        if self.persistent_state and self.cfg.model.lower() == "persistentlstm":
            # NOTE: we deliberately DO NOT use any "persist across epochs" file saving here.
            # We only ensure: persistent across batches within epoch + random interleaving of basins.
            basin_to_sorted_indices = self._build_basin_chrono_index(ds)

            self._basin_batch_sampler = BasinChronoInterleaveBatchSampler(
                basin_to_sorted_indices=basin_to_sorted_indices,
                batch_size=self.cfg.batch_size,
                drop_last=True,
                seed=self.cfg.seed if self.cfg.seed is not None else 0,
            )

            # DataLoader uses our sampler; don't enable shuffle here.
            self.loader = DataLoader(
                ds,
                batch_sampler=self._basin_batch_sampler,
                num_workers=self.cfg.num_workers,
                collate_fn=ds.collate_fn,
            )
            LOGGER.info("### Using BasinChronoInterleaveBatchSampler (interleaved basins, chrono within basin).")
        else:
            # Original non-persistent training
            self.loader = DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                collate_fn=ds.collate_fn,
            )

        self.model = self._get_model().to(self.device)

        if self.cfg.checkpoint_path is not None:
            LOGGER.info(f"Starting training from Checkpoint {self.cfg.checkpoint_path}")
            self.model.load_state_dict(torch.load(str(self.cfg.checkpoint_path), map_location=self.device))
        elif self.cfg.checkpoint_path is None and self.cfg.is_finetuning:
            checkpoint_path = [x for x in sorted(list(self.cfg.base_run_dir.glob("model_epoch*.pt")))][-1]
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
                    "'validate_n_random_basins' not set or set to zero. Will validate on the entire validation set.",
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
                self.optimizer,
                mode="min",
                factor=self._factor_dynamic_learning_rate,
                patience=self._patience_dynamic_learning_rate,
            )

        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            # IMPORTANT: changes interleaving pattern each epoch (still chrono within basin)
            if self._basin_batch_sampler is not None:
                self._basin_batch_sampler.set_epoch(epoch)

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
                    experiment_logger=self.experiment_logger.valid(),
                )

                valid_metrics = self.experiment_logger.summarise()
                print_msg = f"Epoch {epoch} average validation loss: {valid_metrics['avg_total_loss']:.5f}"
                if self.cfg.metrics:
                    print_msg += " -- Median validation metrics: "
                    print_msg += ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != "avg_total_loss")
                LOGGER.info(print_msg)

                if (
                    self._early_stopping
                    and epoch > self._minimum_epochs_before_early_stopping
                    and early_stopper.check_early_stopping(valid_metrics["avg_total_loss"])
                ):
                    LOGGER.info(
                        f"Early stopping triggered at epoch {epoch} with validation loss "
                        f"{valid_metrics['avg_total_loss']:.5f}. Training stopped."
                    )
                    break

                if self._dynamic_learning_rate:
                    scheduler.step(valid_metrics["avg_total_loss"])

        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _get_start_epoch_number(self):
        if self.cfg.is_continue_training:
            if self.cfg.continue_from_epoch is not None:
                epoch = self.cfg.continue_from_epoch
            else:
                weight_path = [x for x in sorted(list(self.cfg.run_dir.glob("model_epoch*.pt")))][-1]
                epoch = weight_path.name[-6:-3]
        else:
            epoch = 0
        return int(epoch)

    def _restore_training_state(self):
        if self.cfg.continue_from_epoch is not None:
            epoch = f"{self.cfg.continue_from_epoch:03d}"
            weight_path = self.cfg.base_run_dir / f"model_epoch{epoch}.pt"
        else:
            weight_path = [x for x in sorted(list(self.cfg.base_run_dir.glob("model_epoch*.pt")))][-1]
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

        # ============================================================
        # Persistent path: persist across batches within epoch only
        # ============================================================
        if self.persistent_state and self.cfg.model.lower() == "persistentlstm":
            # Reset all basin states at epoch start (NO cross-epoch persistence)
            epoch_state_cache: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}

            n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
            pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
            pbar.set_description(f"# Epoch {epoch} (interleaved basins, chrono within basin)")

            nan_count = 0

            for i, data in enumerate(pbar):
                if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                    break

                # Move tensors to device
                for key in data.keys():
                    if key.startswith("x_d"):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith("date"):
                        data[key] = data[key].to(self.device)

                data = self.model.pre_model_hook(data, is_train=True)

                # Determine basin for this batch
                if "basin_idx" not in data or (not torch.is_tensor(data["basin_idx"])):
                    raise RuntimeError("Persistent training requires basin_idx in batch.")

                bidx = data["basin_idx"]

                # HARD SAFETY CHECK: ensure the batch contains only ONE basin
                flat = bidx.view(-1)
                if not torch.all(flat == flat[0]):
                    raise RuntimeError("Batch contains multiple basins -> hidden-state leakage risk. Fix sampler/dataset.")

                # Extract basin id
                basin_id = int(flat[0].item())

                # Load basin hidden state (None if first time this epoch)
                hidden = epoch_state_cache.get(basin_id, None)

                # Infer B,L from any dynamic input
                any_xd = None
                if "x_d" in data and isinstance(data["x_d"], dict) and len(data["x_d"]) > 0:
                    any_xd = next(iter(data["x_d"].values()))
                if any_xd is None or any_xd.dim() < 2:
                    raise RuntimeError("Persistent training: could not infer [B,L] from x_d.")
                B_inferred, L_inferred = int(any_xd.shape[0]), int(any_xd.shape[1])

                # Flatten to batch=1 so the model sees one continuous segment
                reshaped: Dict[str, object] = {}
                for key, val in data.items():
                    if key.startswith("x_d"):
                        reshaped[key] = {feat: self._flatten_time_batch(v) for feat, v in val.items()}
                    elif key.startswith("x_s"):
                        reshaped[key] = val
                    elif key.startswith("y") and torch.is_tensor(val):
                        reshaped[key] = self._flatten_time_batch(val)
                    elif key == "basin_idx" and torch.is_tensor(val):
                        # Ensure basin_idx becomes [B,L] before flatten so it aligns with time
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

                # Fix x_s to batch=1 (static features should be per-basin, not per-time)
                if "x_s" in data:
                    if isinstance(data["x_s"], dict):
                        seg_data["x_s"] = {feat: v[0:1, ...] for feat, v in data["x_s"].items()}
                    else:
                        seg_data["x_s"] = data["x_s"][0:1, ...]

                # Forward with carried hidden state
                preds = self.model(seg_data, hidden_state=hidden)

                # Update basin hidden state (ONLY within this epoch)
                new_hidden = preds.get("hidden_state", None)
                if new_hidden is not None:
                    h, c = new_hidden
                    epoch_state_cache[basin_id] = (h.detach(), c.detach())
                else:
                    epoch_state_cache[basin_id] = None

                # Loss
                loss_val, all_losses = self.loss_obj(preds, seg_data)

                if torch.isnan(loss_val):
                    nan_count += 1
                    if nan_count > self._allow_subsequent_nan_losses:
                        raise RuntimeError(f"Loss was NaN for {nan_count} times in a row. Stopped training.")
                    LOGGER.warning(f"Loss is NaN; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")
                    continue
                nan_count = 0

                self.optimizer.zero_grad()
                loss_val.backward()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)

                self.optimizer.step()

                pbar.set_postfix_str(f"Loss: {loss_val.item():.4f}")
                self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})

            return

        # ============================================================
        # Original non-persistent path (unchanged)
        # ============================================================
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f"# Epoch {epoch}")

        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            for key in data.keys():
                if key.startswith("x_d"):
                    data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                elif not key.startswith("date"):
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
            run_name = f"{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}"

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
