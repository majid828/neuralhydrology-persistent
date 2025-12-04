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

        # Persistent LSTM
        self.persistent_state = getattr(self.cfg, "persistent_state", False)
        self._persistent_hidden = None
        self._current_basin_idx = None

        # load basins
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # start epoch
        self._epoch = self._get_start_epoch_number()

        self._create_folder_structure()
        setup_logging(str(self.cfg.run_dir / "output.log"))
        LOGGER.info(f"### Folder structure created at {self.cfg.run_dir}")

        for key, val in self.cfg.as_dict().items():
            LOGGER.info(f"{key}: {val}")

        self._set_random_seeds()
        self._set_device()

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
        for param in self.model.parameters():
            param.requires_grad = False

        unresolved_modules = []

        if isinstance(self.cfg.finetune_modules, list):
            for module_part in self.cfg.finetune_modules:
                if module_part in self.model.module_parts:
                    module = getattr(self.model, module_part)
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    unresolved_modules.append(module_part)
        else:
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
            LOGGER.warning(f"Could not resolve finetuning modules: {unresolved_modules}")

    def initialize_training(self):
        if self.cfg.is_finetuning:
            self._scaler = load_scaler(self.cfg.base_run_dir)

        ds = self._get_dataset()
        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")
        self.loader = self._get_data_loader(ds=ds)

        self.model = self._get_model().to(self.device)

        if self.cfg.checkpoint_path is not None:
            self.model.load_state_dict(torch.load(str(self.cfg.checkpoint_path), map_location=self.device))

        if self.cfg.is_finetuning and self.cfg.checkpoint_path is None:
            checkpoint_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))

        if self.cfg.is_finetuning:
            self._freeze_model_parts()

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
            self.validator = self._get_tester()

        if self.cfg.target_noise_std is not None:
            self.noise_sampler_y = torch.distributions.Normal(loc=0, scale=self.cfg.target_noise_std)
            self._target_mean = torch.from_numpy(
                ds.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values).to(self.device)
            self._target_std = torch.from_numpy(
                ds.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values).to(self.device)

    def train_and_validate(self):
        if self._early_stopping:
            early_stopper = EarlyStopper(
                patience=self._patience_early_stopping, min_delta=0.0001
            )

        if self._dynamic_learning_rate:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=self._factor_dynamic_learning_rate,
                patience=self._patience_dynamic_learning_rate
            )

        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):

            if not self._dynamic_learning_rate:
                if epoch in self.cfg.learning_rate.keys():
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.cfg.learning_rate[epoch]

            self._train_epoch(epoch=epoch)
            avg_losses = self.experiment_logger.summarise()
            LOGGER.info(f"Epoch {epoch} avg loss: {avg_losses}")

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

                if self._early_stopping and epoch > self._minimum_epochs_before_early_stopping:
                    if early_stopper.check_early_stopping(valid_metrics['avg_total_loss']):
                        LOGGER.info(f"Early stopping at epoch {epoch}")
                        break

                if self._dynamic_learning_rate:
                    scheduler.step(valid_metrics['avg_total_loss'])

        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.experiment_logger.train()

        # Reset persistent LSTM state at epoch start
        if self.persistent_state and self.cfg.model.lower() == "persistentlstm":
            self._persistent_hidden = None
            self._current_basin_idx = None

        # progress bar
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f'# Epoch {epoch}')

        nan_count = 0

        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            # move dynamic/static inputs to device
            for key in data.keys():
                if key.startswith("x_d"):
                    data[key] = {feat: v.to(self.device) for feat, v in data[key].items()}
                elif not key.startswith("date"):
                    data[key] = data[key].to(self.device)

            data = self.model.pre_model_hook(data, is_train=True)

            # ----------------------------------------------------------------------
            # PERSISTENT LSTM LOGIC (simple + correct)
            # ----------------------------------------------------------------------
            if self.persistent_state and self.cfg.model.lower() == "persistentlstm":

                batch_size = data["basin_idx"].shape[0]
                total_loss = 0.0
                all_losses_accum = None

                for b in range(batch_size):

                    # Create single-sample batch
                    sample_b = {}
                    for key, val in data.items():
                        if key.startswith("date"):
                            sample_b[key] = val[b:b + 1]
                        elif key.startswith("x_d"):
                            sample_b[key] = {feat: ten[b:b + 1] for feat, ten in val.items()}
                        else:
                            sample_b[key] = val[b:b + 1]

                    basin_idx = int(sample_b["basin_idx"].item())

                    # reset hidden state if basin changes
                    if self._current_basin_idx is None or basin_idx != self._current_basin_idx:
                        self._persistent_hidden = None
                        self._current_basin_idx = basin_idx

                    preds_b = self.model(sample_b, hidden_state=self._persistent_hidden)

                    new_hidden = preds_b.get("hidden_state", None)
                    if new_hidden is not None:
                        h, c = new_hidden
                        self._persistent_hidden = (h.detach(), c.detach())

                    loss_b, all_losses_b = self.loss_obj(preds_b, sample_b)

                    total_loss += loss_b
                    if all_losses_accum is None:
                        all_losses_accum = {k: v.clone() for k, v in all_losses_b.items()}
                    else:
                        for k in all_losses_b:
                            all_losses_accum[k] += all_losses_b[k]

                loss = total_loss / batch_size
                all_losses = {k: v / batch_size for k, v in all_losses_accum.items()}

            # ----------------------------------------------------------------------
            # NORMAL MODE (non-persistent models)
            # ----------------------------------------------------------------------
            else:
                predictions = self.model(data)
                loss, all_losses = self.loss_obj(predictions, data)

            # nan handling
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError("Too many NaN losses.")
                LOGGER.warning("NaN loss encountered; ignoring this step.")
            else:
                nan_count = 0

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)
                self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
            self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})

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

        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(str(optimizer_path), map_location=self.device))

    def _save_weights_and_optimizer(self, epoch: int):
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), str(weight_path))

        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

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
                    raise RuntimeError(f"GPU {gpu_id} not available")
                self.device = torch.device(self.cfg.device)
            elif self.cfg.device == "mps":
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    raise RuntimeError("MPS not available")
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        LOGGER.info(f"### Using device: {self.device}")

    def _create_folder_structure(self):
        if self.cfg.is_continue_training:
            folder_name = f"continue_training_from_epoch{self._epoch:03d}"
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.run_dir = self.cfg.run_dir / folder_name
        else:
            now = datetime.now()
            day = f"{now.day:02d}"
            month = f"{now.month:02d}"
            hour = f"{now.hour:02d}"
            minute = f"{now.minute:02d}"
            second = f"{now.second:02d}"
            run_name = f'{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}'

            if self.cfg.run_dir is None:
                self.cfg.run_dir = Path().cwd() / "runs" / run_name
            else:
                self.cfg.run_dir = self.cfg.run_dir / run_name

        if not self.cfg.run_dir.is_dir():
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True)
        else:
            raise RuntimeError(f"Folder already exists: {self.cfg.run_dir}")

        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
            self.cfg.img_log_dir.mkdir(parents=True)
