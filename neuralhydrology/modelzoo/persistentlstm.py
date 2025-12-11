import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel


class PersistentLSTMModel(BaseModel):
    """
    Persistent-State LSTM for hydrological modeling.

    - Sequences (and optional non-overlapping windows) are created by BaseDataset
      using seq_stride / non_overlapping_sequences.
    - Hidden state can be carried across windows/batches via BaseTrainer when
      cfg.persistent_state = True and model = "persistentlstm".
    - Backward compatible: if you don't set persistent_state, it behaves like a
      normal sequence-to-sequence LSTM.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Use NH config fields
        input_size = len(cfg.dynamic_inputs_flattened)
        output_size = len(cfg.target_variables)

        hidden_size = cfg.hidden_size
        # Use 1 layer by default; avoid needing a Config property
        num_layers = getattr(cfg, "num_layers", 1)

        dropout = getattr(cfg, "dropout", 0.0)

        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output layer
        self.head = nn.Linear(hidden_size, output_size)

        # Whether trainer should keep state between batches
        self.persistent_state = getattr(cfg, "persistent_state", False)

    # ------------------------------------------------------------------
    # Helper: build [B, L, F] tensor from NH batch format
    # ------------------------------------------------------------------
    def _build_sequence(self, data):
        """
        data["x_d"] is a dict: feature_name -> [B, L, 1] tensors.
        We stack them in the order of cfg.dynamic_inputs_flattened.
        """
        x_d = data["x_d"]  # dict[str, Tensor[B, L, 1]]
        feature_names = self.cfg.dynamic_inputs_flattened

        x_list = [x_d[name] for name in feature_names]
        x = torch.cat(x_list, dim=-1)  # [B, L, F]
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, data, hidden_state=None):
        """
        data: dict from BaseDataset / BaseTrainer (after pre_model_hook)

        If hidden_state is provided:
            the LSTM continues from the previous state (persistent mode).
        If hidden_state is None:
            initialize fresh hidden states (normal mode).
        """

        x = self._build_sequence(data)          # [B, L, F]
        batch_size = x.size(0)
        device = x.device

        if hidden_state is None:
            h0 = torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device
            )
            c0 = torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device
            )
            hidden_state = (h0, c0)

        out, new_hidden = self.lstm(x, hidden_state)   # [B, L, H]
        preds = self.head(out)                         # [B, L, n_targets]

        # NH-style output dict
        return {
            "y_hat": preds,         # loss is computed on all time steps
            "hidden_state": new_hidden,
        }
