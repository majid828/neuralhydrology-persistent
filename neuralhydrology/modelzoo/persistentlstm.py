from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class PersistentLSTMModel(BaseModel):
    """
    Persistent-state single-frequency LSTM, NH-style.

    - Uses the same InputLayer + head design as CudaLSTM.
    - Supports carrying hidden state across segments/batches via
      the `hidden_state` argument and the `"hidden_state"` output key.
    - Backward compatible: if you don't use `persistent_state` in the config
      and don't pass `hidden_state`, it behaves like a normal CudaLSTM-style
      sequence model.

    Assumptions for your current use:
    - Single frequency (len(cfg.use_frequencies) == 1).
    - No static attributes are used (so InputLayer only builds dynamic inputs).
    """

    # Parts that can be finetuned (same pattern as CudaLSTM)
    module_parts = ["embedding_net", "lstm", "head"]

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

        # Embedding layer for dynamic (and optionally static) inputs
        self.embedding_net = InputLayer(cfg)

        # Core LSTM (seq_first, batch_second as in CudaLSTM)
        self.lstm = nn.LSTM(
            input_size=self.embedding_net.output_size,
            hidden_size=cfg.hidden_size,
        )

        # Dropout + head, same as in CudaLSTM
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        # Optional forget-gate bias initialisation (NH style)
        self._reset_parameters()

        # Whether trainer is *supposed* to keep state between segments/batches
        # (the actual logic lives in BaseTrainer / tester.py)
        self.persistent_state = getattr(cfg, "persistent_state", False)

    def _reset_parameters(self):
        """Special initialization of certain model weights (same as CudaLSTM)."""
        if self.cfg.initial_forget_bias is not None:
            # Set forget gate bias of first layer to a positive value
            self.lstm.bias_hh_l0.data[
                self.cfg.hidden_size : 2 * self.cfg.hidden_size
            ] = self.cfg.initial_forget_bias

    def forward(
        self,
        data: Dict[str, torch.Tensor | Dict[str, torch.Tensor]],
        hidden_state=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional persistent hidden state.

        Parameters
        ----------
        data : dict
            NH batch dictionary (after BaseModel.pre_model_hook).
        hidden_state : tuple(h, c) or None
            If provided, it is passed as `hx` to the LSTM so that the sequence
            continues from a previous state (persistent mode).
            Shape must be:
                h: [num_layers, batch, hidden_size]
                c: [num_layers, batch, hidden_size]

        Returns
        -------
        dict
            NH-style output dict including:
              - 'y_hat': [batch, seq, n_targets]
              - 'lstm_output': [batch, seq, hidden_size]
              - 'h_n': [batch, 1, hidden_size] (last hidden per layer)
              - 'c_n': [batch, 1, hidden_size] (last cell per layer)
              - 'hidden_state': (h, c) in LSTM-native shape [layers, batch, hidden]
        """
        # Build embedded dynamic inputs (and static if present) in NH format:
        # x_d: [seq_len, batch_size, input_dim]
        x_d = self.embedding_net(data)

        # LSTM forward
        if hidden_state is None:
            lstm_output, (h_n, c_n) = self.lstm(x_d)
        else:
            lstm_output, (h_n, c_n) = self.lstm(x_d, hx=hidden_state)

        # Convert to [batch, seq, hidden] for the head, like CudaLSTM does
        lstm_output_b = lstm_output.transpose(0, 1)  # [B, L, H]
        h_n_b = h_n.transpose(0, 1)                  # [B, num_layers, H]
        c_n_b = c_n.transpose(0, 1)                  # [B, num_layers, H]

        # Core prediction dict
        pred = {
            "lstm_output": lstm_output_b,
            "h_n": h_n_b,
            "c_n": c_n_b,
            # This is what BaseTrainer/tester use to carry state:
            # keep LSTM-native layout (layers, batch, hidden) so they can reuse it.
            "hidden_state": (h_n, c_n),
        }

        # Add head outputs (y_hat etc.)
        pred.update(self.head(self.dropout(lstm_output_b)))

        return pred
