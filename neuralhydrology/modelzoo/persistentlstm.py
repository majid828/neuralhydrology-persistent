from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class PersistentLSTMModel(BaseModel):
    """
    Persistent LSTM model (CudaLSTM-style) with optional hidden-state carry-over.

    - Uses the same embedding pipeline and head structure as CudaLSTM:
        * InputLayer -> LSTM -> Dropout -> Head
    - Adds an optional `hidden_state` argument to `forward` so that the trainer
      can pass in (h, c) from the previous segment / batch.
    - Returns an extra `"hidden_state"` key in the prediction dict:
        * "hidden_state": (h_n, c_n) with shape [num_layers, batch, hidden_size]
      which is directly reusable as `hx` for the next call.

    Backward-compatibility:
    - If you call it like a normal NH model: `model(data)`, with no `hidden_state`,
      PyTorch will initialize zeros internally and it behaves like a standard
      single-timescale CudaLSTM.
    - Existing loss / head logic still expects `"y_hat"` etc., which are provided
      via `get_head`.
    """

    # for finetuning etc., same parts as CudaLSTM
    module_parts = ["embedding_net", "lstm", "head"]

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

        self.cfg = cfg

        # Embedding network for dynamic/static inputs
        self.embedding_net = InputLayer(cfg)

        # LSTM with the same interface as CudaLSTM
        # (input: [seq_len, batch, in_features])
        self.lstm = nn.LSTM(
            input_size=self.embedding_net.output_size,
            hidden_size=cfg.hidden_size,
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        # Head that maps hidden states to targets, same as CudaLSTM
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights (forget gate bias)."""
        init_forget = getattr(self.cfg, "initial_forget_bias", None)
        if init_forget is not None:
            # bias_hh_l0 has size 4 * hidden_size (i, f, g, o gates)
            # forget gate bias is the 2nd chunk [H:2H]
            H = self.cfg.hidden_size
            self.lstm.bias_hh_l0.data[H:2 * H] = init_forget

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
            Batch as produced by the NH dataset + pre_model_hook.
        hidden_state : tuple(h, c) or None
            If None: behaves like normal LSTM (PyTorch initializes state).
            If not None: must be a tuple of Tensors (h, c) with shapes
                [num_layers, batch, hidden_size].

        Returns
        -------
        Dict[str, torch.Tensor]
            - "y_hat": predictions [batch, seq, n_targets] (from head)
            - "lstm_output": [batch, seq, hidden_size]
            - "h_n": final hidden state [batch, 1, hidden_size]
            - "c_n": final cell state [batch, 1, hidden_size]
            - "hidden_state": (h_n_raw, c_n_raw) with shapes
                [num_layers, batch, hidden_size] for reuse by trainer.
        """
        # Embedding: produces [seq_len, batch, in_features]
        x_d = self.embedding_net(data)

        # Run LSTM with or without provided state
        if hidden_state is None:
            lstm_output, (h_n, c_n) = self.lstm(input=x_d)
        else:
            lstm_output, (h_n, c_n) = self.lstm(input=x_d, hx=hidden_state)

        # Convert LSTM output to [batch, seq, hidden] like CudaLSTM
        lstm_output = lstm_output.transpose(0, 1)
        h_n_b = h_n.transpose(0, 1)  # [batch, num_layers, hidden] but kept for compatibility
        c_n_b = c_n.transpose(0, 1)

        pred: Dict[str, torch.Tensor] = {
            "lstm_output": lstm_output,
            "h_n": h_n_b,
            "c_n": c_n_b,
            # raw shapes as PyTorch returns, for persistence:
            "hidden_state": (h_n, c_n),
        }

        # Add "y_hat" etc. from the head
        pred.update(self.head(self.dropout(lstm_output)))

        return pred
