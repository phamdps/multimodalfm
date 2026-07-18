import torch
import numpy as np
from typing import Dict, Tuple

class DataProcessor:
    """
    Handles preprocessing and transformation of multimodal data.
    """
    def __init__(self, target_len: int = 50):
        self.target_len = target_len

    def prepare_time_series(self, data: np.ndarray) -> torch.Tensor:
        """
        Ensures time-series data is the correct length and format.
        """
        # Convert to tensor and add channel dim if missing
        tensor = torch.tensor(data, dtype=torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)  # [Batch, Seq, 1]
        return tensor

    def prepare_context(self, context_data: np.ndarray) -> torch.Tensor:
        """
        Converts auxiliary metadata to tensors.
        """
        return torch.tensor(context_data, dtype=torch.float32)

    def create_batch(self, ts_data: np.ndarray, ctx_data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper to prepare a full batch for the model.
        """
        return self.prepare_time_series(ts_data), self.prepare_context(ctx_data)