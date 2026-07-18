import pandas as pd
import torch
import numpy as np

class DataProcessor:
    def __init__(self, target_col='OT'):
        self.target_col = target_col

    def load_etth1(self, path):
        """Loads and normalizes ETTh1 data."""
        df = pd.read_csv(path)
        # Assuming 'date' is the timestamp column
        df['date'] = pd.to_datetime(df['date'])
        data = df.drop(columns=['date']).values
        
        # Simple Z-score normalization
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        data = (data - mean) / std
        return torch.tensor(data, dtype=torch.float32)

    def prepare_batch(self, data, context, seq_len=48):
        """Creates sliding window batches."""
        # This is a simplified batcher for demonstration
        ts_input = data[:seq_len, :] # Numerical history
        ctx_input = context[:10, :]  # Simulated contextual info
        return ts_input.unsqueeze(0), ctx_input.unsqueeze(0)