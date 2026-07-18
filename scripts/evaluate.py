import torch
import numpy as np
from src.models.fusion import CrossModalFusion
from src.data.processors import DataProcessor
from src.utils.config_manager import load_config

def evaluate():
    config = load_config()
    processor = DataProcessor()
    
    # Initialize the model
    model = CrossModalFusion(
        ts_dim=config['model']['ts_dim'],
        context_dim=config['model']['context_dim'],
        embed_dim=config['model']['embed_dim']
    )
    model.eval() # Set to evaluation mode
    
    # Generate dummy test data
    ts_data = torch.randn(10, 50, 1)    # 10 samples
    ctx_data = torch.randn(10, 10, 128) # 10 context samples
    targets = torch.randn(10, 50, 64)   # Expected output
    
    with torch.no_grad():
        # --- Experiment 1: Full Multimodal ---
        full_pred = model(ts_data, ctx_data)
        full_mse = torch.nn.functional.mse_loss(full_pred, targets).item()
        
        # --- Experiment 2: Unimodal Ablation (Zeroing out context) ---
        # By setting context to zeros, we see how much the model relies on it
        zero_ctx = torch.zeros_like(ctx_data)
        uni_pred = model(ts_data, zero_ctx)
        uni_mse = torch.nn.functional.mse_loss(uni_pred, targets).item()
        
    # Report findings
    print("-" * 30)
    print("MODEL EVALUATION: ABLATION STUDY")
    print("-" * 30)
    print(f"Full Multimodal MSE : {full_mse:.6f}")
    print(f"Unimodal (TS only)  MSE : {uni_mse:.6f}")
    print(f"Performance Gain    : {((uni_mse - full_mse) / uni_mse) * 100:.2f}%")
    print("-" * 30)
    
    if full_mse < uni_mse:
        print("RESULT: Multimodal fusion is successfully providing predictive signal.")
    else:
        print("RESULT: Contextual data is not currently improving accuracy.")

if __name__ == "__main__":
    evaluate()