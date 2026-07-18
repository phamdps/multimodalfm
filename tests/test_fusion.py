import torch
import torch.nn as nn
# Import the module we created in the previous step
from src.models.fusion import AdvancedGatedFusion

def run_ablation_study(model, ts_data, ctx_data, targets):
    """
    Measures the performance delta when auxiliary data is removed.
    """
    model.eval()
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        # 1. Full Multimodal Performance
        full_pred = model(ts_data, ctx_data)
        full_mse = criterion(full_pred, targets).item()
        
        # 2. Unimodal Performance (Ablation: zero out context)
        zero_ctx = torch.zeros_like(ctx_data)
        uni_pred = model(ts_data, zero_ctx)
        uni_mse = criterion(uni_pred, targets).item()
        
    print(f"\n--- Ablation Study Results ---")
    print(f"Full Multimodal MSE : {full_mse:.6f}")
    print(f"Unimodal MSE        : {uni_mse:.6f}")
    
    # Calculate relative improvement
    improvement = ((uni_mse - full_mse) / uni_mse) * 100
    print(f"Multimodal Gain     : {improvement:.2f}%")
    
    if full_mse < uni_mse:
        print("Success: Multimodal fusion is successfully extracting predictive signal.")
    else:
        print("Warning: Contextual data is not improving accuracy.")

def main():
    # Setup mock dimensions
    embed_dim = 64
    batch_size = 8
    seq_len = 50
    ctx_len = 10
    
    # Initialize the Gated Fusion model
    model = AdvancedGatedFusion(embed_dim=embed_dim)
    
    # Create random input tensors
    ts_mock = torch.randn(batch_size, seq_len, embed_dim)
    ctx_mock = torch.randn(batch_size, ctx_len, embed_dim)
    targets_mock = torch.randn(batch_size, seq_len, embed_dim)
    
    # Execute the test
    run_ablation_study(model, ts_mock, ctx_mock, targets_mock)

if __name__ == "__main__":
    main()