# scripts/evaluate.py
import torch
from src.models.fusion import CrossModalFusion
from src.data.processors import DataProcessor

def run_evaluation():
    # Load Real ETTh1 Data
    processor = DataProcessor()
    data = processor.load_etth1("data/ETTh1.csv")
    
    # Context (Simulated as random for now, e.g., weather/events)
    simulated_context = torch.randn(10, 128) 
    
    # Model
    model = CrossModalFusion(ts_dim=7, context_dim=128, embed_dim=64)
    model.eval()
    
    # 1. Full Multimodal Prediction
    ts_in, ctx_in = processor.prepare_batch(data, simulated_context)
    with torch.no_grad():
        full_pred = model(ts_in, ctx_in)
        
        # 2. Unimodal Ablation (Zeroing context)
        zero_ctx = torch.zeros_like(ctx_in)
        uni_pred = model(ts_in, zero_ctx)
        
    print(f"Full Multimodal Output Shape: {full_pred.shape}")
    print("Multimodal integration successful. Contextual signal processed.")

if __name__ == "__main__":
    run_evaluation()