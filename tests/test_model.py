import torch
from src.models.fusion import CrossModalFusion
from src.data.processors import DataProcessor

def test_full_pipeline():
    # 1. Setup
    processor = DataProcessor()
    model = CrossModalFusion(ts_dim=1, context_dim=128, embed_dim=64)
    
    # 2. Simulate raw data
    raw_ts = torch.randn(2, 50).numpy()  # Batch 2, Seq 50
    raw_ctx = torch.randn(2, 10, 128).numpy()
    
    # 3. Process
    ts_in, ctx_in = processor.create_batch(raw_ts, raw_ctx)
    
    # 4. Predict
    output = model(ts_in, ctx_in)
    
    # 5. Assert
    assert output.shape == (2, 50, 64)
    print("Test passed: Data pipeline and model forward pass are integrated.")

if __name__ == "__main__":
    test_full_pipeline()