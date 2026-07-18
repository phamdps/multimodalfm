
import torch
import sys
import pytest
from src.fusion import CrossModalFusion

def test_model_forward_pass():
    # Initialize the module
    model = CrossModalFusion(ts_dim=1, context_dim=128, embed_dim=64)
    
    # Create dummy inputs
    ts_input = torch.randn(2, 50, 1)    # Batch size 2
    ctx_input = torch.randn(2, 10, 128)
    
    # Run forward pass
    output = model(ts_input, ctx_input)
    
    # Verify output shape (assuming output is [Batch, SeqLen, EmbedDim])
    assert output.shape[0] == 2
    assert output.shape[2] == 64
    print("Test passed: Model forward pass is correct.")

if __name__ == "__main__":
    test_model_forward_pass()
