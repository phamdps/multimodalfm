import torch
import torch.optim as optim
from src.models.fusion import CrossModalFusion
from src.data.processors import DataProcessor
from src.utils.config_manager import load_config

def train():
    # 1. Load config
    config = load_config()
    
    # 2. Init components
    processor = DataProcessor()
    model = CrossModalFusion(
        ts_dim=config['model']['ts_dim'],
        context_dim=config['model']['context_dim'],
        embed_dim=config['model']['embed_dim']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.MSELoss()

    # 3. Dummy training loop
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        # Simulate batch
        ts_raw = torch.randn(config['training']['batch_size'], 50).numpy()
        ctx_raw = torch.randn(config['training']['batch_size'], 10, 128).numpy()
        
        ts_in, ctx_in = processor.create_batch(ts_raw, ctx_raw)
        
        # Forward & Backward
        optimizer.zero_grad()
        output = model(ts_in, ctx_in)
        loss = criterion(output, torch.randn_like(output)) # Dummy target
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()

# python -m scripts.train