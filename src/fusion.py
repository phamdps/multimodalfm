import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, ts_dim, context_dim, embed_dim):
        super(CrossModalFusion, self).__init__()
        # Projections to align different modalities to the same embedding dimension
        self.ts_proj = nn.Linear(ts_dim, embed_dim)
        self.ctx_proj = nn.Linear(context_dim, embed_dim)
        
        # Cross-Attention mechanism
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        # Final output layer
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, ts_input, ctx_input):
        # Project inputs
        ts_feat = self.ts_proj(ts_input)   # [Batch, SeqLen, Embed]
        ctx_feat = self.ctx_proj(ctx_input) # [Batch, CtxLen, Embed]
        
        # Use Time-Series as Query, Context as Key/Value
        fused, _ = self.cross_attn(ts_feat, ctx_feat, ctx_feat)
        
        return self.fc(fused)