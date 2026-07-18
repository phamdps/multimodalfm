import torch
import torch.nn as nn

class AdvancedGatedFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        # Cross-attention to align context to time-series
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # Gating mechanism to determine context importance
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ts_feat: torch.Tensor, ctx_feat: torch.Tensor) -> torch.Tensor:
        # Cross-attention: TS as Query, Context as Key/Value
        attn_out, _ = self.cross_attn(ts_feat, ctx_feat, ctx_feat)
        
        # Gating: Combine attn_out with original TS features
        combined = torch.cat([attn_out, ts_feat], dim=-1)
        gate_weight = self.gate(combined)
        
        # Residual fusion with learned gating
        fused = self.norm(ts_feat + (gate_weight * attn_out))
        return fused