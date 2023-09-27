import torch
from torch import nn
from typing import Optional, Tuple

class BiCrossLayer(nn.Module):
    def __init__(self, 
                feature_dim: int, 
                num_heads: int, 
                dropout: float = 0.1,
                layer_norm_eps: float = 1e-5):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(feature_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x1_mask: Optional[torch.Tensor] = None, x2_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y1= self.attn1(query=x1, key=x2, value=x2, key_padding_mask=x2_mask, need_weights=False)[0]
        y2 = self.attn2(query=x2, key=x1, value=x1, key_padding_mask=x1_mask, need_weights=False)[0]
        y1, y2 = self.dropout1(y1), self.dropout2(y2)
        y1, y2 = self.norm1(x1 + y1), self.norm1(x2 + y2)
        return y1, y2


class BiCrossAttentionModule(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, num_layers: int) -> None:
        '''
        Description : Neck of model，use CrossAttention to fuse features of Graph and Instruction
        param        {*} self: 
        param        {int} feature_dim: d_model
        param        {int} num_heads: 
        param        {int} num_layers: num of bi-cross layer
        return       {*}
        '''
        super().__init__()
        self.layers = nn.ModuleList([BiCrossLayer(feature_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x1_mask: Optional[torch.Tensor] = None, x2_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer in self.layers:
            x1, x2 = layer(x1, x2, x1_mask, x2_mask)
        return x1, x2
    