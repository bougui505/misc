#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2026 Institut Pasteur                                       #
#############################################################################
#
# creation_date: 2026-02-23

import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads, key_padding_mask=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # The learned "summary" query
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.key_padding_mask = key_padding_mask

    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        batch_size = x.size(0)
        
        # Expand learned query to match batch size
        q = self.query.expand(batch_size, -1, -1)
        
        # Q attends to x (K and V)
        # attn_output shape: [batch, 1, embed_dim]
        attn_output, _ = self.mha(q, x, x, key_padding_mask=self.key_padding_mask)
        
        # Squeeze to get [batch, embed_dim]
        return attn_output.squeeze(1)

if __name__ == "__main__":
    # Example usage
    key_padding_mask = torch.rand(32,100) > 0.5
    pooler = AttentionPooling(embed_dim=512, num_heads=8, key_padding_mask=key_padding_mask)
    features = torch.randn(32, 100, 512) # 32 samples, 100 tokens
    print(f"{features.shape=}")
    pooled_out = pooler(features) # [32, 512]
    print(f"{pooled_out.shape=}")
