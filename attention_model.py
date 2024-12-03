import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to(x.device)
        return x

class AttentionBasedModel(nn.Module):
    def __init__(self, config):
        super(AttentionBasedModel, self).__init__()
        d_model = config['attention_dim']
        embed_dim = config['embed_dim']
        nhead = config['num_heads']
        dropout = config['dropout']
        self.use_positional_encoding = config.get('use_positional_encoding', True)

        # Projection layers
        self.audio_proj = nn.Linear(embed_dim, d_model)
        self.caption_proj = nn.Linear(embed_dim, d_model)
        self.hypothesis_proj = nn.Linear(embed_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Multihead self-attention
        self.audio_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.caption_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.hypothesis_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network for classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, config['hidden_dim']),  # Adjust input size for fusion strategy
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['num_classes'])
        )

    def forward(self, E_a, E_c, E_h):
        """
        Forward pass with optional return of attention weights.

        Args:
        - E_a (torch.Tensor): Audio embeddings.
        - E_c (torch.Tensor): Caption embeddings.
        - E_h (torch.Tensor): Hypothesis embeddings.
        - return_attention (bool): Whether to return attention weights.

        Returns:
        - logits (torch.Tensor): Output logits for classification.
        - attention_weights (dict, optional): Attention weights if requested.
        """
        # Project to d_model
        E_a = self.audio_proj(E_a)
        E_c = self.caption_proj(E_c)
        E_h = self.hypothesis_proj(E_h)

        # Add sequence length dimension if missing
        if E_a.dim() == 2:
            E_a = E_a.unsqueeze(1)
        if E_c.dim() == 2:
            E_c = E_c.unsqueeze(1)
        if E_h.dim() == 2:
            E_h = E_h.unsqueeze(1)

        # Transpose for MultiheadAttention [seq_len, batch_size, d_model]
        E_a = E_a.transpose(0, 1)
        E_c = E_c.transpose(0, 1)
        E_h = E_h.transpose(0, 1)

        # Handle seq_len = 1
        seq_len = 5  # Simulate sequence length
        E_a = E_a.repeat(seq_len, 1, 1)
        E_c = E_c.repeat(seq_len, 1, 1)
        E_h = E_h.repeat(seq_len, 1, 1)

        # Apply positional encoding
        if self.use_positional_encoding:
            E_a = self.positional_encoding(E_a)
            E_c = self.positional_encoding(E_c)
            E_h = self.positional_encoding(E_h)

        # Self-attention
        E_a_sa, _ = self.audio_self_attn(E_a, E_a, E_a)  # [seq_len, batch_size, d_model]
        E_c_sa, _ = self.caption_self_attn(E_c, E_c, E_c)
        E_h_sa, _ = self.hypothesis_self_attn(E_h, E_h, E_h)

        # Cross-modal attention
        E_h_cross, _ = self.cross_attn(E_h_sa, E_a_sa, E_a_sa)
        E_c_cross, _ = self.cross_attn(E_c_sa, E_a_sa, E_a_sa)

        # Mean pooling over sequence length
        E_h_sa_mean = E_h_sa.mean(dim=0)
        E_h_cross_mean = E_h_cross.mean(dim=0)
        E_c_sa_mean = E_c_sa.mean(dim=0)
        E_c_cross_mean = E_c_cross.mean(dim=0)

        # Feature fusion
        F_h = torch.cat([E_h_sa_mean, E_h_cross_mean], dim=-1)
        F_c = torch.cat([E_c_sa_mean, E_c_cross_mean], dim=-1)
        F_combined = torch.cat([F_h, F_c], dim=-1)

        # Classification
        logits = self.classifier(F_combined)

        return logits