import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        pe = pe.unsqueeze(1)  # Adjusted dimension
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
        num_layers = config['num_layers']
        dim_feedforward = config['dim_feedforward']
        dropout = config['dropout']
        
        # Projection layers from embed_dim to d_model
        self.audio_proj = nn.Linear(embed_dim, d_model)
        self.caption_proj = nn.Linear(embed_dim, d_model)
        self.hypothesis_proj = nn.Linear(embed_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer layers (as before)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.audio_self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.caption_self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hypothesis_self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-attention layers
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward network for classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['num_classes'])
        )
        
    def forward(self, E_a, E_c, E_h):
        # E_a, E_c, E_h have shapes [batch_size, seq_len, embed_dim] or [batch_size, embed_dim]
        # Apply projection before transposing
        E_a = self.audio_proj(E_a)
        E_c = self.caption_proj(E_c)
        E_h = self.hypothesis_proj(E_h)
        
        # Ensure seq_len dimension is present
        if E_a.dim() == 2:
            E_a = E_a.unsqueeze(1)  # Add seq_len dimension
        if E_c.dim() == 2:
            E_c = E_c.unsqueeze(1)
        if E_h.dim() == 2:
            E_h = E_h.unsqueeze(1)
        
        # Transpose to [seq_len, batch_size, embed_dim] for Transformer
        E_a = E_a.transpose(0, 1)
        E_c = E_c.transpose(0, 1)
        E_h = E_h.transpose(0, 1)
        
        # Add positional encoding
        E_a = self.positional_encoding(E_a)
        E_c = self.positional_encoding(E_c)
        E_h = self.positional_encoding(E_h)
        
        # Intra-modal self-attention
        E_a_sa = self.audio_self_attn(E_a)
        E_c_sa = self.caption_self_attn(E_c)
        E_h_sa = self.hypothesis_self_attn(E_h)
        
        # Cross-modal cross-attention (Hypothesis attends to Audio)
        E_h_cross, _ = self.cross_attn(E_h_sa, E_a_sa, E_a_sa)
        
        # Cross-modal cross-attention (Caption attends to Audio)
        E_c_cross, _ = self.cross_attn(E_c_sa, E_a_sa, E_a_sa)
        
        # Mean pooling over sequence length
        E_h_sa_mean = E_h_sa.mean(dim=0)
        E_h_cross_mean = E_h_cross.mean(dim=0)
        E_c_sa_mean = E_c_sa.mean(dim=0)
        E_c_cross_mean = E_c_cross.mean(dim=0)
        
        # Feature fusion and representation
        F_h = torch.cat([E_h_sa_mean, E_h_cross_mean], dim=-1)
        F_c = torch.cat([E_c_sa_mean, E_c_cross_mean], dim=-1)
        
        # Combine fused features
        F_combined = torch.cat([F_h, F_c], dim=-1)
        
        # Classification
        logits = self.classifier(F_combined)
        
        return logits