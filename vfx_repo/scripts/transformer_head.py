import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerHead(nn.Module):
    """Transformer-based head for audio classification"""
    
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=3, 
                 num_heads=8, dropout=0.1, max_seq_len=100):
        super(TransformerHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global average pooling + classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Create attention mask for padding (if needed)
        if seq_len < self.max_seq_len:
            # Pad to max_seq_len
            padding = torch.zeros(batch_size, self.max_seq_len - seq_len, self.hidden_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)
            seq_len = self.max_seq_len
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, seq_len, hidden_dim)
        
        # Global average pooling across sequence dimension
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        
        # Classification
        x = self.classifier(x)  # (batch_size, 1)
        
        return x
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability (optional)"""
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # Get attention weights from transformer
        attention_weights = []
        for layer in self.transformer.layers:
            # Extract attention weights from self-attention
            attn_output, attn_weights = layer.self_attn(x, x, x)
            attention_weights.append(attn_weights)
            x = layer.norm1(attn_output + x)
            x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))) + x
            x = layer.norm2(x)
        
        return attention_weights

class TransformerHeadV2(nn.Module):
    """Enhanced Transformer head with residual connections and layer norm"""
    
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=4, 
                 num_heads=8, dropout=0.1, max_seq_len=100):
        super(TransformerHeadV2, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer encoder with more sophisticated architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with better initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Classification
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    # Test the model
    model = TransformerHeadV2(input_dim=1024, hidden_dim=512, num_layers=4)
    
    # Create dummy input
    batch_size, seq_len, input_dim = 4, 50, 1024
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
