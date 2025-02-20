class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output):
        # Shape debugging
        print(f"\nCross-Attention Debug:")
        print(f"Text features shape: {x.shape}")
        print(f"Image features shape: {encoder_output.shape}")
        
        # Transform inputs
        query = self.query(x)
        key = self.key(encoder_output)
        value = self.value(encoder_output)
        
        # Compute attention scores
        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = attention_weights / math.sqrt(self.attention_head_size)
        print(f"Attention weights stats: min={attention_weights.min():.4f}, max={attention_weights.max():.4f}")
        
        # Check attention distribution
        attention_distribution = torch.softmax(attention_weights, dim=-1)
        print(f"Attention distribution: min={attention_distribution.min():.4f}, max={attention_distribution.max():.4f}")
        
        # Apply attention to values
        context = torch.matmul(attention_distribution, value)
        
        return context 