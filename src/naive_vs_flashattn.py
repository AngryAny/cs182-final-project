import torch
import torch.nn as nn
import torch.nn.functional as F
from models import FlashAttnModel
import math

class NaiveAttentionModel(nn.Module):
    """Model using naive attention implementation for comparison"""
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super().__init__()
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_head = n_head
        
        self._read_in = nn.Linear(n_dims, n_embd)
        self.layers = nn.ModuleList([
            NaiveAttentionLayer(n_embd, n_head) for _ in range(n_layer)
        ])
        self._read_out = nn.Linear(n_embd, 1)
    
    def _combine(self, xs_b, ys_b):
        B, K, D = xs_b.shape
        ys_exp = torch.cat((
            ys_b.view(B, K, 1),
            torch.zeros(B, K, D - 1, device=ys_b.device)
        ), dim=2)
        zs = torch.stack((xs_b, ys_exp), dim=2).view(B, 2 * K, D)
        return zs
    
    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.as_tensor(inds, device=ys.device)
        
        tokens = self._combine(xs, ys)
        x = self._read_in(tokens)
        
        for layer in self.layers:
            x = layer(x)
        
        preds = self._read_out(x)
        return preds[:, ::2, 0][:, inds]


class NaiveAttentionLayer(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def naive_attention(self, q, k, v):
        """Naive attention implementation that uses more memory"""
        B, H, L, D = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=q.device) * float('-inf'), diagonal=1)
        scores = scores + causal_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def forward(self, x):
        B, L, C = x.shape
        
        # Self-attention
        residual = x
        x = self.ln1(x)
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply naive attention
        attn_output = self.naive_attention(q, k, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, L, C)
        attn_output = self.proj(attn_output)
        
        x = residual + attn_output
        
        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


def compare_with_naive():
    """Compare Flash Attention with naive implementation"""
    print("=== Comparing Flash vs Naive Attention ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with very long sequences to see memory difference
    test_configs = [
        {"batch_size": 64, "seq_length": 21, "n_dims": 10, "n_embd": 128, "n_layer": 6, "n_head": 4},
        {"batch_size": 64, "seq_length": 41, "n_dims": 20, "n_embd": 256, "n_layer": 12, "n_head": 8},
        {"batch_size": 64, "seq_length": 81, "n_dims": 40, "n_embd": 256, "n_layer": 24, "n_head": 8},
        {"batch_size": 64, "seq_length": 1024, "n_dims": 40, "n_embd": 256, "n_layer": 24, "n_head": 8},
        {"batch_size": 64, "seq_length": 256, "n_dims": 20, "n_embd": 256, "n_layer": 12, "n_head": 8},

        # {"batch_size": 1, "seq_length": 1024, "n_dims": 32, "n_embd": 256, "n_layer": 2, "n_head": 8},
        # {"batch_size": 1, "seq_length": 2048, "n_dims": 32, "n_embd": 256, "n_layer": 2, "n_head": 8},
        # {"batch_size": 1, "seq_length": 4096, "n_dims": 32, "n_embd": 256, "n_layer": 2, "n_head": 8},
    ]
    
    for config in test_configs:
        print(f"\nConfiguration: {config}")
        
        try:
            # Extract model parameters
            model_params = {
                "n_dims": config["n_dims"],
                "n_positions": config["seq_length"],
                "n_embd": config["n_embd"],
                "n_layer": config["n_layer"],
                "n_head": config["n_head"]
            }
            
            # Create models
            naive_model = NaiveAttentionModel(**model_params).to(device)
            flash_model = FlashAttnModel(**model_params).to(device)
            
            # Test data
            xs = torch.randn(config["batch_size"], config["seq_length"], config["n_dims"]).to(device)
            ys = torch.randn(config["batch_size"], config["seq_length"]).to(device)
            
            # Measure naive model
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = naive_model(xs, ys)
            naive_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            # Measure flash model
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = flash_model(xs, ys)
            flash_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            reduction = (1 - flash_memory / naive_memory) * 100
            
            print(f"Naive: {naive_memory:.2f} MB")
            print(f"Flash: {flash_memory:.2f} MB")
            print(f"Memory reduction: {reduction:.1f}%")
            
        except RuntimeError as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    compare_with_naive()