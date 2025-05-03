import torch
import torch.nn as nn
from models import FlashAttnModel, TransformerModel
import time
import gc

def test_memory_and_outputs():
    """Test memory usage and output similarity with longer sequences"""
    print("=== Flash Attention Memory and Output Test ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}\n")
    
    # Test parameters - use longer sequence for memory benefits
    batch_size = 4
    seq_length = 2048  # Longer sequence to see memory benefits
    n_dims = 32
    n_embd = 256
    n_layer = 4
    n_head = 8
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Embedding dim: {n_embd}")
    print(f"  Layers: {n_layer}")
    print(f"  Heads: {n_head}\n")
    
    # Create models
    standard_model = TransformerModel(
        n_dims=n_dims, 
        n_positions=seq_length, 
        n_embd=n_embd, 
        n_layer=n_layer, 
        n_head=n_head
    ).to(device)
    
    flash_model = FlashAttnModel(
        n_dims=n_dims, 
        n_positions=seq_length, 
        n_embd=n_embd, 
        n_layer=n_layer, 
        n_head=n_head
    ).to(device)
    
    # Copy weights to ensure identical parameters
    flash_model.load_state_dict(standard_model.state_dict())
    
    # Create test data
    xs = torch.randn(batch_size, seq_length, n_dims).to(device)
    ys = torch.randn(batch_size, seq_length).to(device)
    
    # Test forward pass (no gradients)
    print("1. Testing forward pass (inference):")
    
    def measure_inference(model, xs, ys, name):
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            output = model(xs, ys)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            peak_memory = 0
        
        elapsed_time = (end_time - start_time) * 1000  # ms
        
        return output, peak_memory, elapsed_time
    
    std_output, std_memory, std_time = measure_inference(standard_model, xs, ys, "Standard")
    flash_output, flash_memory, flash_time = measure_inference(flash_model, xs, ys, "Flash")
    
    print(f"  Standard Model - Memory: {std_memory:.2f} MB, Time: {std_time:.2f} ms")
    print(f"  Flash Model    - Memory: {flash_memory:.2f} MB, Time: {flash_time:.2f} ms")
    
    if device.type == 'cuda' and std_memory > 0:
        memory_reduction = (1 - flash_memory / std_memory) * 100
        speedup = std_time / flash_time
        print(f"  Memory reduction: {memory_reduction:.1f}%")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Check output similarity
    output_diff = torch.abs(std_output - flash_output).mean().item()
    max_diff = torch.abs(std_output - flash_output).max().item()
    print(f"  Mean output difference: {output_diff:.6f}")
    print(f"  Max output difference: {max_diff:.6f}")
    
    if output_diff < 0.01:
        print("  ✓ Outputs are similar")
    else:
        print("  ✗ WARNING: Outputs are significantly different!")
    
    # Test with gradients (training)
    print("\n2. Testing training (with gradients):")
    
    def measure_training(model, xs, ys, name):
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        start_time = time.time()
        
        # Forward pass
        output = model(xs, ys)
        loss = output.mean()  # Simple loss for testing
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            peak_memory = 0
        
        elapsed_time = (end_time - start_time) * 1000  # ms
        
        # Clear gradients
        optimizer.zero_grad()
        model.eval()
        
        return peak_memory, elapsed_time
    
    # Clone models for training test
    std_train_model = TransformerModel(n_dims=n_dims, n_positions=seq_length, n_embd=n_embd, n_layer=n_layer, n_head=n_head).to(device)
    flash_train_model = FlashAttnModel(n_dims=n_dims, n_positions=seq_length, n_embd=n_embd, n_layer=n_layer, n_head=n_head).to(device)
    
    std_train_memory, std_train_time = measure_training(std_train_model, xs, ys, "Standard")
    flash_train_memory, flash_train_time = measure_training(flash_train_model, xs, ys, "Flash")
    
    print(f"  Standard Model - Memory: {std_train_memory:.2f} MB, Time: {std_train_time:.2f} ms")
    print(f"  Flash Model    - Memory: {flash_train_memory:.2f} MB, Time: {flash_train_time:.2f} ms")
    
    if device.type == 'cuda' and std_train_memory > 0:
        train_memory_reduction = (1 - flash_train_memory / std_train_memory) * 100
        train_speedup = std_train_time / flash_train_time
        print(f"  Memory reduction: {train_memory_reduction:.1f}%")
        print(f"  Speedup: {train_speedup:.2f}x")
    
    # Test with different sequence lengths
    print("\n3. Testing different sequence lengths:")
    test_lengths = [512, 1024, 2048, 4096]
    
    for length in test_lengths:
        if length > seq_length:
            continue  # Skip if longer than model's max position
            
        test_xs = torch.randn(2, length, n_dims).to(device)
        test_ys = torch.randn(2, length).to(device)
        
        _, std_mem, _ = measure_inference(standard_model, test_xs, test_ys, "Standard")
        _, flash_mem, _ = measure_inference(flash_model, test_xs, test_ys, "Flash")
        
        if device.type == 'cuda' and std_mem > 0:
            reduction = (1 - flash_mem / std_mem) * 100
            print(f"  Seq length {length}: Memory reduction {reduction:.1f}%")


if __name__ == "__main__":
    test_memory_and_outputs()