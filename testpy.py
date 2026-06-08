import torch
import sys
import time

def allocate_vram():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        sys.exit(1)
        
    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    
    # Clear any cached memory before starting
    torch.cuda.empty_cache()
    
    try:
        # Target: 13 GB 
        # Float32 uses 4 bytes per element
        bytes_target = 13 * 1024 * 1024 * 1024
        num_elements = bytes_target // 4
        
        # Define a massive 2D matrix shape
        rows = 40000
        cols = num_elements // rows
        
        print(f"Allocating a [{rows} x {cols}] tensor (13 GB)...")
        
        # Directly initialize on the GPU
        large_tensor = torch.zeros((rows, cols), dtype=torch.float32, device='cuda')
        
        # Read a tiny piece to force CUDA to physically back the allocation
        _ = large_tensor[0, 0].item()
        
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"Successfully holding {allocated_gb:.2f} GB of VRAM.")
        
        input("\nPress Enter to release the VRAM and exit...")
        
        # Explicit cleanup
        del large_tensor
        torch.cuda.empty_cache()
        print("VRAM released.")
        
    except RuntimeError as e:
        print(f"\nAllocation failed: {e}")
        print("You might not have 13 GB of *free* VRAM available right now.")

if __name__ == "__main__":
    allocate_vram()
