import torch
import sys

def allocate_gpu_memory():
    # Check if CUDA (GPU) is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU installation.")
        sys.exit(1)
        
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Using GPU: {gpu_name}")
    
    # Reset tracking and clear cache
    torch.cuda.empty_cache()
    
    try:
        # A float32 uses 4 bytes of memory.
        # We target a 14 GB allocation to comfortably exceed 12 GB.
        # 14 GB = 14 * 1024 * 1024 * 1024 bytes
        bytes_target = 14 * 1024 * 1024 * 1024
        num_elements = bytes_target // 4
        
        # Dimensions for a large 2D matrix
        rows = 40000
        cols = num_elements // rows
        
        print(f"Allocating a [{rows} x {cols}] float32 matrix...")
        print("This will request approximately 14 GB of VRAM.")
        
        # Allocate the memory directly on the GPU
        large_tensor = torch.randn(rows, cols, dtype=torch.float32, device='cuda')
        
        # Perform a dummy operation to ensure the memory is actively used
        dummy_result = torch.matmul(large_tensor[:100, :], large_tensor[:cols, :100])
        
        # Print actual allocated memory
        allocated_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"Successfully allocated {allocated_vram:.2f} GB of VRAM.")
        
        # Keep the memory held until user input
        input("Press Enter to release the memory and close the script...")
        
        # Cleanup
        del large_tensor
        del dummy_result
        torch.cuda.empty_cache()
        print("Memory released successfully.")
        
    except RuntimeError as e:
        print(f"\nAllocation Failed: {e}")
        print("Your GPU may have less than 14 GB of available VRAM.")

if __name__ == "__main__":
    allocate_gpu_memory()
