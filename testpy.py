import torch
import sys
import time

def large_matrix_multiplication():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        sys.exit(1)
        
    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    
    # Clear cache before starting
    torch.cuda.empty_cache()
    
    # Dimensions for a heavy workload
    # Two 40,000 x 40,000 matrices in float16 will utilize significant VRAM
    dim = 40000 
    
    try:
        print(f"\nInitializing two [{dim} x {dim}] matrices on the GPU...")
        
        # Using float16 via mixed precision to fit comfortably within 15GB VRAM
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            print("Allocating Matrix A and Matrix B...")
            mat_a = torch.randn(dim, dim, device='cuda')
            mat_b = torch.randn(dim, dim, device='cuda')
            
            allocated_init = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"VRAM used for input matrices: {allocated_init:.2f} GB")
            
            print("\nExecuting matrix multiplication (A x B)...")
            start_time = time.time()
            
            # Perform the multiplication
            result = torch.matmul(mat_a, mat_b)
            
            # Force CUDA to finish the operation before stopping the timer
            torch.cuda.synchronize()
            end_time = time.time()
            
            allocated_total = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"Multiplication complete!")
            print(f"Peak VRAM held: {allocated_total:.2f} GB")
            print(f"Execution time: {end_time - start_time:.2f} seconds")
            
        # Keep memory alive until user decides to exit
        input("\nPress Enter to release the matrices and exit...")
        
        # Explicitly clean up
        del mat_a
        del mat_b
        del result
        torch.cuda.empty_cache()
        print("VRAM successfully released.")
        
    except RuntimeError as e:
        print(f"\nExecution failed: {e}")
        print("If you hit an Out of Memory error, try reducing the 'dim' variable slightly (e.g., to 35000).")

if __name__ == "__main__":
    large_matrix_multiplication()
