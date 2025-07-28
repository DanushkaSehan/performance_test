# cpu_intensive_task.py
import time
import numpy as np
import os # For potential future use, though not directly for core metrics here

def matrix_multiplication(size):
    """
    Performs matrix multiplication of two randomly generated matrices.
    Measures and returns the execution time.
    """
    print(f"Starting matrix multiplication for size {size}x{size}...")

    # Create two random matrices of the specified size
    # Using float32 for potentially faster computation and less memory for large matrices
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    # Record the start time
    start_time = time.time()

    # Perform matrix multiplication
    C = np.dot(A, B)

    # Record the end time
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time

    print(f"Matrix multiplication completed in {execution_time:.4f} seconds.")
    return execution_time

if __name__ == "__main__":
    # Adjust matrix size based on system capability and desired execution time.
    # Start with a smaller size (e.g., 500) and increase it until the execution
    # time is noticeable (e.g., 30 seconds to a few minutes) on your VMs.
    # A size of 1000 (as in the sample) might be too long for 2GB RAM VMs.
    # You might need different sizes for different configurations/OSes to keep runtimes manageable.
    matrix_size = 2000 # [cite: 106] - Adjusted from 1000 for practical testing

    # You could potentially loop this multiple times and average, but for this assignment
    # a single run with a sufficiently large 'size' is likely what's expected.
    
    try:
        execution_time = matrix_multiplication(matrix_size)
        print(f"Final Reported Execution Time: {execution_time:.4f} seconds") # [cite: 107, 108]

        # In a real scenario, you might also want to log this directly to a file
        # along with configuration details.
        # with open("performance_log.txt", "a") as f:
        #     f.write(f"Size: {matrix_size}, Execution Time: {execution_time:.4f}s\n")

    except MemoryError:
        print(f"MemoryError: Matrix size {matrix_size} is too large for the current VM's RAM.")
        print("Please reduce the 'matrix_size' variable in the script.")
    except Exception as e:
        print(f"An error occurred: {e}")
