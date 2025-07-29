import time
import numpy as np
import psutil
import os

def matrix_multiplication_performance(matrix_size):
    # Initialize matrices
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    # Get current process for memory metrics
    current_process = psutil.Process(os.getpid())

    # Capture initial system metrics
    try:
        initial_cpu_percent = psutil.cpu_percent(interval=None)
        initial_memory_mb = current_process.memory_info().rss / (1024 * 1024)  # Convert to MB
        disk_io_initial = psutil.disk_io_counters()
        initial_read_bytes = disk_io_initial.read_bytes if disk_io_initial else 0
        initial_write_bytes = disk_io_initial.write_bytes if disk_io_initial else 0
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError) as e:
        print(f"Error accessing initial system metrics: {e}")
        return None

    # Perform matrix multiplication and measure time
    start_time = time.time()
    result_matrix = np.dot(matrix_a, matrix_b)
    end_time = time.time()

    # Capture final system metrics
    try:
        final_cpu_percent = psutil.cpu_percent(interval=1)
        final_memory_mb = current_process.memory_info().rss / (1024 * 1024)  # Convert to MB
        disk_io_final = psutil.disk_io_counters()
        final_read_bytes = disk_io_final.read_bytes if disk_io_final else 0
        final_write_bytes = disk_io_final.write_bytes if disk_io_final else 0
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError) as e:
        print(f"Error accessing final system metrics: {e}")
        return None

    # Calculate performance metrics
    execution_time = end_time - start_time
    average_cpu_percent = (initial_cpu_percent + final_cpu_percent) / 2
    total_memory_used_mb = final_memory_mb
    memory_change_mb = final_memory_mb - initial_memory_mb
    disk_read_mb = (final_read_bytes - initial_read_bytes) / (1024 * 1024)  # Convert to MB
    disk_write_mb = (final_write_bytes - initial_write_bytes) / (1024 * 1024)  # Convert to MB

    return (
        execution_time,
        average_cpu_percent,
        total_memory_used_mb,
        memory_change_mb,
        disk_read_mb,
        disk_write_mb
    )

if __name__ == "__main__":
    matrix_size = 1000  # Adjust based on system capability
    results = matrix_multiplication_performance(matrix_size)

    if results:
        print("\n--- Performance Metrics ---")
        print(f"Execution Time        : {results[0]:.4f} seconds")
        print(f"Average CPU Usage     : {results[1]:.2f}%")
        print(f"Total Memory Used     : {results[2]:.2f} MB")
        print(f"Memory Usage Change   : {results[3]:.2f} MB")
        print(f"Disk Read I/O         : {results[4]:.4f} MB")
        print(f"Disk Write I/O        : {results[5]:.4f} MB")
    else:
        print("Failed to collect performance metrics.")
