import time
import numpy as np
import psutil
import os
import platform

def matrix_multiplication_performance(matrix_size):
    # Initialize matrices
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    current_process = psutil.Process(os.getpid())

    # Initial metrics
    start_cpu_times = psutil.cpu_times_percent(interval=None)
    initial_memory_percent = psutil.virtual_memory().percent
    disk_io_initial = psutil.disk_io_counters()
    initial_read_bytes = disk_io_initial.read_bytes
    initial_write_bytes = disk_io_initial.write_bytes

    # Execution time measurement
    start_time = time.time()
    result_matrix = np.dot(matrix_a, matrix_b)
    end_time = time.time()

    # Final metrics
    end_cpu_times = psutil.cpu_times_percent(interval=1)
    final_memory_percent = psutil.virtual_memory().percent
    disk_io_final = psutil.disk_io_counters()
    final_read_bytes = disk_io_final.read_bytes
    final_write_bytes = disk_io_final.write_bytes

    # Calculations
    execution_time = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_usage_percent = final_memory_percent
    total_data_transferred_mb = ((final_read_bytes - initial_read_bytes) +
                                 (final_write_bytes - initial_write_bytes)) / (1024 * 1024)
    disk_throughput = total_data_transferred_mb / execution_time if execution_time > 0 else 0
    pei = 1 / (execution_time * cpu_usage * memory_usage_percent) if cpu_usage > 0 and memory_usage_percent > 0 else 0

    return {
        "matrix_size": matrix_size,
        "execution_time": execution_time,
        "cpu_usage": cpu_usage,
        "memory_usage_percent": memory_usage_percent,
        "total_data_transferred_mb": total_data_transferred_mb,
        "disk_throughput": disk_throughput,
        "pei": pei
    }

if __name__ == "__main__":
    matrix_size = 1000  # Adjustable
    results = matrix_multiplication_performance(matrix_size)

    print("\n--- System & Test Details ---")
    print(f"OS                 : {platform.system()} {platform.release()}")
    print(f"CPU Cores          : {psutil.cpu_count(logical=True)}")
    print(f"Total RAM          : {round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB")
    print(f"Matrix Size        : {results['matrix_size']} x {results['matrix_size']}")

    print("\n--- Performance Metrics ---")
    print(f"Execution Time (T)           : {results['execution_time']:.4f} seconds")
    print(f"CPU Utilization (%)          : {results['cpu_usage']:.2f}%")
    print(f"Memory Utilization (%)       : {results['memory_usage_percent']:.2f}%")
    print(f"Total Data Transferred (MB)  : {results['total_data_transferred_mb']:.4f} MB")
    print(f"Disk Throughput (MB/s)       : {results['disk_throughput']:.4f} MB/s")
    print(f"Performance Efficiency Index : {results['pei']:.6f}")
