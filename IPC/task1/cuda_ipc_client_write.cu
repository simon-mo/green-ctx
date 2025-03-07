#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#define ITERATIONS 100

// Kernel that writes a constant value to every element of dst.
__global__ void write_kernel(float* dst, float value, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        dst[tid] = value;
    }
}

// Check if a file exists
bool file_exists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

// Error checking function
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Client Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Open results file for writing (CSV)
    FILE* fp_out = fopen("results_write.csv", "w");
    if (!fp_out) {
        perror("Client: fopen results_write.csv");
        exit(EXIT_FAILURE);
    }
    // CSV header: Memory size (bytes), IPC write kernel avg time (ms), Local write kernel avg time (ms)
    fprintf(fp_out, "Size_Bytes,IPC_Write_Kernel_Time_ms,Local_Write_Kernel_Time_ms\n");
    fflush(fp_out);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    while (true) {
        if (file_exists("server_done.txt")) {
            printf("Client: Detected server completion. Exiting.\n");
            break;
        }
        if (!file_exists("ipc_handle.bin") || !file_exists("size.txt")) {
            usleep(100000); // 100ms
            continue;
        }

        // Read the memory size from file "size.txt"
        size_t size_bytes;
        FILE* fp = fopen("size.txt", "r");
        if (!fp) {
            perror("Client: fopen size.txt");
            exit(EXIT_FAILURE);
        }
        fscanf(fp, "%zu", &size_bytes);
        fclose(fp);

        // Read the IPC handle from "ipc_handle.bin"
        cudaIpcMemHandle_t ipcHandle;
        fp = fopen("ipc_handle.bin", "rb");
        if (!fp) {
            perror("Client: fopen ipc_handle.bin");
            exit(EXIT_FAILURE);
        }
        fread(&ipcHandle, sizeof(ipcHandle), 1, fp);
        fclose(fp);

        // Open the server-shared GPU memory using the IPC handle
        float* ipc_d_ptr = nullptr;
        checkCuda(cudaIpcOpenMemHandle((void**)&ipc_d_ptr, ipcHandle, cudaIpcMemLazyEnablePeerAccess),
                  "cudaIpcOpenMemHandle in client");

        // Calculate the number of float elements
        size_t numElements = size_bytes / sizeof(float);

        // Determine grid and block dimensions for the kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // ---------------------- IPC Memory Write Kernel Test ----------------------
        // Pre-warm the kernel on IPC memory
        write_kernel<<<blocksPerGrid, threadsPerBlock>>>(ipc_d_ptr, 3.14f, numElements);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warm-up (IPC)");

        float totalTime_ipc = 0.0f;
        for (int i = 0; i < ITERATIONS; i++) {
            checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start (IPC)");
            write_kernel<<<blocksPerGrid, threadsPerBlock>>>(ipc_d_ptr, 3.14f, numElements);
            checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord stop (IPC)");
            checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize (IPC)");
            float ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime (IPC)");
            totalTime_ipc += ms;
        }
        float avgTime_ipc = totalTime_ipc / ITERATIONS;

        // ---------------------- Local Memory Write Kernel Test ----------------------
        // Allocate local GPU memory in the client
        float* local_d_ptr = nullptr;
        checkCuda(cudaMalloc(&local_d_ptr, size_bytes), "cudaMalloc for local memory in client");

        // Pre-warm the kernel on local memory
        write_kernel<<<blocksPerGrid, threadsPerBlock>>>(local_d_ptr, 3.14f, numElements);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warm-up (local)");

        float totalTime_local = 0.0f;
        for (int i = 0; i < ITERATIONS; i++) {
            checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start (local)");
            write_kernel<<<blocksPerGrid, threadsPerBlock>>>(local_d_ptr, 3.14f, numElements);
            checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord stop (local)");
            checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize (local)");
            float ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime (local)");
            totalTime_local += ms;
        }
        float avgTime_local = totalTime_local / ITERATIONS;

        // Write the test results to CSV
        fprintf(fp_out, "%zu,%.6f,%.6f\n", size_bytes, avgTime_ipc, avgTime_local);
        fflush(fp_out);
        printf("Client: Size: %zu bytes, IPC write kernel avg time: %.6f ms, Local write kernel avg time: %.6f ms\n",
               size_bytes, avgTime_ipc, avgTime_local);

        // Free all allocated resources to minimize memory usage
        checkCuda(cudaFree(local_d_ptr), "cudaFree local memory in client");
        checkCuda(cudaIpcCloseMemHandle(ipc_d_ptr), "cudaIpcCloseMemHandle in client");

        // Signal completion for this test iteration
        fp = fopen("client_done.txt", "w");
        if (fp) {
            fprintf(fp, "done");
            fclose(fp);
        }
        remove("ipc_handle.bin");
        remove("size.txt");

        usleep(100000); // Delay before next iteration
    }

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start in client");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop in client");
    fclose(fp_out);

    return 0;
}
