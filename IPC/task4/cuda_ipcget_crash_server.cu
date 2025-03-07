#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>
#include <sys/stat.h>

// Function to check CUDA errors
void checkCuda(cudaError_t err, const char* msg) {
    if(err != cudaSuccess) {
        fprintf(stderr, "Server Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to write IPC handle to a file
void writeIpcHandleToFile(const char* filename, cudaIpcMemHandle_t* handle) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Server: fopen ipc_handle.bin");
        exit(EXIT_FAILURE);
    }
    fwrite(handle, sizeof(cudaIpcMemHandle_t), 1, fp);
    fclose(fp);
    printf("Server: IPC handle written to %s\n", filename);
}

// Helper function to create an external signal file
void createSignalFile(const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Server: fopen signal file");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "crash");
    fclose(fp);
    printf("Server: Signal file %s created\n", filename);
}

int main() {
    const int numElements = 10;
    // Allocate GPU memory for an array of 10 floats
    float* d_ptr = nullptr;
    checkCuda(cudaMalloc(&d_ptr, numElements * sizeof(float)), "cudaMalloc");

    // Initialize the array with known values: h_data[i] = i * 1.0f
    float* h_data = new float[numElements];
    for (int i = 0; i < numElements; i++) {
        h_data[i] = static_cast<float>(i);
    }
    checkCuda(cudaMemcpy(d_ptr, h_data, numElements * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy init");

    printf("Server: GPU memory allocated at %p, capacity = %d floats\n", d_ptr, numElements);

    // Obtain the IPC handle for the allocated memory
    cudaIpcMemHandle_t ipcHandle;
    checkCuda(cudaIpcGetMemHandle(&ipcHandle, d_ptr), "cudaIpcGetMemHandle");

    // Write the IPC handle to a file for the client to read
    writeIpcHandleToFile("ipc_handle.bin", &ipcHandle);

    // Optionally, print the initial array values
    printf("Server: Initial array values:\n");
    for (int i = 0; i < numElements; i++) {
        printf("  Value[%d] = %.1f\n", i, h_data[i]);
    }

    // Sleep for 10 seconds before simulating a crash
    printf("Server: Sleeping for 10 seconds before simulating crash...\n");
    sleep(10);

    // Create an external signal file to indicate the server is about to crash
    createSignalFile("server_crash.signal");

    // Simulate a crash by accessing illegal memory
    printf("Server: Simulating crash by accessing illegal memory...\n");
    int* illegal_ptr = nullptr;
    *illegal_ptr = 42;  // This will cause a segmentation fault and crash the server

    // Clean-up (never reached)
    delete[] h_data;
    checkCuda(cudaFree(d_ptr), "cudaFree");
    return 0;
}
