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

int main() {
    const int numElements = 1000;
    // Allocate GPU memory for an array of 1000 floats
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

    // Write the IPC handle to a file so that the client can open it
    writeIpcHandleToFile("ipc_handle.bin", &ipcHandle);

    // Server loop: every 100ms read one element sequentially from the array
    int index = 0;
    float readValue = 0.0f;
    while (true) {
        checkCuda(cudaMemcpy(&readValue, d_ptr + index, sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy read element");
        printf("Server: Read element at index %d = %.1f\n", index, readValue);
        
        index = (index + 1) % numElements; // Cycle through the array
        usleep(100000); // Sleep 100ms
    }

    // Clean-up (never reached)
    delete[] h_data;
    checkCuda(cudaFree(d_ptr), "cudaFree");
    return 0;
}
