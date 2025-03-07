#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

// Function to check if a file exists
bool file_exists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

// Error checking function
void checkCuda(cudaError_t err, const char* msg) {
    if(err != cudaSuccess) {
        fprintf(stderr, "Client Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Wait until the IPC handle file is available
    while (!file_exists("ipc_handle.bin")) {
        usleep(100000);  // wait 100ms
    }
    
    // Read the IPC handle from file
    cudaIpcMemHandle_t ipcHandle;
    FILE* fp = fopen("ipc_handle.bin", "rb");
    if (!fp) {
        perror("Client: fopen ipc_handle.bin");
        exit(EXIT_FAILURE);
    }
    fread(&ipcHandle, sizeof(ipcHandle), 1, fp);
    fclose(fp);
    printf("Client: IPC handle read from file.\n");

    // Open the shared GPU memory using the IPC handle
    float* d_ptr = nullptr;
    checkCuda(cudaIpcOpenMemHandle((void**)&d_ptr, ipcHandle, cudaIpcMemLazyEnablePeerAccess),
              "cudaIpcOpenMemHandle");
    printf("Client: Shared GPU memory opened at %p (client's virtual address)\n", d_ptr);

    const int numElements = 10;
    float* h_data = new float[numElements];
    // Read the initial content once and output it
    checkCuda(cudaMemcpy(h_data, d_ptr, numElements * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy initial read");
    printf("Client: Initial array values read from shared memory:\n");
    for (int i = 0; i < numElements; i++) {
        printf("  Value[%d] = %.1f\n", i, h_data[i]);
    }

    // Polling loop: wait for the external signal file from the server
    printf("Client: Polling for server crash signal...\n");
    while (!file_exists("server_crash.signal")) {
        usleep(100000); // Poll every 100ms
    }
    printf("Client: Detected server crash signal. Attempting to re-read shared memory after sleeping 5 seconds...\n");
    sleep(5);
    // Attempt to re-read the shared memory after the signal is detected
    bool accessSuccess = true;
    int err = cudaMemcpy(h_data, d_ptr, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        accessSuccess = false;
        fprintf(stderr, "Client Error: Failed to re-read shared memory: %s\n", cudaGetErrorString((cudaError_t)err));
    } else {
        printf("Client: Successfully re-read shared memory after server crash. New values:\n");
        for (int i = 0; i < numElements; i++) {
            printf("  Value[%d] = %.1f\n", i, h_data[i]);
        }
    }

    // Clean up
    delete[] h_data;
    // Attempt to close the IPC memory handle (if still valid)
    checkCuda(cudaIpcCloseMemHandle(d_ptr), "cudaIpcCloseMemHandle");
    return 0;
}
