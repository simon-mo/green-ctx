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

    // const int numElements = 1000;
    int iteration = 0;
    float readValue = 0.0f;
    while (true) {
        if (iteration < 9) {
            // Read a valid element at index = iteration
            checkCuda(cudaMemcpy(&readValue, d_ptr + iteration, sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy read valid element");
            printf("Client: Read valid element at index %d = %.1f\n", iteration, readValue);
        } else if (iteration == 9) {
            // On the 10th iteration, intentionally access illegal memory (index = numElements, which is out-of-bound)
            // printf("Client: Attempting illegal memory access at index %d (out-of-bound)...\n", numElements);
            // checkCuda(cudaMemcpy(&readValue, d_ptr + numElements, sizeof(float), cudaMemcpyDeviceToHost),
            //           "cudaMemcpy illegal access");
            // printf("Client: Read illegal element = %.1f\n", readValue);
            printf("Client: Simulating crash now...\n");
            // abort();  // This will cause the client process to crash access illegal memory
            int *illegal_ptr = nullptr;
            *illegal_ptr = 42;
            // printf("Client: Simulating crash now...\n");
        } else {
            break;
        }
        iteration++;
        usleep(100000); // Sleep 100ms between reads
    }

    // Close the IPC memory handle (not reached if crash occurs)
    checkCuda(cudaIpcCloseMemHandle(d_ptr), "cudaIpcCloseMemHandle");
    return 0;
}
