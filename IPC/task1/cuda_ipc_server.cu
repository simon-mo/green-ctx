#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#define INITIAL_SIZE (40ULL * 1024ULL * 1024ULL)      // 40MB
#define MAX_SIZE (40ULL * 1024ULL * 1024ULL * 1024ULL) // 40GB

bool file_exists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Server Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    size_t size_bytes = INITIAL_SIZE;
    while (size_bytes <= MAX_SIZE) {
        printf("Server: Testing memory size: %zu bytes\n", size_bytes);
        float* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, size_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "Server: Failed to allocate %zu bytes on GPU. Error: %s\n",
                    size_bytes, cudaGetErrorString(err));
            break;
        }
        checkCuda(cudaMemset(d_ptr, 1, size_bytes), "cudaMemset in server");
        cudaIpcMemHandle_t ipcHandle;
        checkCuda(cudaIpcGetMemHandle(&ipcHandle, d_ptr), "cudaIpcGetMemHandle in server");
        FILE* fp = fopen("ipc_handle.bin", "wb");
        if (!fp) {
            perror("Server: fopen ipc_handle.bin");
            exit(EXIT_FAILURE);
        }
        fwrite(&ipcHandle, sizeof(ipcHandle), 1, fp);
        fclose(fp);
        fp = fopen("size.txt", "w");
        if (!fp) {
            perror("Server: fopen size.txt");
            exit(EXIT_FAILURE);
        }
        fprintf(fp, "%zu", size_bytes);
        fclose(fp);

        printf("Server: IPC handle and size written for %zu bytes. Waiting for client...\n", size_bytes);
        while (!file_exists("client_done.txt")) {
            usleep(100000); // 100ms
        }
        remove("client_done.txt");
        checkCuda(cudaFree(d_ptr), "cudaFree in server");
        printf("Server: Test for %zu bytes completed. Moving to next size...\n", size_bytes);

        size_bytes *= 2;
    }

    FILE* fp_done = fopen("server_done.txt", "w");
    if (fp_done) {
        fprintf(fp_done, "done");
        fclose(fp_done);
    }

    printf("Server: All tests completed. Exiting.\n");
    return 0;
}
