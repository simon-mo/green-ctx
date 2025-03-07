#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void polling_kernel(const double* shared, double* result) {
    const double eps = 1e-9;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (fabs(shared[0] - 1.0) > eps) {
            __threadfence();
        }
        unsigned long long client_cycles = clock64();
        unsigned long long server_cycles = (unsigned long long) shared[1];
        unsigned long long latency_cycles = client_cycles - server_cycles;
        result[0] = shared[0];  
        result[1] = (double) latency_cycles;
        printf("Client kernel: Detected update. Server cycles = %llu, client cycles = %llu, latency cycles = %llu\n",
               server_cycles, client_cycles, latency_cycles);
    }
}

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Client Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

bool file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

int main() {
    while (!file_exists("ipc_handle.bin")) {
        usleep(100000);  
    }

    cudaIpcMemHandle_t ipcHandle;
    FILE* fp = fopen("ipc_handle.bin", "rb");
    if (!fp) {
        perror("Client: fopen ipc_handle.bin");
        exit(EXIT_FAILURE);
    }
    fread(&ipcHandle, sizeof(ipcHandle), 1, fp);
    fclose(fp);
    printf("Client: IPC handle read from file\n");


    double* d_shared = nullptr;
    checkCuda(cudaIpcOpenMemHandle((void**)&d_shared, ipcHandle, cudaIpcMemLazyEnablePeerAccess),
              "cudaIpcOpenMemHandle");
    printf("Client: Shared GPU memory opened at %p\n", d_shared);

    double* d_result = nullptr;
    checkCuda(cudaMalloc(&d_result, 2 * sizeof(double)), "cudaMalloc d_result");

    polling_kernel<<<1, 1>>>(d_shared, d_result);
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    double* h_result = (double*)malloc(2 * sizeof(double));
    if (!h_result) {
        fprintf(stderr, "Client Error: malloc failed\n");
        exit(EXIT_FAILURE);
    }
    checkCuda(cudaMemcpy(h_result, d_result, 2 * sizeof(double), cudaMemcpyDeviceToHost),
              "cudaMemcpy d_result to h_result");
    unsigned int clock_rate_khz = 0;
    checkCuda(cudaDeviceGetAttribute((int*)&clock_rate_khz, cudaDevAttrClockRate, 0),
              "cudaDeviceGetAttribute");
    double us_per_cycle = 1000.0 / clock_rate_khz;
    double latency_us = h_result[1] * us_per_cycle;
    printf("Client: Detected update. Value = %.0f, latency cycles = %.0f, latency = %.2f us\n",
           h_result[0], h_result[1], latency_us);
    checkCuda(cudaIpcCloseMemHandle(d_shared), "cudaIpcCloseMemHandle");
    checkCuda(cudaFree(d_result), "cudaFree d_result");
    free(h_result);

    return 0;
}