#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>

__global__ void server_update_kernel(double* d_ptr) {
    unsigned long long cycles = clock64();
    d_ptr[0] = 1.0;
    d_ptr[1] = (double)cycles;
    printf("Server kernel: Updated value = %.0f, recorded cycles = %llu\n", d_ptr[0], cycles);
}

void checkCuda(cudaError_t err, const char* msg) {
    if(err != cudaSuccess) {
        fprintf(stderr, "Server Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    double* d_ptr = nullptr;
    checkCuda(cudaMalloc(&d_ptr, 2 * sizeof(double)), "cudaMalloc");
    double init[2] = {0.0, 0.0};
    checkCuda(cudaMemcpy(d_ptr, init, 2 * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy init");
    cudaIpcMemHandle_t ipcHandle;
    checkCuda(cudaIpcGetMemHandle(&ipcHandle, d_ptr), "cudaIpcGetMemHandle");

    FILE* fp = fopen("ipc_handle.bin", "wb");
    if (!fp) {
        perror("Server: fopen ipc_handle.bin");
        exit(EXIT_FAILURE);
    }
    fwrite(&ipcHandle, sizeof(ipcHandle), 1, fp);
    fclose(fp);
    printf("Server: IPC handle written to ipc_handle.bin\n");
    sleep(5);
    server_update_kernel<<<1, 1>>>(d_ptr);
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double values[2] = {0.0, 0.0};
    checkCuda(cudaMemcpy(values, d_ptr, 2 * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy update");
    printf("Server: Updated value = %.0f, recorded cycles = %.0f\n", values[0], values[1]);
    sleep(10);
    return 0;
}
