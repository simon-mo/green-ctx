#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define SIZE (1024 * 1024)     // transmision size (4MB)
#define ITERATIONS 1000        // iteration times

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    FILE* fp_out = fopen("results.csv", "w");
    if (!fp_out) {
        perror("fopen results.csv");
        exit(EXIT_FAILURE);
    }
    fprintf(fp_out, "Size_Bytes,IPC_Time_ms,Local_Time_ms\n");

    if (argc != 2) {
        printf("Usage: %s [server|client]\n", argv[0]);
        return 0;
    }

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    
    if (strcmp(argv[1], "server") == 0) {
        // Server
        float* d_ptr = nullptr;
        // init
        checkCuda(cudaMalloc(&d_ptr, SIZE * sizeof(float)), "cudaMalloc in server");
        checkCuda(cudaMemset(d_ptr, 1, SIZE * sizeof(float)), "cudaMemset in server");
        // IPC handle
        cudaIpcMemHandle_t ipcHandle;
        checkCuda(cudaIpcGetMemHandle(&ipcHandle, d_ptr), "cudaIpcGetMemHandle");
        // write IPC to file
        FILE* fp = fopen("ipc_handle.bin", "wb");
        if (!fp) {
            perror("fopen");
            exit(EXIT_FAILURE);
        }
        fwrite(&ipcHandle, sizeof(ipcHandle), 1, fp);
        fclose(fp);
        
        printf("Server: IPC handle written to file. d_ptr = %p\n", d_ptr);
        printf("Server: Press Enter to exit...\n");
        getchar();
        
        checkCuda(cudaFree(d_ptr), "cudaFree in server");
        printf("Server: Memory freed and exiting.\n");
    } 
    else if (strcmp(argv[1], "client") == 0) {
        // Client
        cudaIpcMemHandle_t ipcHandle;
        FILE* fp = fopen("ipc_handle.bin", "rb");
        if (!fp) {
            perror("fopen");
            exit(EXIT_FAILURE);
        }
        fread(&ipcHandle, sizeof(ipcHandle), 1, fp);
        fclose(fp);
        
        float* ipc_d_ptr = nullptr;
        checkCuda(cudaIpcOpenMemHandle((void**)&ipc_d_ptr, ipcHandle, cudaIpcMemLazyEnablePeerAccess),
                  "cudaIpcOpenMemHandle");
        printf("Client: Opened IPC memory: %p\n", ipc_d_ptr);
        

        float* h_data = (float*)malloc(SIZE * sizeof(float));
        if (!h_data) {
            perror("malloc");
            exit(EXIT_FAILURE);
        }
        
        // CUDA event for timing
        cudaEvent_t start, stop;
        checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
        checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
        // warm up
        checkCuda(cudaMemcpy(h_data, ipc_d_ptr, SIZE * sizeof(float), cudaMemcpyDeviceToHost),
                  "warm up memcpy (IPC)");
        
        // Result1: IPC memory copy
        float totalTime_ipc = 0.0f;
        for (int i = 0; i < ITERATIONS; i++) {
            checkCuda(cudaEventRecord(start), "cudaEventRecord start (IPC)");
            checkCuda(cudaMemcpy(h_data, ipc_d_ptr, SIZE * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy (IPC)");
            checkCuda(cudaEventRecord(stop), "cudaEventRecord stop (IPC)");
            checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize (IPC)");
            float ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime (IPC)");
            totalTime_ipc += ms;
        }
        float avgTime_ipc = totalTime_ipc / ITERATIONS;
        printf("Client: Average cudaMemcpy time from IPC memory: %f ms\n", avgTime_ipc);
        
        // Result2: baseline
        float* local_d_ptr = nullptr;
        checkCuda(cudaMalloc(&local_d_ptr, SIZE * sizeof(float)), "cudaMalloc local");
        checkCuda(cudaMemset(local_d_ptr, 1, SIZE * sizeof(float)), "cudaMemset local");
        // warmup
        checkCuda(cudaMemcpy(h_data, local_d_ptr, SIZE * sizeof(float), cudaMemcpyDeviceToHost),
                  "warm up memcpy (local)");
        
        float totalTime_local = 0.0f;
        for (int i = 0; i < ITERATIONS; i++) {
            checkCuda(cudaEventRecord(start), "cudaEventRecord start (local)");
            checkCuda(cudaMemcpy(h_data, local_d_ptr, SIZE * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy (local)");
            checkCuda(cudaEventRecord(stop), "cudaEventRecord stop (local)");
            checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize (local)");
            float ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime (local)");
            totalTime_local += ms;
        }
        float avgTime_local = totalTime_local / ITERATIONS;
        printf("Client: Average cudaMemcpy time from local memory: %f ms\n", avgTime_local);
        
        // clear
        checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
        checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
        free(h_data);
        checkCuda(cudaIpcCloseMemHandle(ipc_d_ptr), "cudaIpcCloseMemHandle");
        checkCuda(cudaFree(local_d_ptr), "cudaFree local");
        
        printf("Client: Tests completed. Exiting.\n");
    } 
    else {
        printf("Unknown mode: %s. Use 'server' or 'client'.\n", argv[1]);
    }
    
    return 0;
}
