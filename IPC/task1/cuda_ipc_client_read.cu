#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#define ITERATIONS 100

// Kernel that reads from src and writes to dst
__global__ void read_kernel(const float* src, float* dst, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        dst[tid] = src[tid];
    }
}

bool file_exists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Client Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    FILE* fp_out = fopen("results.csv", "w");
    if (!fp_out) {
        perror("Client: fopen results.csv");
        exit(EXIT_FAILURE);
    }
    fprintf(fp_out, "Size_Bytes,IPC_Kernel_Time_ms,Local_Kernel_Time_ms\n");
    fflush(fp_out);

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

        size_t size_bytes;
        FILE* fp = fopen("size.txt", "r");
        if (!fp) {
            perror("Client: fopen size.txt");
            exit(EXIT_FAILURE);
        }
        fscanf(fp, "%zu", &size_bytes);
        fclose(fp);
        
        cudaIpcMemHandle_t ipcHandle;
        fp = fopen("ipc_handle.bin", "rb");
        if (!fp) {
            perror("Client: fopen ipc_handle.bin");
            exit(EXIT_FAILURE);
        }
        fread(&ipcHandle, sizeof(ipcHandle), 1, fp);
        fclose(fp);

        float* ipc_d_ptr = nullptr;
        checkCuda(cudaIpcOpenMemHandle((void**)&ipc_d_ptr, ipcHandle, cudaIpcMemLazyEnablePeerAccess),
                  "cudaIpcOpenMemHandle in client");

        size_t numElements = size_bytes / sizeof(float);
        
        // test1: IPC time
        float* d_dst_ipc = nullptr;
        checkCuda(cudaMalloc(&d_dst_ipc, size_bytes), "cudaMalloc d_dst_ipc in client");
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        read_kernel<<<blocksPerGrid, threadsPerBlock>>>(ipc_d_ptr, d_dst_ipc, numElements);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warm-up (IPC)"); // 一直不退出，直到读到第一个process
        float totalTime_ipc = 0.0f;
        for (int i = 0; i < ITERATIONS; i++) {
            checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start (IPC)");
            read_kernel<<<blocksPerGrid, threadsPerBlock>>>(ipc_d_ptr, d_dst_ipc, numElements);
            checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord stop (IPC)");
            checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize (IPC)");
            float ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime (IPC)");
            totalTime_ipc += ms;
        }
        float avgTime_ipc = totalTime_ipc / ITERATIONS;

        // printf("Client: Size: %zu bytes, IPC kernel avg time: %.6f ms, \n",
        //        size_bytes, avgTime_ipc);

        checkCuda(cudaFree(d_dst_ipc), "cudaFree d_dst_ipc in client");
        checkCuda(cudaIpcCloseMemHandle(ipc_d_ptr), "cudaIpcCloseMemHandle in client");

        // test2: local memory
        float* local_d_ptr = nullptr;
        checkCuda(cudaMalloc(&local_d_ptr, size_bytes), "cudaMalloc for local memory in client");
        checkCuda(cudaMemset(local_d_ptr, 1, size_bytes), "cudaMemset for local memory in client");
        float* d_dst_local = nullptr;
        checkCuda(cudaMalloc(&d_dst_local, size_bytes), "cudaMalloc d_dst_local in client");

        // warmup
        read_kernel<<<blocksPerGrid, threadsPerBlock>>>(local_d_ptr, d_dst_local, numElements);
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warm-up (local)");

        float totalTime_local = 0.0f;
        for (int i = 0; i < ITERATIONS; i++) {
            checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start (local)");
            read_kernel<<<blocksPerGrid, threadsPerBlock>>>(local_d_ptr, d_dst_local, numElements);
            checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord stop (local)");
            checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize (local)");
            float ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime (local)");
            totalTime_local += ms;
        }
        float avgTime_local = totalTime_local / ITERATIONS;

        fprintf(fp_out, "%zu,%.6f,%.6f\n", size_bytes, avgTime_ipc, avgTime_local);
        fflush(fp_out);

        printf("Client: Size: %zu bytes, IPC kernel avg time: %.6f ms, Local kernel avg time: %.6f ms\n",
               size_bytes, avgTime_ipc, avgTime_local);


        
        checkCuda(cudaFree(local_d_ptr), "cudaFree local memory in client");
        checkCuda(cudaFree(d_dst_local), "cudaFree d_dst_local in client");
        

        fp = fopen("client_done.txt", "w");
        if (fp) {
            fprintf(fp, "done");
            fclose(fp);
        }

        remove("ipc_handle.bin");
        remove("size.txt");

        usleep(100000);
    }

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start in client");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop in client");
    fclose(fp_out);

    return 0;
}
