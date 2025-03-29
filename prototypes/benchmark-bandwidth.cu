// Include necessary headers
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#define CHECK_CUDA(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            const char *errorString; \
            cuGetErrorString(result, &errorString); \
            fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, errorString); \
            exit(1); \
        } \
    } while(0)


// Constants
const unsigned int strideLen = 16; // Cache line size: 128 Bytes, 16 words
const unsigned int numBlocks = 132*9; // Number of SMs on H100
const unsigned int numThreadsPerBlock = 1024;
const unsigned long long loopCount = 500;
const unsigned int numGreenSMs = 24;

// CUDA kernel to test memory bandwidth
__global__ void bandwidthTestKernel(unsigned long long *data) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = tid * strideLen * loopCount; // Stride access

    for (unsigned int j = 0; j < loopCount; j++) {
        // read all data in stride
        #pragma unroll
        for (unsigned int i = 0; i < strideLen; i++) {
            unsigned long long value = data[index+j*strideLen+i];
        }
    }

    // #pragma unroll
    // for (unsigned int i = 0; i < strideLen; i++) {
    //     unsigned long long value = data[index+i];
    // }
    // if (index < n) {
    //     // read all data in stride
    //     unsigned long long value_1 = data[index+0];
    // }
}

void setGreenSMs(unsigned int numSMs) {
    CUdevice device;
    CUcontext context;

    CHECK_CUDA(cuDeviceGet(&device, 0));
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // now, opt into green context
    CUdevResource sm_resource;
    CHECK_CUDA(cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));
    printf("SM Resource: %d\n", sm_resource.sm.smCount);

    // Split the SM resource
    unsigned int nbGroups = 1; // Desired number of groups
    CUdevResource result_resources[nbGroups]; // Array to store split resources
    CUdevResource remaining;
    CHECK_CUDA(cuDevSmResourceSplitByCount(result_resources, &nbGroups, &sm_resource, &remaining, 0, numSMs));

    CUdevResourceDesc desc;
    CHECK_CUDA(cuDevResourceGenerateDesc(&desc, &result_resources[0], 1));

    CUgreenCtx green_ctx;
    CHECK_CUDA(cuGreenCtxCreate(&green_ctx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));

    CUdevResource green_sm_resource;
    CHECK_CUDA(cuGreenCtxGetDevResource(green_ctx, &green_sm_resource, CU_DEV_RESOURCE_TYPE_SM));
    printf("Green SM Resource: %d\n", green_sm_resource.sm.smCount);

    CUcontext green_ctx_ctx;
    CHECK_CUDA(cuCtxFromGreenCtx(&green_ctx_ctx, green_ctx));

    CHECK_CUDA(cuCtxSetCurrent(green_ctx_ctx));
}

int main() {
    // Number of trials for averaging bandwidth measurements
    const int numTrials = 10;

    // Define the size of the data array
    // Total number of threads
    unsigned long long totalThreads = numThreadsPerBlock * numBlocks ;

    // Total number of elements in the data array
    unsigned long long n = totalThreads * strideLen * loopCount;

    // Calculate total data size in bytes
    size_t dataSize = n * sizeof(unsigned long long);
    size_t sizeInGB = dataSize / (1024 * 1024 * 1024);
    printf("Data size: %lu GB\n", sizeInGB);

    // Allocate memory on the GPU
    unsigned long long *d_data;
    cudaError_t err = cudaMalloc((void **)&d_data, dataSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Initialize device memory to zero
    err = cudaMemset(d_data, 0, dataSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to initialize device memory - %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    // Configure kernel launch parameters
    dim3 blockSize(numThreadsPerBlock);
    dim3 gridSize(numBlocks);

    // CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Set the number of SMs to use
    setGreenSMs(numGreenSMs);

    float totalElapsedTime = 0.0f;

    // Run the kernel multiple times to average the bandwidth measurement
    for (int trial = 0; trial < numTrials; trial++) {
        // Record the start event
        cudaEventRecord(startEvent, 0);

        // Launch the kernel
        bandwidthTestKernel<<<gridSize, blockSize>>>(d_data);

        // Record the stop event
        cudaEventRecord(stopEvent, 0);

        // Wait for the event to complete
        cudaEventSynchronize(stopEvent);

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch kernel - %s\n", cudaGetErrorString(err));
            cudaFree(d_data);
            return EXIT_FAILURE;
        }

        // Calculate elapsed time
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent); // Time in milliseconds

        // Accumulate the total elapsed time
        totalElapsedTime += elapsedTime;
    }

    // Compute average elapsed time in seconds
    float avgElapsedTime = (totalElapsedTime / numTrials) / 1000.0f; // Convert to seconds

    // Calculate total bytes transferred
    // Each thread reads and writes an 8-byte value
    unsigned long long totalBytes = (unsigned long long)totalThreads * sizeof(unsigned long long);

    // Compute memory bandwidth in GB/s
    double bandwidth = (double)totalBytes / (avgElapsedTime * 1e9); // Convert bytes to GB

    // Output the results
    printf("Average kernel execution time: %f seconds\n", avgElapsedTime);
    printf("Memory Bandwidth: %f GB/s\n", bandwidth);

    // Clean up
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_data);

    return EXIT_SUCCESS;
}