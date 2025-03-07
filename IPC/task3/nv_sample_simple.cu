#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <cuda.h>

#define IPC_HANDLE_TYPE CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

void checkCudaErrors(CUresult err, const char* msg) {
    if(err != CUDA_SUCCESS) {
        const char *errStr = nullptr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "Error: %s: %s\n", msg, errStr ? errStr : "Unknown error");
        exit(EXIT_FAILURE);
    }
}

int main() {
    checkCudaErrors(cuInit(0), "cuInit");
    CUdevice cuDevice;
    checkCudaErrors(cuDeviceGet(&cuDevice, 0), "cuDeviceGet");
    CUcontext cuContext;
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice), "cuCtxCreate");

    size_t allocSize = 10 * 1024 * 1024; // 10 MB

    CUmemAllocationProp allocProp = {};
    allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = 0;
    allocProp.requestedHandleTypes = IPC_HANDLE_TYPE;

    size_t granularity = 0;
    checkCudaErrors(cuMemGetAllocationGranularity(&granularity, &allocProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                    "cuMemGetAllocationGranularity");
    if (allocSize % granularity != 0) {
        allocSize = ((allocSize + granularity - 1) / granularity) * granularity;
    }

    CUmemGenericAllocationHandle memHandle;
    checkCudaErrors(cuMemCreate(&memHandle, allocSize, &allocProp, 0), "cuMemCreate");
    printf("Allocated physical memory of size %zu bytes\n", allocSize);

    CUdeviceptr basePtr = 0;
    checkCudaErrors(cuMemAddressReserve(&basePtr, allocSize, 0, 0, 0), "cuMemAddressReserve");
    checkCudaErrors(cuMemMap(basePtr, allocSize, 0, memHandle, 0), "cuMemMap");

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkCudaErrors(cuMemSetAccess(basePtr, allocSize, &accessDesc, 1), "cuMemSetAccess");
    printf("Server: Mapped physical memory to virtual address %p\n", (void*)basePtr);

    double initData[2] = {3.1415, 2.71828};
    checkCudaErrors(cuMemcpyHtoD(basePtr, initData, 2 * sizeof(double)), "cuMemcpyHtoD");
    printf("Server: Wrote initial data: %.4f, %.4f\n", initData[0], initData[1]);

    int shareableHandle;
    checkCudaErrors(cuMemExportToShareableHandle(&shareableHandle, memHandle, IPC_HANDLE_TYPE, 0),
                    "cuMemExportToShareableHandle");
    printf("Server: Exported shareable handle: %d\n", shareableHandle);

    CUmemGenericAllocationHandle importedHandle;
    checkCudaErrors(cuMemImportFromShareableHandle(&importedHandle,
                  (void*)(uintptr_t)shareableHandle, IPC_HANDLE_TYPE),
                  "cuMemImportFromShareableHandle");
    printf("Client: Imported memory handle successfully\n");
    CUdeviceptr clientBasePtr = 0;
    checkCudaErrors(cuMemAddressReserve(&clientBasePtr, allocSize, 0, 0, 0), "cuMemAddressReserve (client)");
    checkCudaErrors(cuMemMap(clientBasePtr, allocSize, 0, importedHandle, 0), "cuMemMap (client)");

    CUmemAccessDesc clientAccessDesc = {};
    clientAccessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    clientAccessDesc.location.id = 0;
    clientAccessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkCudaErrors(cuMemSetAccess(clientBasePtr, allocSize, &clientAccessDesc, 1), "cuMemSetAccess (client)");
    printf("Client: Mapped imported memory to virtual address %p\n", (void*)clientBasePtr);

    double clientData[2] = {0.0, 0.0};
    checkCudaErrors(cuMemcpyDtoH(clientData, clientBasePtr, 2 * sizeof(double)), "cuMemcpyDtoH (client)");
    printf("Client: Read data: %.4f, %.4f\n", clientData[0], clientData[1]);

    checkCudaErrors(cuMemUnmap(basePtr, allocSize), "cuMemUnmap (server)");
    checkCudaErrors(cuMemAddressFree(basePtr, allocSize), "cuMemAddressFree (server)");

    checkCudaErrors(cuMemUnmap(clientBasePtr, allocSize), "cuMemUnmap (client)");
    checkCudaErrors(cuMemAddressFree(clientBasePtr, allocSize), "cuMemAddressFree (client)");

    checkCudaErrors(cuMemRelease(memHandle), "cuMemRelease");

    checkCudaErrors(cuCtxDestroy(cuContext), "cuCtxDestroy");

    printf("Program completed successfully.\n");
    return 0;
}
