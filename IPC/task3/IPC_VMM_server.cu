#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <cuda.h>

#define SOCKET_PATH "/tmp/cuda_ipc_socket"
#define IPC_HANDLE_TYPE CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

void checkCudaErrors(CUresult err, const char* msg) {
    if(err != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "Server Error: %s: %s\n", msg, errStr ? errStr : "Unknown error");
        exit(EXIT_FAILURE);
    }
}

int send_fd(int socket, int fd_to_send) {
    struct msghdr msg = {0};
    char buf[CMSG_SPACE(sizeof(fd_to_send))];
    memset(buf, 0, sizeof(buf));
    
    struct iovec io = { .iov_base = (void*)"ABC", .iov_len = 3 };
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);
    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type  = SCM_RIGHTS;
    cmsg->cmsg_len   = CMSG_LEN(sizeof(fd_to_send));
    memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(fd_to_send));
    msg.msg_controllen = cmsg->cmsg_len;
    
    if (sendmsg(socket, &msg, 0) < 0) {
        perror("Server: sendmsg");
        return -1;
    }
    return 0;
}

int main() {
    checkCudaErrors(cuInit(0), "cuInit");
    CUdevice cuDevice;
    checkCudaErrors(cuDeviceGet(&cuDevice, 0), "cuDeviceGet");
    CUcontext cuContext;
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice), "cuCtxCreate");
    
    size_t allocSize = 15ULL * 1024ULL * 1024ULL * 1024ULL;
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
    printf("Server: Allocated physical memory of size %zu bytes\n", allocSize);

    CUdeviceptr basePtr = 0;
    checkCudaErrors(cuMemAddressReserve(&basePtr, allocSize, 0, 0, 0), "cuMemAddressReserve");
    checkCudaErrors(cuMemMap(basePtr, allocSize, 0, memHandle, 0), "cuMemMap");
    
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkCudaErrors(cuMemSetAccess(basePtr, allocSize, &accessDesc, 1), "cuMemSetAccess");
    printf("Server: Mapped memory to virtual address %p\n", (void*)basePtr);

    double initData[2] = {3.1415, 2.71828};
    checkCudaErrors(cuMemcpyHtoD(basePtr, initData, 2 * sizeof(double)), "cuMemcpyHtoD");
    printf("Server: Wrote initial data: %.4f, %.4f\n", initData[0], initData[1]);
    

    int shareableFd;
    checkCudaErrors(cuMemExportToShareableHandle(&shareableFd, memHandle, IPC_HANDLE_TYPE, 0),
                     "cuMemExportToShareableHandle"); 
    printf("Server: Exported shareable handle (fd): %d\n", shareableFd);
    
    int server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock < 0) {
        perror("Server: socket");
        exit(EXIT_FAILURE);
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    unlink(SOCKET_PATH);
    if (bind(server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Server: bind");
        exit(EXIT_FAILURE);
    }
    if (listen(server_sock, 1) < 0) {
        perror("Server: listen");
        exit(EXIT_FAILURE);
    }
    
    printf("Server: Waiting for client connection...\n");
    int client_sock = accept(server_sock, NULL, NULL);
    if (client_sock < 0) {
        perror("Server: accept");
        exit(EXIT_FAILURE);
    }
    printf("Server: Client connected, sending FD...\n");
    

    if (send_fd(client_sock, shareableFd) < 0) {
        fprintf(stderr, "Server: Failed to send FD\n");
        exit(EXIT_FAILURE);
    }
    printf("Server: FD sent successfully.\n");
    
    close(client_sock);
    close(server_sock);
    sleep(10);
    
    checkCudaErrors(cuMemUnmap(basePtr, allocSize), "cuMemUnmap");
    checkCudaErrors(cuMemAddressFree(basePtr, allocSize), "cuMemAddressFree");
    checkCudaErrors(cuMemRelease(memHandle), "cuMemRelease");
    checkCudaErrors(cuCtxDestroy(cuContext), "cuCtxDestroy");
    
    printf("Server: Cleanup complete, exiting.\n");
    return 0;
}
