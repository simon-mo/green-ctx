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
        fprintf(stderr, "Client Error: %s: %s\n", msg, errStr ? errStr : "Unknown error");
        exit(EXIT_FAILURE);
    }
}

int recv_fd(int socket) {
    struct msghdr msg = {0};
    char m_buffer[256];
    struct iovec io = { .iov_base = m_buffer, .iov_len = sizeof(m_buffer) };
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    
    char c_buffer[CMSG_SPACE(sizeof(int))];
    memset(c_buffer, 0, sizeof(c_buffer));
    msg.msg_control = c_buffer;
    msg.msg_controllen = sizeof(c_buffer);
    
    if (recvmsg(socket, &msg, 0) < 0) {
        perror("Client: recvmsg");
        return -1;
    }
    
    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == NULL) {
        fprintf(stderr, "Client: No passed fd\n");
        return -1;
    }
    
    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
    return fd;
}

int main() {
    checkCudaErrors(cuInit(0), "cuInit");
    CUdevice cuDevice;
    checkCudaErrors(cuDeviceGet(&cuDevice, 0), "cuDeviceGet");
    CUcontext cuContext;
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice), "cuCtxCreate");
    
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Client: socket");
        exit(EXIT_FAILURE);
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path)-1);
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Client: connect");
        exit(EXIT_FAILURE);
    }
    printf("Client: Connected to server socket\n");
    
    int received_fd = recv_fd(sock);
    if (received_fd < 0) {
        fprintf(stderr, "Client: Failed to receive FD\n");
        exit(EXIT_FAILURE);
    }
    printf("Client: Received FD: %d\n", received_fd);
    
    close(sock);
    
    CUmemGenericAllocationHandle importedHandle;
    checkCudaErrors(cuMemImportFromShareableHandle(&importedHandle,
                  (void*)(uintptr_t)received_fd, IPC_HANDLE_TYPE),
                  "cuMemImportFromShareableHandle");
    printf("Client: Imported memory handle successfully\n");
    
    size_t allocSize = 10 * 1024 * 1024; 
    CUdeviceptr basePtr = 0;
    checkCudaErrors(cuMemAddressReserve(&basePtr, allocSize, 0, 0, 0), "cuMemAddressReserve");
    checkCudaErrors(cuMemMap(basePtr, allocSize, 0, importedHandle, 0), "cuMemMap");
    
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkCudaErrors(cuMemSetAccess(basePtr, allocSize, &accessDesc, 1), "cuMemSetAccess");
    printf("Client: Mapped imported memory to virtual address %p\n", (void*)basePtr);
    
    double data[2] = {0.0, 0.0};
    checkCudaErrors(cuMemcpyDtoH(data, basePtr, 2 * sizeof(double)), "cuMemcpyDtoH");
    printf("Client: Read initial data: %.4f, %.4f\n", data[0], data[1]);
    
    double newData[2] = { data[0] + 1.0, data[1] + 1.0 };
    checkCudaErrors(cuMemcpyHtoD(basePtr, newData, 2 * sizeof(double)), "cuMemcpyHtoD");
    printf("Client: Wrote new data: %.4f, %.4f\n", newData[0], newData[1]);
    
    double verifyData[2] = {0.0, 0.0};
    checkCudaErrors(cuMemcpyDtoH(verifyData, basePtr, 2 * sizeof(double)), "cuMemcpyDtoH");
    printf("Client: Verified data: %.4f, %.4f\n", verifyData[0], verifyData[1]);
    
    checkCudaErrors(cuMemUnmap(basePtr, allocSize), "cuMemUnmap");
    checkCudaErrors(cuMemAddressFree(basePtr, allocSize), "cuMemAddressFree");
    checkCudaErrors(cuCtxDestroy(cuContext), "cuCtxDestroy");
    
    printf("Client: Cleanup complete, exiting.\n");
    return 0;
}
