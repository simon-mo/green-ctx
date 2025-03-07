#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <cuda.h>
#include <errno.h>

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
        fprintf(stderr, "Client: No passed FD\n");
        return -1;
    }
    
    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
    return fd;
}

void clientProcessWork(int procIndex, int sharedFd) {
    printf("Child process %d: Starting\n", procIndex);
    
    checkCudaErrors(cuInit(0), "cuInit");
    CUdevice cuDevice;
    checkCudaErrors(cuDeviceGet(&cuDevice, 0), "cuDeviceGet");
    CUcontext cuContext;
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice), "cuCtxCreate");
    
    CUmemGenericAllocationHandle importedHandle;
    checkCudaErrors(cuMemImportFromShareableHandle(&importedHandle,
                  (void*)(uintptr_t)sharedFd, IPC_HANDLE_TYPE),
                  "cuMemImportFromShareableHandle");
    printf("Child process %d: Imported memory handle successfully\n", procIndex);
    
    // size_t allocSize = 10 * 1024 * 1024; // 10 MB
    size_t allocSize = 15ULL * 1024ULL * 1024ULL * 1024ULL;
    CUdeviceptr basePtr = 0;
    checkCudaErrors(cuMemAddressReserve(&basePtr, allocSize, 0, 0, 0),
                    "cuMemAddressReserve");
    checkCudaErrors(cuMemMap(basePtr, allocSize, 0, importedHandle, 0),
                    "cuMemMap");
    
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkCudaErrors(cuMemSetAccess(basePtr, allocSize, &accessDesc, 1),
                    "cuMemSetAccess");
    printf("Child process %d: Mapped imported memory to virtual address %p\n",
           procIndex, (void*)basePtr);
    
    double data[2] = {0.0, 0.0};
    checkCudaErrors(cuMemcpyDtoH(data, basePtr, 2 * sizeof(double)),
                    "cuMemcpyDtoH");
    printf("Child process %d: Read data: %.4f, %.4f\n", procIndex, data[0], data[1]);
    
    double newData[2] = { data[0] + procIndex, data[1] + procIndex };
    checkCudaErrors(cuMemcpyHtoD(basePtr, newData, 2 * sizeof(double)),
                    "cuMemcpyHtoD");
    printf("Child process %d: Wrote new data: %.4f, %.4f\n", procIndex, newData[0], newData[1]);
    
    double verifyData[2] = {0.0, 0.0};
    checkCudaErrors(cuMemcpyDtoH(verifyData, basePtr, 2 * sizeof(double)),
                    "cuMemcpyDtoH verify");
    printf("Child process %d: Verified data: %.4f, %.4f\n", procIndex, verifyData[0], verifyData[1]);
    
    checkCudaErrors(cuMemUnmap(basePtr, allocSize), "cuMemUnmap");
    checkCudaErrors(cuMemAddressFree(basePtr, allocSize), "cuMemAddressFree");
    checkCudaErrors(cuCtxDestroy(cuContext), "cuCtxDestroy");
    
    printf("Child process %d: Exiting\n", procIndex);
    exit(EXIT_SUCCESS);
}

int main() {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Client (parent): socket");
        exit(EXIT_FAILURE);
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path)-1);
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Client (parent): connect");
        exit(EXIT_FAILURE);
    }
    printf("Client (parent): Connected to server socket\n");
    
    int sharedFd = recv_fd(sock);
    if (sharedFd < 0) {
        fprintf(stderr, "Client (parent): Failed to receive FD\n");
        exit(EXIT_FAILURE);
    }
    printf("Client (parent): Received shared FD: %d\n", sharedFd);
    close(sock);
    
    int numClients = 3;
    pid_t pids[numClients];
    for (int i = 0; i < numClients; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("Client (parent): fork");
            exit(EXIT_FAILURE);
        }
        if (pid == 0) {
            int dupFd = dup(sharedFd);
            if (dupFd < 0) {
                perror("Client (child): dup");
                exit(EXIT_FAILURE);
            }
            clientProcessWork(i, dupFd);
        } else {
            pids[i] = pid;
        }
    }
    
    for (int i = 0; i < numClients; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        printf("Client (parent): Child %d exited with status %d\n", i, status);
    }
    
    close(sharedFd); 
    printf("Client (parent): All child processes completed.\n");
    return 0;
}
