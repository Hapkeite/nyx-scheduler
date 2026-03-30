#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// Function pointers to hold the REAL CUDA functions
static cudaError_t (*real_cudaMalloc)(void **, size_t) = NULL;
static cudaError_t (*real_cudaFree)(void *) = NULL;

// Helper function to send JSON to the Rust scheduler and wait for a reply
void send_to_scheduler(const char *json_msg) {
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0) return; // If socket creation fails, just silently continue

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, "/tmp/nyx.sock", sizeof(addr.sun_path) - 1);

  // Try to connect to the Rust scheduling daemon
  if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
    
    // 1. Send the request (e.g., {"action": "malloc", "bytes": 16000000})
    send(sock, json_msg, strlen(json_msg), 0);

    // 2. Block and wait for the scheduler's response (e.g., {"status": "Go"})
    char buffer[256];
    int n = recv(sock, buffer, sizeof(buffer) - 1, 0);
    
    if (n > 0) {
        buffer[n] = '\0';
        // The interceptor successfully waited for the Rust daemon.
        // In the future, you can parse this buffer to handle "Wait" commands
        // and loop until "Go" is received.
    }
  }
  
  // 3. Close only after the handshake is complete
  close(sock);
}

// 1. Intercept cudaMalloc
extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
  if (!real_cudaMalloc) {
    real_cudaMalloc =
        (cudaError_t(*)(void **, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
  }

  // Format our message as a JSON string
  char msg[256];
  snprintf(msg, sizeof(msg), "{\"action\": \"malloc\", \"bytes\": %zu}", size);

  // Send to the background scheduler and wait for permission
  send_to_scheduler(msg);

  // Execute the actual GPU allocation
  return real_cudaMalloc(devPtr, size);
}

// 2. Intercept cudaFree
extern "C" cudaError_t cudaFree(void *devPtr) {
  if (!real_cudaFree) {
    real_cudaFree = (cudaError_t(*)(void *))dlsym(RTLD_NEXT, "cudaFree");
  }

  // Format the free message
  char msg[256];
  snprintf(msg, sizeof(msg), "{\"action\": \"free\", \"ptr\": \"%p\"}", devPtr);

  // Send to the background scheduler so it can update its VRAM ledger
  send_to_scheduler(msg);

  // Execute the actual GPU deallocation
  return real_cudaFree(devPtr);
}