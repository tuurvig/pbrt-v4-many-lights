// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_UTIL_H
#define PBRT_GPU_UTIL_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/progressreporter.h>

#include <map>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef NVTX
#ifdef UNICODE
#undef UNICODE
#endif
#include <nvtx3/nvToolsExt.h>

#ifdef RGB
#undef RGB
#endif  // RGB
#endif

#define CUDA_CHECK(EXPR)                                        \
    if (EXPR != cudaSuccess) {                                  \
        cudaError_t error = cudaGetLastError();                 \
        LOG_FATAL("CUDA error: %s", cudaGetErrorString(error)); \
    } else /* eat semicolon */

#define CU_CHECK(EXPR)                                              \
    do {                                                            \
        CUresult result = EXPR;                                     \
        if (result != CUDA_SUCCESS) {                               \
            const char *str;                                        \
            CHECK_EQ(CUDA_SUCCESS, cuGetErrorString(result, &str)); \
            LOG_FATAL("CUDA error: %s", str);                       \
        }                                                           \
    } while (false) /* eat semicolon */

namespace pbrt {

enum ProfilerKernelGroup {
    WAVEFRONT, HPLOC, END
};

std::pair<cudaEvent_t, cudaEvent_t> GetProfilerEvents(const char *description, ProfilerKernelGroup group);

void ReportKernelStats(ProfilerKernelGroup group);

// Neat timer wrapper class to used RAII to measure time for kernel execution time.
class KernelTimerWrapper{
public:
    KernelTimerWrapper(cudaEvent_t start, cudaEvent_t end) : m_start(start), m_end(end) { cudaEventRecord(m_start); }
    KernelTimerWrapper(const std::pair<cudaEvent_t, cudaEvent_t>& e) : KernelTimerWrapper(e.first, e.second) {}
    ~KernelTimerWrapper() { cudaEventRecord(m_end); }
private:
    cudaEvent_t m_start, m_end;
};

// GPU Synchronization Function Declarations
void GPUInit();
void GPUThreadInit();

static void GPUWait() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void GPUWait(cudaEvent_t event) {
    CUDA_CHECK(cudaEventSynchronize(event));
}

static void GPUWait(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename F>
inline int GetBlockSize(const char *description, F kernel) {
    // Note: this isn't reentrant, but that's fine for our purposes...
    static std::map<std::type_index, int> kernelBlockSizes;

    std::type_index index = std::type_index(typeid(F));

    auto iter = kernelBlockSizes.find(index);
    if (iter != kernelBlockSizes.end())
        return iter->second;

    int minGridSize, blockSize;
    CUDA_CHECK(
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
    kernelBlockSizes[index] = blockSize;
    LOG_VERBOSE("[%s]: block size %d", description, blockSize);

    return blockSize;
}

#ifdef __NVCC__
template <typename F>
__global__ void Kernel(F func, int nItems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;

    func(tid);
}

// GPU Launch Function Declarations
template <typename F>
void GPUParallelFor(const char *description, ProfilerKernelGroup group, int nItems, F func);

template <typename F>
void GPUParallelFor(const char *description, ProfilerKernelGroup group, int nItems,
                    F func) {
#ifdef NVTX
    nvtxRangePush(description);
#endif
    auto kernel = &Kernel<F>;
    int blockSize = GetBlockSize(description, kernel);

#ifdef PBRT_DEBUG_BUILD
    LOG_VERBOSE("Launching %s", description);
#endif
    {
        KernelTimerWrapper timer(GetProfilerEvents(description, group));
        int gridSize = (nItems + blockSize - 1) / blockSize;
        kernel<<<gridSize, blockSize>>>(func, nItems);
    }

#ifdef PBRT_DEBUG_BUILD
    GPUWait();
    LOG_VERBOSE("Post-sync %s", description);
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

#endif  // __NVCC__

// GPU Allocation function definition
template<typename T>
static T* GPUAllocate(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * count));
    return ptr;
}

template<typename T>
static T* GPUAllocAsync(size_t count, cudaStream_t stream = 0) {
    T* ptr;
    CUDA_CHECK(cudaMallocAsync((void**)&ptr, sizeof(T) * count, 0));
    return ptr;
}

template<typename T>
static T* GPUAllocHostAsync(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMallocHost((void**)&ptr, sizeof(T) * count));
    return ptr;
}

template<typename T>
static T* GPUAllocUnified(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMallocManaged((void**)&ptr, sizeof(T) * count));
    return ptr;
}

template<typename T>
static void GPUCopyToDevice(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy((void*)dst, (const void*)src, sizeof(T) * count, cudaMemcpyHostToDevice));
}

template<typename T>
static void GPUCopyAsyncToDevice(T* dst, const T* src, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync((void*)dst, (const void*)src, sizeof(T) * count, cudaMemcpyHostToDevice, stream));
}

template<typename T>
static void GPUCopyToHost(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy((void*)dst, (const void*)src, sizeof(T) * count, cudaMemcpyDeviceToHost));
}

template<typename T>
static void GPUCopyAsyncToHost(T* dst, const T* src, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync((void*)dst, (const void*)src, sizeof(T) * count, cudaMemcpyDeviceToHost, stream));
}

static void GPUMemset(void *ptr, int32_t byte, size_t bytes) {
    CUDA_CHECK(cudaMemset(ptr, byte, bytes));
}

static void GPUMemsetAsync(void* ptr, int32_t byte, size_t bytes)
{
    CUDA_CHECK(cudaMemsetAsync(ptr, byte, bytes));
}

static void GPUFree(void* ptr)
{
    CUDA_CHECK(cudaFree(ptr));
}

static void GPUFreeAsync(void* ptr)
{
    CUDA_CHECK(cudaFreeAsync(ptr, 0));
}

void GPURegisterThread(const char *name);
void GPUNameStream(cudaStream_t stream, const char *name);

struct BufferGPU {
    void Init(size_t size) {
        // GPU-side memory for sample points
        ptr = (CUdeviceptr)GPUAllocate<uint8_t>(size);

        // Event to keep track of when the buffer has been processed on the GPU.
        CUDA_CHECK(cudaEventCreate(&finishedEvent));

        // Host-side staging buffer for async memcpy in pinned host memory.
        hostPtr = GPUAllocHostAsync<uint8_t>(size);
    }

    bool used = false;
    cudaEvent_t finishedEvent;
    CUdeviceptr ptr = 0;
    void *hostPtr = nullptr;
};

}  // namespace pbrt

#endif  // PBRT_GPU_UTIL_H
