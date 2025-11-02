// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/memory.h>

#include <pbrt/gpu/util.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace pbrt {

void *CUDAMemoryResource::do_allocate(size_t size, size_t alignment) {
    uint8_t* ptr = GPUAllocUnified<uint8_t>(size);
    CHECK_EQ(0, intptr_t(ptr) % alignment);
    return ptr;
}

void CUDAMemoryResource::do_deallocate(void *p, size_t bytes, size_t alignment) {
    GPUFree(p);
}

void *CUDATrackedMemoryResource::do_allocate(size_t size, size_t alignment) {
    if (size == 0)
        return nullptr;

    uint8_t *ptr = GPUAllocUnified<uint8_t>(size);
    DCHECK_EQ(0, intptr_t(ptr) % alignment);

    std::lock_guard<std::mutex> lock(mutex);
    allocations[ptr] = size;
    bytesAllocated += size;

    return ptr;
}

void CUDATrackedMemoryResource::do_deallocate(void *p, size_t size, size_t alignment) {
    if (!p)
        return;

    GPUFree(p);

    std::lock_guard<std::mutex> lock(mutex);
    auto iter = allocations.find(p);
    DCHECK(iter != allocations.end());
    allocations.erase(iter);
    bytesAllocated -= size;
}

void CUDATrackedMemoryResource::PrefetchToGPU() const {
    int deviceIndex;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));

    std::lock_guard<std::mutex> lock(mutex);

    LOG_VERBOSE("Prefetching %d allocations to GPU memory", allocations.size());
    size_t bytes = 0;
    for (auto iter : allocations) {
        CUDA_CHECK(
            cudaMemPrefetchAsync(iter.first, iter.second, deviceIndex, 0 /* stream */));
        bytes += iter.second;
    }
    GPUWait();
    LOG_VERBOSE("Done prefetching: %d bytes total", bytes);
}

CUDATrackedMemoryResource CUDATrackedMemoryResource::singleton;

}  // namespace pbrt
