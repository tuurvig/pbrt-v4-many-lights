// shadingpoints.h is Copyright(c) 2026 Richard Kvasnica.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_SHADINGPOINTS_H
#define PBRT_UTIL_SHADINGPOINTS_H

#include <pbrt/pbrt.h>
#include <pbrt/util/check.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/stats.h>
#include <atomic>

#ifdef __CUDACC__
#ifdef PBRT_IS_WINDOWS
#if (__CUDA_ARCH__ < 700)
#ifndef PBRT_USE_LEGACY_CUDA_ATOMICS
#define PBRT_USE_LEGACY_CUDA_ATOMICS
#endif
#endif
#else
#if (__CUDA_ARCH__ < 600)
#ifndef PBRT_USE_LEGACY_CUDA_ATOMICS
#define PBRT_USE_LEGACY_CUDA_ATOMICS
#endif
#endif
#endif  // PBRT_IS_WINDOWS

#ifndef PBRT_USE_LEGACY_CUDA_ATOMICS
#include <cuda/atomic>
#endif
#endif  // __CUDACC__

namespace pbrt {

struct ShadingPoint {
    Point3f p;
    Normal3f n;
};

STAT_MEMORY_COUNTER("Memory/Shading Point Collector", shadingPointCollectorBytes);

class ShadingPointCollector {
  public:
    ShadingPointCollector() = default;

    explicit ShadingPointCollector(const Bounds2i &pixelBounds, int maxDepth, Allocator alloc)
        : alloc(alloc), points(nullptr), capacity(0) {
        DCHECK_LT(0, pixelBounds.Area());
        const int pixelCount = pixelBounds.Area();
        capacity = pixelCount * maxDepth;
        points = alloc.allocate_object<ShadingPoint>(capacity);
        shadingPointCollectorBytes += capacity * sizeof(ShadingPoint);
    }

    ~ShadingPointCollector() {
        if (points) {
            alloc.deallocate_object(points, capacity);
        }
    }

    ShadingPointCollector &operator=(const ShadingPointCollector &w) {
        points = w.points;
        capacity = w.capacity;
#if defined(PBRT_IS_GPU_CODE) && defined(PBRT_USE_LEGACY_CUDA_ATOMICS)
        size = w.size;
#else
        size.store(w.size.load());
#endif
        return *this;
    }

    PBRT_CPU_GPU
    void Append(Point3f p, Normal3f ns) {
        uint32_t index = AllocateEntry();
        DCHECK_LT(index, capacity);
        points[index] = ShadingPoint{p, ns};
    }

    PBRT_CPU_GPU
    uint32_t Size() const {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        return size;
#else
        return size.load(cuda::std::memory_order_relaxed);
#endif
#else
        return size.load(std::memory_order_relaxed);
#endif
    }

    PBRT_CPU_GPU
    uint32_t Capacity() const { return capacity; }

    pstd::span<const ShadingPoint> Points() const { return {points, Size()}; }
    pstd::span<ShadingPoint> Points() { return {points, Size()}; }
protected:
    PBRT_CPU_GPU
    uint32_t AllocateEntry() {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        return atomicAdd(&size, 1ull);
#else
        return size.fetch_add(1, cuda::std::memory_order_relaxed);
#endif
#else
        return size.fetch_add(1, std::memory_order_relaxed);
#endif
    }

private:
    Allocator alloc;
    ShadingPoint* points = nullptr;
    uint32_t capacity = 0;
    
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
    uint32_t size = 0;
#else
    cuda::atomic<uint32_t, cuda::thread_scope_device> size{0};
#endif
#else
    std::atomic<uint32_t> size{0};
#endif  // PBRT_IS_GPU_CODE
};

}  // namespace pbrt

#endif  // PBRT_UTIL_SHADINGPOINTS_H
