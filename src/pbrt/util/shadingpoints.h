// Copyright(c) 2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
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

namespace pbrt {

struct alignas(16) ShadingPoint {
    ShadingPoint() = default;

    PBRT_CPU_GPU
    ShadingPoint(const Point3f& p, const Normal3f& n) :
        point(p), dir(Vector3f(n)) {}

    Point3f point;
    UniformDiskVector dir;
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
        size.Store(w.size.Load());
        return *this;
    }

    PBRT_CPU_GPU
    void Append(Point3f p, Normal3f ns) {
        uint32_t index = AllocateEntry();
        DCHECK_LT(index, capacity);
        points[index] = ShadingPoint(p, ns);
    }

    PBRT_CPU_GPU
    uint32_t Size() const {
        return size.Load();
    }

    PBRT_CPU_GPU
    uint32_t Capacity() const { return capacity; }

    ShadingPoint *Data() { return points; }
    const ShadingPoint *Data() const { return points; }

    pstd::span<const ShadingPoint> Points() const { return {points, Size()}; }
    pstd::span<ShadingPoint> Points() { return {points, Size()}; }
protected:
    PBRT_CPU_GPU
    uint32_t AllocateEntry() {
        return size.FetchAdd(1);
    }

private:
    Allocator alloc;
    ShadingPoint* points = nullptr;
    uint32_t capacity = 0;
    
    pbrt::AtomicInt<uint32_t> size;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_SHADINGPOINTS_H
