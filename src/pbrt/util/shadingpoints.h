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

/// @brief Compact descriptor of a shading sample.
/// Stores world-space position and a quantized normal direction in
/// `UniformDiskVector` form so spatial and directional clustering can share one
/// fixed-size record.
struct alignas(16) ShadingPoint {
    ShadingPoint() = default;

    PBRT_CPU_GPU
    ShadingPoint(const Point3f& p, const Normal3f& n) :
        point(p), dir(Vector3f(n)) {}

    Point3f point;
    UniformDiskVector dir;
};

STAT_MEMORY_COUNTER("Memory/Shading Point Collector", shadingPointCollectorBytes);

/// @brief Fixed-capacity append-only container for first-wave shading points.
class ShadingPointCollector {
  public:
    ShadingPointCollector() = default;

    /// @brief Preallocates storage for one render wave.
    /// @param pixelBounds Rendered pixel bounds used to estimate sample count.
    /// @param maxDepth Integrator path-depth bound used for estimating the maximum amount of posible samples.
    /// @param alloc Pbrt allocator.
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

    /// @brief Appends one shading-point sample.
    /// Uses an atomic index so concurrent producers can append safely.
    /// @param p Shading position.
    /// @param ns Shading normal.
    PBRT_CPU_GPU
    void Append(Point3f p, Normal3f ns) {
        uint32_t index = AllocateEntry();
        DCHECK_LT(index, capacity);
        points[index] = ShadingPoint(p, ns);
    }

    /// @brief Returns the current number of appended points.
    PBRT_CPU_GPU
    uint32_t Size() const {
        return size.Load();
    }

    /// @brief Returns the preallocated maximum number of points.
    PBRT_CPU_GPU
    uint32_t Capacity() const { return capacity; }

    ShadingPoint *Data() { return points; }
    const ShadingPoint *Data() const { return points; }

    pstd::span<const ShadingPoint> Points() const { return {points, Size()}; }
    pstd::span<ShadingPoint> Points() { return {points, Size()}; }
protected:
    /// @brief Reserves one slot and returns its index.
    /// @return Index of the newly reserved entry.
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
