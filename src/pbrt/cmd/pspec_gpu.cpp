// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// pspec_gpu.cpp

// Unfortunately, cmake doesn't seem to like building a CUDA executable out
// of a single-cpp file and thus, the code that does the GPU launches is
// separate here.

#ifdef PBRT_BUILD_GPU_RENDERER

#include <pbrt/pbrt.h>

#include <pbrt/gpu/util.h>
#include <pbrt/util/image.h>
#include <pbrt/util/vecmath.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>

namespace pbrt {

// Ring-buffer of Buffers that hold sets of sample Points
static std::vector<BufferGPU> bufferPool;
static int nextBufferOffset;

void UPSInit(int nPoints) {
    bufferPool.resize(16);  // should be plenty
    for (BufferGPU &b : bufferPool) {
        b.Init(nPoints * sizeof(Point2f));
    }
}

void UpdatePowerSpectrum(const std::vector<Point2f> &points, Image *pspec) {
    BufferGPU &b = bufferPool[nextBufferOffset];
    if (++nextBufferOffset == bufferPool.size())
        nextBufferOffset = 0;
    if (!b.used)
        b.used = true;
    else
        // If it's been used previously, make sure that the kernel that
        // consumed it has completed.
        GPUWait(b.finishedEvent);

    // Copy the sample points to host-side pinned memory
    memcpy(b.hostPtr, points.data(), points.size() * sizeof(Point2f));
    GPUCopyAsyncToDevice<Point2f>(reinterpret_cast<Point2f*>(b.ptr), reinterpret_cast<Point2f*>(b.hostPtr), points.size());

    int nPoints = points.size();

    GPUParallelFor("Fourier transform", ProfilerKernelGroup::WAVEFRONT, pspec->Resolution().x * pspec->Resolution().y,
                   [=] PBRT_GPU(int tid) {
                       int res = pspec->Resolution().x;
                       Point2i p(tid % res, tid / res);
                       Point2f uv(0, 0);
                       Float wx = p.x - res / 2, wy = p.y - res / 2;

                       const Point2f *pts = (const Point2f *)b.ptr;
                       for (int i = 0; i < nPoints; ++i) {
                           float exp = -2 * Pi * (wx * pts[i][0] + wy * pts[i][1]);
                           uv[0] += std::cos(exp);
                           uv[1] += std::sin(exp);
                       }

                       // Update power spectrum
                       pspec->SetChannel(
                           p, 0, pspec->GetChannel(p, 0) + Sqr(uv[0]) + Sqr(uv[1]));
                   });

    // Indicate that the buffer has been consumed and is safe for reuse.
    CUDA_CHECK(cudaEventRecord(b.finishedEvent));
}

}  // namespace pbrt

#endif  //  PBRT_BUILD_GPU_RENDERER
