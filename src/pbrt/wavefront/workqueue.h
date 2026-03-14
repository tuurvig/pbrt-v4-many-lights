// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_WAVEFRONT_WORKQUEUE_H
#define PBRT_WAVEFRONT_WORKQUEUE_H

#include <pbrt/pbrt.h>

#include <pbrt/options.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>

#include <atomic>
#include <utility>

namespace pbrt {

// WorkQueue Definition
template <typename WorkItem>
class WorkQueue : public SOA<WorkItem> {
  public:
    // WorkQueue Public Methods
    WorkQueue() = default;
    WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
    WorkQueue &operator=(const WorkQueue &w) {
        SOA<WorkItem>::operator=(w);
        size.Store(w.size.Load());
        return *this;
    }

    PBRT_CPU_GPU
    int Size() const {
        return size.Load();
    }

    PBRT_CPU_GPU
    void Reset() {
        size.Store(0);
    }

    PBRT_CPU_GPU
    int Push(WorkItem w) {
        int index = AllocateEntry();
        (*this)[index] = w;
        return index;
    }

    PBRT_CPU_GPU
    int ReserveEntries(int count) {
        if (count < 1) {
            return -1;
        }

        return size.FetchAdd(count);
    }

  protected:
    // WorkQueue Protected Methods
    PBRT_CPU_GPU
    int AllocateEntry() {
        return size.FetchAdd(1);
    }

  private:
    // WorkQueue Private Members
    AtomicInt<int> size;
};

// WorkQueue Inline Functions
template <typename F, typename WorkItem>
void ForAllQueued(const char *desc, ProfilerKernelGroup group, const WorkQueue<WorkItem> *q, int maxQueued,
                  F &&func) {
    if (Options->useGPU) {
        // Launch GPU threads to process _q_ using _func_
#ifdef PBRT_BUILD_GPU_RENDERER
        GPUParallelFor(desc, group, maxQueued,
                       [=] PBRT_GPU(int index) mutable {
            if (index >= q->Size())
                return;
            func((*q)[index]);
        });
#else
        LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif

    } else {
        // Process _q_ using _func_ with CPU threads
        ParallelFor(0, q->Size(), [&](int index) { func((*q)[index]); });
    }
}

// MultiWorkQueue Definition
template <typename T>
class MultiWorkQueue;

template <typename... Ts>
class MultiWorkQueue<TypePack<Ts...>> {
  public:
    // MultiWorkQueue Public Methods
    template <typename T>
    PBRT_CPU_GPU WorkQueue<T> *Get() {
        return &pstd::get<WorkQueue<T>>(queues);
    }

    MultiWorkQueue(int n, Allocator alloc, pstd::span<const bool> haveType) {
        int index = 0;
        ((*Get<Ts>() = WorkQueue<Ts>(haveType[index++] ? n : 1, alloc)), ...);
    }

    template <typename T>
    PBRT_CPU_GPU int Size() const {
        return Get<T>()->Size();
    }
    template <typename T>
    PBRT_CPU_GPU int Push(const T &value) {
        return Get<T>()->Push(value);
    }

    PBRT_CPU_GPU
    void Reset() { (Get<Ts>()->Reset(), ...); }

  private:
    // MultiWorkQueue Private Members
    pstd::tuple<WorkQueue<Ts>...> queues;
};

}  // namespace pbrt

#endif  // PBRT_WAVEFRONT_WORKQUEUE_H
