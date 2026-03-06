// heap.h is Copyright(c) 2026 Richard Kvasnica
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_HEAP_H
#define PBRT_UTIL_HEAP_H

#include <pbrt/pbrt.h>

namespace pbrt {

template<typename Key, typename Data>
PBRT_CPU_GPU void HeapBubbleDown(Key* keys, Data* data, int size) {
    int currIndex = 0;
    const auto keyVal = keys[0];
    const auto dataVal = data[0];

    while (true) {
        const int leftIndex = (currIndex << 1) + 1;
        if (leftIndex >= size) break;

        const int rightIndex = leftIndex + 1;
        int childIndex = rightIndex < size && keys[rightIndex] > keys[leftIndex] ? rightIndex : leftIndex;

        const auto childKey = keys[childIndex];
        if (childKey <= keyVal) break;

        keys[currIndex] = childKey;
        data[currIndex] = data[childIndex];

        currIndex = childIndex;
    }

    keys[currIndex] = keyVal;
    data[currIndex] = dataVal;
}

template<typename Key, typename Data>
PBRT_CPU_GPU void HeapBubbleUp(Key* keys, Data* data, int size) {
    int currIndex = size - 1;

    const auto keyVal = keys[currIndex];
    const auto dataVal = data[currIndex];

    while (currIndex > 0) {
        const int parentIndex = (currIndex - 1) >> 1;
        const auto parentKey = keys[parentIndex];
        if (keyVal <= parentKey) break;

        keys[currIndex] = parentKey;
        data[currIndex] = data[parentIndex];

        currIndex = parentIndex;
    }

    keys[currIndex] = keyVal;
    data[currIndex] = dataVal;
}

}

#endif