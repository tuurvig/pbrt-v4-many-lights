// Copyright(c) 2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_HEAP_H
#define PBRT_UTIL_HEAP_H

#include <pbrt/pbrt.h>

namespace pbrt {

/// @brief Restores the max-heap property by moving the root element down the tree.
/// Operating on pre-allocated parallel arrays (`keys` and `data`) to achieve
/// GPU execution without dynamic allocation.
/// 
/// @tparam Key The data type of the sorting keys
/// @tparam Data The data type of the associated payload
/// 
/// @param keys Pointer to the array of keys representing the max-heap.
/// @param data Pointer to the array of data payloads parallel to the `keys` array.
/// @param size The current number of active elements in the heap.
template<typename Key, typename Data>
PBRT_CPU_GPU void HeapBubbleDown(Key* keys, Data* data, int size) {
    int currIndex = 0;
    const auto keyVal = keys[0];
    const auto dataVal = data[0];

    while (true) {
        // Left child index is 2*i + 1
        const int leftIndex = (currIndex << 1) + 1;
        if (leftIndex >= size) break;

        const int rightIndex = leftIndex + 1;

        // Find the index of the largest child
        int childIndex = rightIndex < size && keys[rightIndex] > keys[leftIndex] ? rightIndex : leftIndex;

        const auto childKey = keys[childIndex];

        // If the current node is larger than the largest child, the heap property is satisfied
        if (childKey <= keyVal) break;

        // Move the child up
        keys[currIndex] = childKey;
        data[currIndex] = data[childIndex];

        currIndex = childIndex;
    }

    // Place the original root value into its correct sorted position
    keys[currIndex] = keyVal;
    data[currIndex] = dataVal;
}

/// @brief Restores the max-heap property by moving the newly added element up the tree.
/// This function is used after a new element is inserted ar the very end of the heap.
/// 
/// @tparam Key The data type of the sorting keys
/// @tparam Data The data type of the associated payload
/// 
/// @param keys Pointer to the array of keys representing the max-heap.
/// @param data Pointer to the array of data payloads parallel to the `keys` array.
/// @param size The new total number of elements in the heap after insertion
template<typename Key, typename Data>
PBRT_CPU_GPU void HeapBubbleUp(Key* keys, Data* data, int size) {
    int currIndex = size - 1;

    const auto keyVal = keys[currIndex];
    const auto dataVal = data[currIndex];

    while (currIndex > 0) {
        // Parent index is (i - 1) / 2
        const int parentIndex = (currIndex - 1) >> 1;
        const auto parentKey = keys[parentIndex];

        // If the newly inserted value is smaller than or equal to the parent, we are done.
        if (keyVal <= parentKey) break;

        // Move the parent down
        keys[currIndex] = parentKey;
        data[currIndex] = data[parentIndex];

        currIndex = parentIndex;
    }

    // Place the newly inserted value into its correct sorted position
    keys[currIndex] = keyVal;
    data[currIndex] = dataVal;
}

}

#endif
