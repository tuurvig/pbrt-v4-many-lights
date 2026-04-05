// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Contributions Copyright(c) 2026 Richard Kvasnica.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_HASH_H
#define PBRT_UTIL_HASH_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>

#include <string.h>
#include <cstdint>
#include <cstring>

namespace pbrt {

// https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
PBRT_CPU_GPU inline uint64_t MurmurHash64A(const unsigned char *key, size_t len,
                                           uint64_t seed) {
    const uint64_t m = 0xc6a4a7935bd1e995ull;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const unsigned char *end = key + 8 * (len / 8);

    while (key != end) {
        uint64_t k;
        std::memcpy(&k, key, sizeof(uint64_t));
        key += 8;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    switch (len & 7) {
    case 7:
        h ^= uint64_t(key[6]) << 48;
    case 6:
        h ^= uint64_t(key[5]) << 40;
    case 5:
        h ^= uint64_t(key[4]) << 32;
    case 4:
        h ^= uint64_t(key[3]) << 24;
    case 3:
        h ^= uint64_t(key[2]) << 16;
    case 2:
        h ^= uint64_t(key[1]) << 8;
    case 1:
        h ^= uint64_t(key[0]);
        h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

// PCG Fast Hashing function usable on GPU
// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
PBRT_CPU_GPU inline uint32_t HashPCG(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hashing Inline Functions
// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
PBRT_CPU_GPU inline uint64_t MixBits(uint64_t v);

inline uint64_t MixBits(uint64_t v) {
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44d;
    v ^= (v >> 33);
    return v;
}

template <typename T>
PBRT_CPU_GPU inline uint64_t HashBuffer(const T *ptr, size_t size, uint64_t seed = 0) {
    return MurmurHash64A((const unsigned char *)ptr, size, seed);
}

template <typename... Args>
PBRT_CPU_GPU inline uint64_t Hash(Args... args);

template <typename... Args>
PBRT_CPU_GPU inline void hashRecursiveCopy(char *buf, Args...);

template <>
PBRT_CPU_GPU inline void hashRecursiveCopy(char *buf) {}

template <typename T, typename... Args>
PBRT_CPU_GPU inline void hashRecursiveCopy(char *buf, T v, Args... args) {
    memcpy(buf, &v, sizeof(T));
    hashRecursiveCopy(buf + sizeof(T), args...);
}

template <typename... Args>
PBRT_CPU_GPU inline uint64_t Hash(Args... args) {
    // C++, you never cease to amaze: https://stackoverflow.com/a/57246704
    constexpr size_t sz = (sizeof(Args) + ... + 0);
    constexpr size_t n = (sz + 7) / 8;
    uint64_t buf[n];
    hashRecursiveCopy((char *)buf, args...);
    return MurmurHash64A((const unsigned char *)buf, sz, 0);
}

template <typename... Args>
PBRT_CPU_GPU inline Float HashFloat(Args... args) {
    return uint32_t(Hash(args...)) * 0x1p-32f;
}

PBRT_CPU_GPU inline uint32_t FastIntegerHash(uint32_t input) {
    return HashPCG(input);
}

PBRT_CPU_GPU inline uint16_t PackNormalizedFloat(const Float val) {
    DCHECK_LE(val, 1);
    DCHECK_LE(0, val);
    return static_cast<uint16_t>(val * std::numeric_limits<uint16_t>::max());
}

template <typename IntType>
PBRT_CPU_GPU inline Float UnpackToFloat(IntType val) {
    return static_cast<Float>(val) / static_cast<Float>(std::numeric_limits<IntType>::max());
}

}  // namespace pbrt

#endif  // PBRT_UTIL_HASH_H
