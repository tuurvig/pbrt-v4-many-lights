#ifndef PBRT_GPU_LIGHTTREEBUILDER_H
#define PBRT_GPU_LIGHTTREEBUILDER_H

#include <pbrt/pbrt.h>

#ifdef PBRT_BUILD_GPU_RENDERER

#include <pbrt/gpu/util.h>
#include <pbrt/lights.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/vecmath.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace pbrt {

// Fixed warp-level constants used by the HPLOC (Hierarchical Pairwise Locally
// Ordered Clustering) builder.
constexpr PBRT_GPU uint32_t kFullMask = std::numeric_limits<uint32_t>::max();
constexpr PBRT_GPU uint32_t kSearchRadiusShift = 3;
constexpr PBRT_GPU uint32_t kSearchRadius = 1u << kSearchRadiusShift;
constexpr PBRT_GPU uint32_t kDecodeMask = (1u << (kSearchRadius + 1u)) - 1u;
constexpr PBRT_GPU uint32_t kEncodeMask = ~(kDecodeMask);
constexpr PBRT_GPU uint32_t kWarpSize = 32;
constexpr PBRT_GPU uint32_t kHalfWarp = 16;
constexpr PBRT_GPU uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();

// Intermediate BVH node that stores spatial bounds and child references.
// Leaves store the light index in both child slots and use kInvalidIndex to
// signal that no further subdivision is needed.
struct LightTreeConstructionNodeGPU {
    LightBounds bounds;
    uint32_t left; // invalidIdx == leaf
    uint32_t right; // leaf => lightIdx
};

// Aggregated pointers to device memory that the builder mutates during HPLOC.
struct LightTreeBuildState {
    Bounds3f allLightBounds;
    uint32_t nLights = 0;

    LightTreeConstructionNodeGPU *dNodes = nullptr;
    uint32_t *dClusterIndices = nullptr;
    uint32_t *dParentIndices = nullptr;
    uint32_t *nMergedClusters = nullptr;
};

template <typename MortonInt, typename CostEvaluator>
__global__ void LightTreeBuilderGPUHplocOuterLoop(LightTreeBuildState state, MortonInt* dMortonCodes);

// Thin RAII wrapper that owns the device buffers required to build the light
// tree and provides a convenience entry point to launch the kernel.
template <typename MortonInt, typename CostEvaluator>
class LightTreeBuilderGPU {
  public:
    LightTreeBuilderGPU() = default;
    LightTreeBuilderGPU(const LightTreeBuilderGPU &) = delete;
    LightTreeBuilderGPU &operator=(const LightTreeBuilderGPU &) = delete;
    ~LightTreeBuilderGPU();

    void Allocate(uint32_t nLights, const Bounds3f &bounds);
    void Release();

    void BuildNodes(const char *description = "Build Nodes");

    LightTreeBuildState &State() { return m_state; }
    const LightTreeBuildState &State() const { return m_state; }

    MortonInt*& MortonCodes() { return m_mortonCodes; }

    static std::array<uint8_t, 3> DetermineAxisOrder(const Bounds3f &bounds);
  private:
    LightTreeBuildState m_state;
    MortonInt* m_mortonCodes;
    bool m_allocated = false;
};
template <typename MortonInt, typename CostEvaluator>
LightTreeBuilderGPU<MortonInt, CostEvaluator>::~LightTreeBuilderGPU() {
    Release();
}

template <typename MortonInt, typename CostEvaluator>
static std::array<uint8_t, 3> LightTreeBuilderGPU<MortonInt, CostEvaluator>::DetermineAxisOrder(const Bounds3f &bounds) {
    std::array<uint8_t, 3> axis{uint8_t(0), uint8_t(1), uint8_t(2)};
    Vector3f diagonal = bounds.Diagonal();

    if (diagonal[axis[0]] < diagonal[axis[1]])
        std::swap(axis[0], axis[1]);
    if (diagonal[axis[1]] < diagonal[axis[2]])
        std::swap(axis[1], axis[2]);
    if (diagonal[axis[0]] < diagonal[axis[1]])
        std::swap(axis[0], axis[1]);

    return axis;
}

template <typename MortonInt, typename CostEvaluator>
void LightTreeBuilderGPU<MortonInt, CostEvaluator>::Allocate(uint32_t nLights, const Bounds3f &bounds) {
    if (m_allocated)
        Release();

    m_state.nLights = nLights;
    m_state.allLightBounds = bounds;

    uint32_t nNodes = nLights > 0 ? (2 * nLights - 1) : 0;
    if (nNodes == 0)
        return;

    m_mortonCodes = GPUAllocAsync<MortonInt>(nLights);

    m_state.dNodes = GPUAllocAsync<LightTreeConstructionNodeGPU>(nNodes);
    m_state.dClusterIndices = GPUAllocAsync<uint32_t>(nLights);
    m_state.dParentIndices = GPUAllocAsync<uint32_t>(nLights);
    m_state.nMergedClusters = GPUAllocAsync<uint32_t>(1);

    uint32_t initialClusters = nLights;
    GPUCopyToDevice(m_state.nMergedClusters, &initialClusters, 1);
    GPUMemsetAsync(m_state.dParentIndices, 0xFF, sizeof(uint32_t) * nLights);

    m_allocated = true;
}

template <typename MortonInt, typename CostEvaluator>
void LightTreeBuilderGPU<MortonInt, CostEvaluator>::Release() {
    if (!m_allocated)
        return;

    if (m_mortonCodes)
        GPUFreeAsync(m_mortonCodes);

    if (m_state.dNodes)
        GPUFreeAsync(m_state.dNodes);
    if (m_state.dClusterIndices)
        GPUFreeAsync(m_state.dClusterIndices);
    if (m_state.dParentIndices)
        GPUFreeAsync(m_state.dParentIndices);
    if (m_state.nMergedClusters)
        GPUFreeAsync(m_state.nMergedClusters);

    m_state = {};
    m_allocated = false;
}

// Compares two Morton codes and returns the bit position of their highest
// differing bit (spatial split level).
template <typename MortonInt>
static PBRT_GPU_INLINE uint64_t MortonCodeDelta(int32_t L, int32_t R, const uint64_t *mortonCodes) {
    MortonInt splitLevel = mortonCodes[L] ^ mortonCodes[R];
    if (splitLevel == 0) {
        // Fake split is used for duplicate Morton codes so the
        // hierarchy continues to grow.
        return L ^ (L + static_cast<MortonInt>(1ull));
    }
        
    return splitLevel;
}

// Walks through the Morton ordering to find the neighbor range that should be
// merged next (i.e. which side becomes the parent).
template <typename MortonInt>
static PBRT_GPU_INLINE uint32_t FindParentIdx(int32_t L, int32_t R, int32_t N,
                                              const uint64_t *mortonCodes) {
    if (L == 0 ||
       (R != N && MortonCodeDelta<MortonInt>(L - 1, L, mortonCodes) >
                  MortonCodeDelta<MortonInt>(R, R + 1, mortonCodes))) {
        return R;
    }

    return L - 1;
}

// Packs the relative lane distance into the low bits so it can travel through
// the warp shuffles together with the encoded cost.
static PBRT_GPU_INLINE uint32_t EncodeRelativeOffset(uint32_t idx, uint32_t neighbor) {
    uint32_t offset = neighbor - idx - 1;
    return offset << 1;
}

static PBRT_GPU_INLINE uint32_t DecodeRelativeOffset(uint32_t idx, uint32_t offset) {
    uint32_t originalOffset = (offset >> 1) + 1;
    uint32_t xorValue = offset ^ idx;
    return (xorValue & 1) == 0 ? idx + originalOffset : idx - originalOffset;
}

template <typename CostEvaluator>
PBRT_GPU uint32_t FindNearestNeighbor(uint32_t nLights, uint32_t clusterIdx,
    uint8_t laneWarpIdx, LightTreeConstructionNodeGPU* dNodes) {

    // Templated functor to evaluate cost
    CostEvaluator costEvaluator;
    
    // Each lane keeps track of the bounds of its current cluster and scans
    // progressively wider radii to find the cheapest merge partner.
    LightBounds clusterBounds;
    if (laneWarpIdx < nLights && clusterIdx != kInvalidIndex) {
        clusterBounds = dNodes[clusterIdx].bounds;
    }
    
    DCHECK_EQ(laneWarpIdx < nLights, clusterIdx != kInvalidIndex);
    const bool active = laneWarpIdx < nLights && clusterIdx != kInvalidIndex;

    // minCostIdx encodes both the relative lane offset and the merge cost and
    // gets propagated across the warp
    uint32_t minCostIdx = kInvalidIndex;
    for (uint32_t r = 1; r <= kSearchRadius; ++r) {
        uint32_t neighborIdx = laneWarpIdx + r;
        uint32_t neighborClusterIdx = __shfl_sync(kFullMask, clusterIdx, neighborIdx);

        uint32_t newCostIdx0 = kInvalidIndex;
        uint32_t newCostIdx1 = kInvalidIndex;
        if (neighborIdx < nLights) {
            LightBounds neighborBounds = dNodes[neighborClusterIdx].bounds;
            neighborBounds = Union(neighborBounds, clusterBounds);

            float newCost = costEvaluator(neighborBounds);
            uint32_t newCostInt = __float_as_uint(newCost);

            uint32_t encode0 = EncodeRelativeOffset(laneWarpIdx, neighborIdx);
            uint32_t encode1 = (newCostInt << 1) & kEncodeMask;

            // encoded for me looking right
            newCostIdx0 = encode1 | encode0 | (laneWarpIdx & 1);
            // encoded for the neighbor + r looking bad at me
            newCostIdx1 = encode1 | encode0 | ((neighborIdx & 1) ^ 1);
        }

        // Accumulate cost to the right
        minCostIdx = std::min(minCostIdx, newCostIdx0);

        // Accumulate cost from the left
        // We need to fetch what our left neighbor computed for us
        // Stored in newCostIdx1 at lane - r
        uint32_t costFromLeft = __shfl_sync(kFullMask, newCostIdx1, laneWarpIdx - r);
        minCostIdx = std::min(minCostIdx, costFromLeft);
    }

    uint32_t decodedNN = kInvalidIndex;
    if (active && minCostIdx != kInvalidIndex) {
        uint32_t unmasked = minCostIdx & kDecodeMask;
        decodedNN = DecodeRelativeOffset(laneWarpIdx, unmasked);
    }

    return decodedNN;
}

// Performs the actual merge between mutually nearest neighbors and compacts
// the active cluster list for the next PlocMerge round.
template <typename CostEvaluator>
PBRT_GPU uint32_t MergeClusters(uint32_t nLights, uint32_t &clusterIdx, uint8_t laneWarpIdx,
    uint32_t* nMergedClustersPtr, LightTreeConstructionNodeGPU* dNodes) {

    uint32_t nearestNeighborIdx = FindNearestNeighbor<CostEvaluator>(nLights, clusterIdx, laneWarpIdx, dNodes);

    uint32_t neighborNNIdx = __shfl_sync(kFullMask, nearestNeighborIdx, nearestNeighborIdx);
    uint32_t neighborClusterIdx = __shfl_sync(kFullMask, clusterIdx, nearestNeighborIdx);

    const bool laneActive = laneWarpIdx < nLights;
    const bool mutual = laneActive && laneWarpIdx == neighborNNIdx;
    const bool merge = mutual && laneWarpIdx < nearestNeighborIdx;

    uint32_t mergeMask = __ballot_sync(kFullMask, merge);
    uint32_t mergeCount = __popc(mergeMask);

    uint32_t baseIdx = kInvalidIndex;
    if (laneWarpIdx == 0)
        baseIdx = atomicAdd(nMergedClustersPtr, mergeCount);
    baseIdx = __shfl_sync(kFullMask, baseIdx, 0);

    // Use a mask to count neighbors with lower lane indices
    uint32_t countMask = (1u << laneWarpIdx) - 1;
    uint32_t relativeIdx = __popc(mergeMask & countMask);

    if (merge) {
        LightBounds clusterBounds = dNodes[clusterIdx].bounds;
        LightBounds neighborBounds = dNodes[neighborClusterIdx].bounds;
        clusterBounds = Union(clusterBounds, neighborBounds);

        LightTreeConstructionNodeGPU node;
        node.bounds = clusterBounds;
        node.left = clusterIdx;
        node.right = neighborClusterIdx;
        clusterIdx = baseIdx + relativeIdx;
        dNodes[clusterIdx] = node;
    }

    uint32_t validMask = __ballot_sync(kFullMask, merge || !mutual);
    int32_t shift = __fns(validMask, 0, laneWarpIdx + 1);

    clusterIdx = __shfl_sync(kFullMask, clusterIdx, shift);
    if (shift == -1)
        clusterIdx = kInvalidIndex;

    return nLights - mergeCount;
}

// Runs the HPLOC reduction for the [start, start+nLeft+nRight) segment until
// the segment reaches the requested threshold (usually kHalfWarp).
template <typename CostEvaluator>
PBRT_GPU void PlocMerge(uint32_t start, uint32_t nLeft, uint32_t nRight, uint32_t threshold,
    uint32_t clusterIdx, uint8_t laneWarpIdx, const LightTreeBuildState &state) {
    uint32_t nLightsInCurrentStep = nLeft + nRight;
    uint32_t nLightsToProcess = nLightsInCurrentStep;

    while (nLightsToProcess > threshold) {
        nLightsToProcess = MergeClusters<CostEvaluator>(nLightsToProcess, clusterIdx, laneWarpIdx,
                           state.nMergedClusters, state.dNodes);
    }

    if (laneWarpIdx < nLightsInCurrentStep) {
         state.dClusterIndices[start + laneWarpIdx] = clusterIdx;
    }   

    __threadfence();
}

// Loads the cluster index for the lane if the lane falls inside the desired
// subrange. Returns whether the load was valid so the caller can ballot on it.
PBRT_GPU_INLINE bool LoadIndex(uint32_t &clusterIdx, uint32_t start, uint32_t end,
    uint32_t offset, uint32_t *clusterIndices, uint8_t laneWarpId) {
    uint32_t index = laneWarpId - offset;
    bool validLaneIdx = index < std::min(end - start, kHalfWarp);

    if (validLaneIdx)
        clusterIdx = clusterIndices[start + index];

    return validLaneIdx;
}

template <typename MortonInt, typename CostEvaluator>
__global__ void LightTreeBuilderGPUHplocOuterLoop(LightTreeBuildState state, MortonInt* dMortonCodes) {
    uint32_t memStart = blockIdx.x * blockDim.x;
    uint32_t tid = memStart + threadIdx.x;

    uint32_t leftIdx = tid;
    uint32_t rightIdx = tid;
    uint32_t splitIdx = 0;

    // Each lane initially owns one light. As merges complete, the active lanes
    // shrink until the root has been produced.
    bool laneActive = (tid < state.nLights);
    while (__ballot_sync(kFullMask, laneActive)) {
        if (laneActive) {
            uint32_t prevIdx = 0;
            uint32_t sibling =
                FindParentIdx<MortonInt>(leftIdx, rightIdx, state.nLights - 1, dMortonCodes);
            if (sibling == rightIdx) {
                prevIdx = atomicExch(&state.dParentIndices[rightIdx], leftIdx);
                if (prevIdx != kInvalidIndex) {
                    splitIdx = rightIdx + 1;
                    rightIdx = prevIdx;
                }
            } else {
                prevIdx = atomicExch(&state.dParentIndices[leftIdx - 1], rightIdx);
                if (prevIdx != kInvalidIndex) {
                    splitIdx = leftIdx;
                    leftIdx = prevIdx;
                }
            }
            if (prevIdx == kInvalidIndex)
                laneActive = false;
        }

        uint32_t size = rightIdx - leftIdx + 1;
        bool isTreeRoot = laneActive && size == state.nLights;

        uint32_t warpMask = __ballot_sync(
            kFullMask, laneActive && (size > kHalfWarp) || isTreeRoot);

        while (warpMask) {
            uint8_t selectedLaneIdx = __ffs(warpMask) - 1;

            uint32_t startL = __shfl_sync(kFullMask, leftIdx, selectedLaneIdx);
            uint32_t endR = __shfl_sync(kFullMask, rightIdx, selectedLaneIdx) + 1;
            uint32_t endL = __shfl_sync(kFullMask, splitIdx, selectedLaneIdx);
            uint32_t startR = endL;
            uint32_t threshold =
                __shfl_sync(kFullMask, isTreeRoot, selectedLaneIdx) ? 1 : kHalfWarp;
            
            uint8_t laneWarpId = threadIdx.x & (kWarpSize - 1);

            uint32_t idx = kInvalidIndex;
            bool isLeftValidIndex = LoadIndex(idx, startL, endL, 0, state.dClusterIndices, laneWarpId);
            uint32_t nLeftClusters = __popc(__ballot_sync(kFullMask, isLeftValidIndex && idx != kInvalidIndex));

            bool isRightValidIndex = LoadIndex(idx, startR, endR, nLeftClusters, state.dClusterIndices, laneWarpId);
            uint32_t nRightClusters = __popc(__ballot_sync(kFullMask, isRightValidIndex && idx != kInvalidIndex));

            PlocMerge<CostEvaluator>(startL, nLeftClusters, nRightClusters, threshold, idx, laneWarpId, state);

            warpMask = warpMask & (warpMask - 1);
        }
    }
}

template <typename MortonInt, typename CostEvaluator>
void LightTreeBuilderGPU<MortonInt, CostEvaluator>::BuildNodes(const char *description) {
    if (m_state.nLights == 0)
        return;

#ifdef NVTX
    nvtxRangePush(description);
#endif

#ifdef PBRT_DEBUG_BUILD
    LOG_VERBOSE("Launching %s", description);
#endif

    auto kernel = &LightTreeBuilderGPUHplocOuterLoop<MortonInt, CostEvaluator>;
    int blockSize = GetBlockSize(description, kernel);
    {
        KernelTimerWrapper timer(GetProfilerEvents(description, ProfilerKernelGroup::HPLOC));
        int gridSize = (m_state.nLights + blockSize - 1) / blockSize;
        kernel<<<gridSize, blockSize>>>(m_state, m_mortonCodes);
    }

#ifdef PBRT_DEBUG_BUILD
    GPUWait();
    LOG_VERBOSE("Post-sync %s", description);
#endif

#ifdef NVTX
    nvtxRangePop();
#endif

    ReportKernelStats(ProfilerKernelGroup::HPLOC);
}

}  // namespace pbrt

#endif  // PBRT_BUILD_GPU_RENDERER

#endif  // PBRT_GPU_LIGHTTREEBUILDER_H
