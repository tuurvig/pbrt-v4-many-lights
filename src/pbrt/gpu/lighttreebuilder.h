// Copyright(c) 2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

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
#include <pbrt/util/lighttree_generic.h>

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace pbrt {

/// Fixed warp-level constants used by the HPLOC (Hierarchical Pairwise Locally
/// Ordered Clustering) builder.
constexpr PBRT_GPU uint32_t kFullMask = std::numeric_limits<uint32_t>::max();
constexpr PBRT_GPU uint32_t kSearchRadiusShift = 3;
constexpr PBRT_GPU uint32_t kSearchRadius = 1u << kSearchRadiusShift;
constexpr PBRT_GPU uint32_t kDecodeMask = (1u << (kSearchRadius + 1u)) - 1u;
constexpr PBRT_GPU uint32_t kEncodeMask = ~(kDecodeMask);
constexpr PBRT_GPU uint32_t kWarpSize = 32;
constexpr PBRT_GPU uint32_t kHalfWarp = 16;
constexpr PBRT_GPU uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();

/// @brief Aggregated pointers to device memory that the builder mutates during the HPLOC process.
/// This struct holds the current state of the tree building algorithm, including the array of nodes,
/// cluster tracking arrays, and the global scene bounds.
/// @tparam LightBoundsType The type used for spatial and directional bounds
template<typename LightBoundsType>
struct LightTreeBuildState {
    Bounds3f allLightBounds;
    uint32_t nLights = 0;

    LightTreeConstructionNodeGPU<LightBoundsType> *dNodes = nullptr;
    uint32_t *dClusterIndices = nullptr;
    uint32_t *dParentIndices = nullptr;
    uint32_t *nMergedClusters = nullptr;
};

template <typename LightBoundsType, typename MortonInt, typename CostEvaluator>
__global__ void LightTreeBuilderGPUHplocOuterLoop(LightTreeBuildState<LightBoundsType> state, MortonInt* dMortonCodes, CostEvaluator evaluator);

/// @brief A thin RAII wrapper managing the device buffers and lifecycle of the
/// GPU-based light tree builder.
/// 
/// This class encapsulates the memory allocation, execution, and cleanup required to build a
/// hierarchical light tree on the GPU using the HPLOC algorithm. It provides an interface for
/// launching the tree building HPLOC GPU kernels in CUDA.
/// 
/// @tparam LightBoundsType The type of the bounding volume (AABB or Spherical)
/// @tparam MortonInt The integer type used for Morton codes (typically 32 or 64-bit).
/// @tparam CostEvaluator A functor for calculating the split/merge cost 
template <typename LightBoundsType, typename MortonInt, typename CostEvaluator>
class LightTreeBuilderGPU {
  public:
    LightTreeBuilderGPU() = default;
    LightTreeBuilderGPU(const LightTreeBuilderGPU &) = delete;
    LightTreeBuilderGPU &operator=(const LightTreeBuilderGPU &) = delete;
    ~LightTreeBuilderGPU();

    /// @brief Allocated the necessary device memory for the tree construction arrays based on the total number of lights.
    void Allocate(uint32_t nLights, const Bounds3f &bounds);

    /// @brief Frees all allocated device memory. Called automatically on destruction.
    void Release();

    /// @brief Launches the main HPLOC kernel to construct the internal tree nodes from te sorted leaves.
    /// @param evaluator The cost evaluation function utilized to drive the clustering heuristics.
    /// @param description Optional parameter name for the kernel launch.
    void BuildNodes(CostEvaluator evaluator, const char *description = "Build Nodes");

    LightTreeBuildState<LightBoundsType> &State() { return m_state; }
    const LightTreeBuildState<LightBoundsType> &State() const { return m_state; }

    MortonInt*& MortonCodes() { return m_mortonCodes; }

    /// @brief Determines the optimal mapping for spatial dimension (X, Y, Z) to bit interleaving order based on the bounding box aspect ration.
    /// @param bounds The spatial bounding box to analyze.
    /// @return An array of axis indices sorted by length (longest to shortest).
    static std::array<uint8_t, 3> DetermineAxisOrder(const Bounds3f &bounds);
  private:
    LightTreeBuildState<LightBoundsType> m_state;
    MortonInt* m_mortonCodes;
    bool m_allocated = false;
};
template <typename LightBoundsType, typename MortonInt, typename CostEvaluator>
LightTreeBuilderGPU<LightBoundsType, MortonInt, CostEvaluator>::~LightTreeBuilderGPU() {
    Release();
}

/// @brief Assign Morton codes to all light primitives and perform a radix sort.
/// This spatial sorting step must precede the HPLOC construction algorithm, because it ensures
/// that primitives located near each other in 3D space are also adjancent in the linear array.
/// 
/// @tparam LightBoundsType The bounding volume type.
/// @tparam MortonInt The integer type used to store the Morton codes.
/// @tparam F A functor responsible for calculating the Morton code given a light index.
///
/// @param buildState The current build state referencing the cluster index array.
/// @param dMortonCode The initially unsorted device array of Morton codes.
/// @param func The mapping function.
/// @return A pointer to the newly allocated and sorted array of Morton codes.
template <typename LightBoundsType, typename MortonInt, typename F>
MortonInt* SortNodesMorton(LightTreeBuildState<LightBoundsType>& buildState, MortonInt* dMortonCodes, F func) {
    GPUParallelFor("Assign Morton Codes", ProfilerKernelGroup::HPLOC, buildState.nLights, func);

    MortonInt *dMortonCodesSorted = GPUAllocAsync<MortonInt>(buildState.nLights);
    uint32_t *dClusterIndicesSorted = GPUAllocAsync<uint32_t>(buildState.nLights);

    void *dTempStorage = nullptr;
    size_t tempStorageBytes = 0;

    constexpr uint32_t beginBit = 0;
    constexpr uint32_t endBit = sizeof(MortonInt) * 8;

    const char *description = "Radix Sort Morton keys";
    {
        KernelTimerWrapper timer(GetProfilerEvents(description, ProfilerKernelGroup::HPLOC));
        cub::DeviceRadixSort::SortPairs(dTempStorage, tempStorageBytes, dMortonCodes,
            dMortonCodesSorted, buildState.dClusterIndices, dClusterIndicesSorted,
            buildState.nLights, beginBit, endBit);

        dTempStorage = GPUAllocAsync<uint8_t>(tempStorageBytes);

        cub::DeviceRadixSort::SortPairs(dTempStorage, tempStorageBytes, dMortonCodes,
            dMortonCodesSorted, buildState.dClusterIndices, dClusterIndicesSorted,
            buildState.nLights, beginBit, endBit);
    }

    GPUFreeAsync(dTempStorage);
    GPUFreeAsync(dMortonCodes);
    GPUFreeAsync(buildState.dClusterIndices);
    buildState.dClusterIndices = dClusterIndicesSorted;

    return dMortonCodesSorted;
}

template <typename LightBoundsType, typename MortonInt, typename CostEvaluator>
std::array<uint8_t, 3> LightTreeBuilderGPU<LightBoundsType, MortonInt, CostEvaluator>::DetermineAxisOrder(const Bounds3f &bounds) {
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

template <typename LightBoundsType, typename MortonInt, typename CostEvaluator>
void LightTreeBuilderGPU<LightBoundsType, MortonInt, CostEvaluator>::Allocate(uint32_t nLights, const Bounds3f &bounds) {
    if (m_allocated)
        Release();

    m_state.nLights = nLights;
    m_state.allLightBounds = bounds;

    uint32_t nNodes = nLights > 0 ? (2 * nLights - 1) : 0;
    if (nNodes == 0)
        return;

    m_mortonCodes = GPUAllocAsync<MortonInt>(nLights);

    m_state.dNodes = GPUAllocAsync<LightTreeConstructionNodeGPU<LightBoundsType>>(nNodes);
    m_state.dClusterIndices = GPUAllocAsync<uint32_t>(nLights);
    m_state.dParentIndices = GPUAllocAsync<uint32_t>(nLights);
    m_state.nMergedClusters = GPUAllocAsync<uint32_t>(1);

    uint32_t initialClusters = nLights;
    GPUCopyToDevice(m_state.nMergedClusters, &initialClusters, 1);
    GPUMemsetAsync(m_state.dParentIndices, 0xFF, sizeof(uint32_t) * nLights);

    m_allocated = true;
}

template <typename LightBoundsType, typename MortonInt, typename CostEvaluator>
void LightTreeBuilderGPU<LightBoundsType, MortonInt, CostEvaluator>::Release() {
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

/// @brief Compares two Morton codes and identifies the index of their highest differing bit (spatial split level).
/// @tparam MortonInt The type of the Morton code.
/// @param L The index of the left element.
/// @param R The index of the right element.
/// @param mortonCodes The array of sorted Morton codes.
/// @return A value representing the level of divergence.
template <typename MortonInt>
static PBRT_GPU_INLINE uint64_t MortonCodeDelta(int32_t L, int32_t R, const MortonInt *mortonCodes) {
    MortonInt splitLevel = mortonCodes[L] ^ mortonCodes[R];
    if (splitLevel == 0) {
        // Fake split is used for duplicate Morton codes so the
        // hierarchy continues to grow.
        return L ^ (L + static_cast<MortonInt>(1ull));
    }
        
    return splitLevel;
}

/// @brief Walks through the Morton ordering to find the neighbor range that should be
/// merged next (i.e. which side becomes the parent).
template <typename MortonInt>
static PBRT_GPU_INLINE uint32_t FindParentIdx(int32_t L, int32_t R, int32_t N,
                                              const MortonInt* mortonCodes) {
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

/// @brief Evaluates a local neighbourhood of cluster to find the optimal partner for a merge.
/// 
/// This function utilizes warp-level shuffle operation (`__shfl_sync`) to inspect neighbours up to `kSearchRadius`
/// slots away in both directions. It applies the provided `CostEvaluator` heuristic to calculate the cost
/// of joining the active cluster with its neighbors, return the index of the neighbor that minimez this cost.
/// 
/// @param nLights Number of active lights/clusters in the current segment
/// @param clusterIdx The global index of the cluster owned by the current thread.
/// @param laneWarpIdx The lane index (0-31) of the current thread within the warp.
/// @param dNodes The array of partially built tree nodes.
/// @param evaluator Instance of the cost evaluation functor.
/// @return the lane index of the nearest neighbor, or kInvalidIndex if none is found.
template <typename LightBoundsType, typename CostEvaluator>
PBRT_GPU uint32_t FindNearestNeighbor(uint32_t nLights, uint32_t clusterIdx,
    uint8_t laneWarpIdx, LightTreeConstructionNodeGPU<LightBoundsType>* dNodes, CostEvaluator evaluator) {
    // Each lane keeps track of the bounds of its current cluster and scans
    // progressively wider radii to find the cheapest merge partner.
    LightBoundsType clusterBounds;
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
            LightBoundsType neighborBounds = dNodes[neighborClusterIdx].bounds;
            neighborBounds = Union(neighborBounds, clusterBounds);

            float newCost = evaluator(neighborBounds);
            const uint32_t newCostInt = FloatToBits(newCost);

            const uint32_t encode0 = EncodeRelativeOffset(laneWarpIdx, neighborIdx);
            const uint32_t encode1 = (newCostInt << 1) & kEncodeMask;

            // encoded for me looking right
            newCostIdx0 = encode1 | encode0 | (laneWarpIdx & 1);
            // encoded for the neighbor at (lane + r) looking back at me
            newCostIdx1 = encode1 | encode0 | ((neighborIdx & 1) ^ 1);
        }

        // Accumulate cost to the right
        minCostIdx = std::min(minCostIdx, newCostIdx0);

        // Accumulate cost from the left
        // We need to fetch what our left neighbor computed for us
        // Stored in neighbour's newCostIdx1 at (lane - r)
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

/// @brief Performs the actual merge between mutually nearest neighbors and compacts
/// the active cluster list for the next PlocMerge round.
///
/// Threads that identified each other as optimal partners will collaborate to allocate an interior node,
/// compute the union of their bounding volumes, and record the new node into `dNodes`.
/// Threads that merged are compacted, and the active count is reduced.
/// 
/// @return The new number of active clusters within the current warp/segment.
template<typename LightBoundsType>
PBRT_GPU_INLINE uint32_t MergeClusters(uint32_t nLights, uint32_t &clusterIdx, uint8_t laneWarpIdx,
    uint32_t nearestNeighborIdx, uint32_t* nMergedClustersPtr, LightTreeConstructionNodeGPU<LightBoundsType>* dNodes) {

    uint32_t neighborNNIdx = __shfl_sync(kFullMask, nearestNeighborIdx, nearestNeighborIdx);
    uint32_t neighborClusterIdx = __shfl_sync(kFullMask, clusterIdx, nearestNeighborIdx);

    const bool laneActive = laneWarpIdx < nLights;

    // Two clusters are "mutual nearest neighbors" if they both point to each other.
    const bool mutual = laneActive && laneWarpIdx == neighborNNIdx;

    // To avoid creating duplicate parent nodes, only ONE thread from the mutual pair
    // should perform the merge. We pick the one with the lower index.
    const bool merge = mutual && laneWarpIdx < nearestNeighborIdx;

    // Count total merges happening accross the entire warp.
    uint32_t mergeMask = __ballot_sync(kFullMask, merge);
    uint32_t mergeCount = __popc(mergeMask); // count the number of bits in the mask.

    // Global memory allocation for the new parent nodes.
    // Instead of every merging thread doing a slow atomicAdd, only lane 0 does a single
    // bulk atomic allocation for the whole warp, reducing memory contention.
    uint32_t baseIdx = kInvalidIndex;
    if (laneWarpIdx == 0) {
        baseIdx = atomicAdd(nMergedClustersPtr, mergeCount);
    }
    // Boadcast the globally allocated base index from lane 0
    baseIdx = __shfl_sync(kFullMask, baseIdx, 0);

    // Compute local offset for writing out nodes.
    // Create a bitmask covering only the lanes strictly lower than the current lane.
    uint32_t countMask = (1u << laneWarpIdx) - 1;
    // By counting the set bits before this lane, the thread knows exactly
    // which offset it should use to write its new node into the allocated block.
    uint32_t relativeIdx = __popc(mergeMask & countMask);

    // Merge and write to global memory.
    if (merge) {
        LightBoundsType clusterBounds = dNodes[clusterIdx].bounds;
        LightBoundsType neighborBounds = dNodes[neighborClusterIdx].bounds;
        clusterBounds = Union(clusterBounds, neighborBounds);

        LightTreeConstructionNodeGPU<LightBoundsType> node(clusterBounds, clusterIdx, neighborClusterIdx);
        clusterIdx = baseIdx + relativeIdx;
        dNodes[clusterIdx] = node;
    }

    // This removes the consumed neighbors from the active list.
    // A lane survives to the next round if it just created a new parent node (merge)
    // or if it was completely ignored in this round (!mutual).
    // The passive half of the mutual pair is dropped.
    uint32_t validMask = __ballot_sync(kFullMask, merge || !mutual);

    // __fns calculates the source lane index that contains the valid data
    // which need to be moved into the current lanes position to fill the gaps.
    int32_t shift = __fns(validMask, 0, laneWarpIdx + 1);

    // Compact the surviving cluster indices by shifting their position.
    // The other pair of merge left a gap that needs to be filled.
    clusterIdx = __shfl_sync(kFullMask, clusterIdx, shift);

    // If the shift index is invalid, this lane no longer holds active data.
    if (shift == -1)
        clusterIdx = kInvalidIndex;

    // Return the new size of the active array for this warp
    return nLights - mergeCount;
}

/// @brief Continuously applies nearest-neighbour matching and merging within
/// a specific thread segment. This function runs the PLOC (Pairwise Locally Ordered Clustering)
/// algorithm locally until the number of active clusters in the segment drops below a specified threshold.
template <typename LightBoundsType, typename CostEvaluator>
PBRT_GPU void PlocMerge(uint32_t start, uint32_t nLeft, uint32_t nRight, uint32_t threshold,
    uint32_t clusterIdx, uint8_t laneWarpIdx, const LightTreeBuildState<LightBoundsType> &state, CostEvaluator evaluator) {

    // Total number of clustes this warp needs to preocess initially.
    uint32_t nLightsInCurrentStep = nLeft + nRight;
    uint32_t nLightsToProcess = nLightsInCurrentStep;

    // Iteratively reduce the clusters until we hit the target threshold (halfWarp).
    while (nLightsToProcess > threshold) {
        //Scan the local neighbourhood to find the cheapest merge partner.
        uint32_t nearestNeighborIdx = FindNearestNeighbor<LightBoundsType, CostEvaluator>(nLightsToProcess, clusterIdx, laneWarpIdx, state.dNodes, evaluator);

        // Perform the merge if mutually paired, create the new parent node, and compact the active lanes.
        nLightsToProcess = MergeClusters<LightBoundsType>(nLightsToProcess, clusterIdx, laneWarpIdx, nearestNeighborIdx,
                           state.nMergedClusters, state.dNodes);
    }

    // After the reduction loop, the surviving active cluster (threshold)
    // need to be written back to the global array so the next level of the hierarchy can read them. 
    if (laneWarpIdx < nLightsInCurrentStep) {
         state.dClusterIndices[start + laneWarpIdx] = clusterIdx;
    }   

    // Ensure all memory writes (nodes and cluster indices) are visible to all other threads
    // before this warp moves on or terminates. This prevents race conditions when
    // higher levels of the tree attempt to read these newly created nodes.
    __threadfence();
}

/// Loads the cluster index for the lane if the lane falls inside the desired
/// subrange. Returns whether the load was valid so the caller can ballot on it.
PBRT_GPU_INLINE bool LoadIndex(uint32_t &clusterIdx, uint32_t start, uint32_t end,
    uint32_t offset, uint32_t *clusterIndices, uint8_t laneWarpId) {
    uint32_t index = laneWarpId - offset;
    bool validLaneIdx = index < std::min(end - start, kHalfWarp);

    if (validLaneIdx)
        clusterIdx = clusterIndices[start + index];

    return validLaneIdx;
}

/// @brief The main CUDA kernel responsible for orchestrating the HPLOC tree construction algorithm.
///
/// This kernel operates in a bottom-up fashion. Each thread starts by owning one leaf node (a light).
/// Threads use Morton codes to identify spatial segments and boundaries. When a segment boundary is resolved,
/// a warp assumes control of a block of clusters and iteratively merges them using `PlocMerge` until the sub-tree
/// is sufficiently reduced. Eventually, all segments merge towards the single root node.
/// 
/// @param state The global state holding pointers to device arrays (nodes, indices).
/// @param dMortonCodes The array of sorted Morton codes.
/// @param evaluator The cost function guiding the local pairwise merges.
template <typename LightBoundsType, typename MortonInt, typename CostEvaluator>
__global__ void LightTreeBuilderGPUHplocOuterLoop(LightTreeBuildState<LightBoundsType> state, MortonInt* dMortonCodes, CostEvaluator evaluator) {
    uint32_t memStart = blockIdx.x * blockDim.x;
    uint32_t tid = memStart + threadIdx.x;

    uint32_t leftIdx = tid;
    uint32_t rightIdx = tid;
    uint32_t splitIdx = 0;

    // Each lane initially owns one light. As merges complete, the active lanes
    // shrink until the root has been produced.
    bool laneActive = (tid < state.nLights);

    // The warp continues executing as long as at least one thread in it is still active.
    while (__ballot_sync(kFullMask, laneActive)) {

        // Segment discovery
        if (laneActive) {
            // Determine if this segment should merge with its left or right neighbour based on morton bits.
            const uint32_t sibling = FindParentIdx<MortonInt>(leftIdx, rightIdx, state.nLights - 1, dMortonCodes);
            uint32_t prevIdx = 0;
            if (sibling == rightIdx) {
                prevIdx = atomicExch(&state.dParentIndices[rightIdx], leftIdx);
                if (prevIdx != kInvalidIndex) {
                    splitIdx = rightIdx + 1;
                    rightIdx = prevIdx; // Expand the right boundary.
                }
            } else {
                prevIdx = atomicExch(&state.dParentIndices[leftIdx - 1], rightIdx);
                if (prevIdx != kInvalidIndex) {
                    splitIdx = leftIdx;
                    leftIdx = prevIdx; // Expand the left boundary.  
                }
            }
            if (prevIdx == kInvalidIndex)
                laneActive = false;
        }

        // Warp delegation and Ploc merge
        uint32_t size = rightIdx - leftIdx + 1;
        bool isTreeRoot = laneActive && size == state.nLights;

        // create a mask of all threads in the warp that have successfully formed a segment
        // large enough to warrant a parallel warp-level merge (size > kHalfWarp), or if it's the final root.
        uint32_t warpMask = __ballot_sync(
            kFullMask, laneActive && (size > kHalfWarp) || isTreeRoot);

        // Process each flagged segment seguentially using the entire warp's compute power.
        while (warpMask) {
            // Find the lowest active lane in the mask
            const uint8_t selectedLaneIdx = __ffs(warpMask) - 1;

            // Broadcast the segment boundaries of the selected lane to all other 31 lanes in the warp.
            const uint32_t startL = __shfl_sync(kFullMask, leftIdx, selectedLaneIdx);
            const uint32_t endR = __shfl_sync(kFullMask, rightIdx, selectedLaneIdx) + 1;
            const uint32_t endL = __shfl_sync(kFullMask, splitIdx, selectedLaneIdx);
            const uint32_t startR = endL;
            const uint32_t threshold =
                __shfl_sync(kFullMask, isTreeRoot, selectedLaneIdx) ? 1 : kHalfWarp;
            
            const uint8_t laneWarpId = threadIdx.x & (kWarpSize - 1);

            uint32_t idx = kInvalidIndex;
            const bool isLeftValidIndex = LoadIndex(idx, startL, endL, 0, state.dClusterIndices, laneWarpId);
            const uint32_t nLeftClusters = __popc(__ballot_sync(kFullMask, isLeftValidIndex && idx != kInvalidIndex));

            const bool isRightValidIndex = LoadIndex(idx, startR, endR, nLeftClusters, state.dClusterIndices, laneWarpId);
            const uint32_t nRightClusters = __popc(__ballot_sync(kFullMask, isRightValidIndex && idx != kInvalidIndex));

            PlocMerge<LightBoundsType, CostEvaluator>(startL, nLeftClusters, nRightClusters, threshold, idx, laneWarpId, state, evaluator);

            warpMask = warpMask & (warpMask - 1);
        }
    }
}

template <typename LightBoundsType, typename MortonInt, typename CostEvaluator>
void LightTreeBuilderGPU<LightBoundsType, MortonInt, CostEvaluator>::BuildNodes(CostEvaluator evaluator, const char *description) {
    if (m_state.nLights == 0)
        return;

#ifdef NVTX
    nvtxRangePush(description);
#endif

#ifdef PBRT_DEBUG_BUILD
    LOG_VERBOSE("Launching %s", description);
#endif

    auto kernel = &LightTreeBuilderGPUHplocOuterLoop<LightBoundsType, MortonInt, CostEvaluator>;
    int blockSize = GetBlockSize(description, kernel);
    {
        KernelTimerWrapper timer(GetProfilerEvents(description, ProfilerKernelGroup::HPLOC));
        int gridSize = (m_state.nLights + blockSize - 1) / blockSize;
        kernel<<<gridSize, blockSize>>>(m_state, m_mortonCodes, evaluator);
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
