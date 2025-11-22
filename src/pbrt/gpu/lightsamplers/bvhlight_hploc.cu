#include <pbrt/lightsamplers/bvhlight.h>

#include <pbrt/gpu/util.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/vecmath.h>

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <limits>
#include <vector>

namespace pbrt {
PBRT_GPU constexpr float kFloatMax = std::numeric_limits<float>::max();
PBRT_GPU constexpr uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();
PBRT_GPU constexpr uint32_t kFullMask = std::numeric_limits<uint32_t>::max();
PBRT_GPU constexpr uint32_t kSearchRadiusShift = 3;
PBRT_GPU constexpr uint32_t kSearchRadius = 1u << kSearchRadiusShift;
PBRT_GPU constexpr uint32_t kDecodeMask = (1u << (kSearchRadius + 1u)) - 1u;
PBRT_GPU constexpr uint32_t kEncodeMask = ~(kDecodeMask);
PBRT_GPU constexpr uint32_t kWarpSize = 32;
PBRT_GPU constexpr uint32_t kHalfWarp = 16;

struct LightBVHConstructionNode {
    LightBounds bounds;
    uint32_t left;  // kInvalidIndex if leaf
    uint32_t right; // lightIdx if leaf
};

struct BuildStateContainer {
    Bounds3f allLightBounds;
    uint32_t nLights;

    LightBVHConstructionNode* dNodes;
    uint64_t* dMortonCodes;
    uint32_t* dClusterIndices;
    uint32_t* dParentIndices;
    uint32_t* nMergedClusters;
};

// Returns the difference between two morton codes.
// Larger values indicate larger difference.
static PBRT_GPU_INLINE uint64_t MortonCodeDelta(uint32_t L, uint32_t R, const uint32_t N,
    const uint64_t* mortonCodes) {
    if (L < 0 || R >= N) {
        return kInvalidIndex;
    }

    const uint64_t splitLevel = mortonCodes[L] ^ mortonCodes[R];
    if (splitLevel == 0) {
        // duplicate morton codes handling
        // by faking a split
        return L ^ (L + 1ull);
    }

    return splitLevel;
}

static PBRT_GPU_INLINE uint32_t FindParentIdx(uint32_t L, uint32_t R, uint32_t N,
    const uint64_t* mortonCodes) {
    if (L == 0 || (R != N && MortonCodeDelta(L - 1, L, N, mortonCodes) > MortonCodeDelta(R, R + 1, N, mortonCodes))) {
        return R;
    }
    
    return L - 1;
}

static PBRT_GPU_INLINE uint32_t EncodeRelativeOffset(uint32_t idx, uint32_t neighbor) {
    const uint32_t offset = neighbor - idx - 1;
    return offset << 1;
}

static PBRT_GPU_INLINE uint32_t DecodeRelativeOffset(const uint32_t idx, const uint32_t offset) {
    const uint32_t originalOffset = (offset >> 1) + 1;
    const uint32_t xor = offset ^ idx;
    return (xor & 1) == 0 ? idx + originalOffset : idx - originalOffset;
}

PBRT_GPU uint32_t FindNearestNeighbor(uint32_t nLights, uint32_t clusterIdx,
    uint8_t laneWarpIdx, const BuildStateContainer& buildState) {

    LightBounds clusterBounds;
    if (laneWarpIdx < nLights && clusterIdx != kInvalidIndex) {
        clusterBounds = buildState.dNodes[clusterIdx].bounds;
    }

    DCHECK_EQ(laneWarpIdx < nLights, clusterIdx != kInvalidIndex);
    const bool active = laneWarpIdx < nLights && clusterIdx != kInvalidIndex; 

    uint32_t minCostIdx = kInvalidIndex;
    for (uint32_t r = 1; r <= kSearchRadius; ++r) {
        uint32_t neighborIdx = laneWarpIdx + r;
        uint32_t neighborClusterIdx = __shfl_sync(kFullMask, clusterIdx, neighborIdx);

        uint32_t newCostIdx0 = kInvalidIndex;
        uint32_t newCostIdx1 = kInvalidIndex;
        if (neighborIdx < nLights) {
            LightBounds neighborBounds = buildState.dNodes[neighborClusterIdx].bounds;
            neighborBounds = Union(neighborBounds, clusterBounds);

            const float newCost = BVHLightSampler::EvaluateCost(neighborBounds);
            uint32_t newCostInt = __float_as_uint(newCost);

            const uint32_t encode0 = EncodeRelativeOffset(laneWarpIdx, neighborIdx);
            const uint32_t encode1 = (newCostInt << 1) & kEncodeMask;
            newCostIdx0 = encode1 | encode0 | (laneWarpIdx & 1);
            newCostIdx1 = encode1 | encode0 | ((neighborIdx & 1) ^ 1);
        }
        
        minCostIdx = std::min(minCostIdx, newCostIdx0);

        uint32_t neighborMinCostIdx = __shfl_sync(kFullMask, minCostIdx, neighborIdx);
        neighborMinCostIdx = std::min(neighborMinCostIdx, newCostIdx1);

        minCostIdx = __shfl_sync(kFullMask, neighborMinCostIdx, laneWarpIdx - r);
    }

    uint32_t decodedNN = kInvalidIndex;
    if (active && minCostIdx != kInvalidIndex) {
        uint32_t unmasked = minCostIdx & kDecodeMask;
        decodedNN = DecodeRelativeOffset(laneWarpIdx, unmasked);
    }
    
    return decodedNN;
}

struct ClusterPack {
    uint32_t NNidx;
    uint32_t clusterIdx;
};

PBRT_GPU uint32_t MergeClusters(uint32_t nLights, uint32_t nearestNeighborIdx, uint32_t& clusterIdx,
    const uint8_t laneWarpIdx, const BuildStateContainer& buildState) {

    uint32_t neighborNNIdx = __shfl_sync(kFullMask, nearestNeighborIdx, nearestNeighborIdx);
    uint32_t neighborClusterIdx = __shfl_sync(kFullMask, clusterIdx, nearestNeighborIdx); 
    
    const bool laneActive = laneWarpIdx < nLights;
    const bool mutual = laneActive && laneWarpIdx == neighborNNIdx;
    const bool merge = mutual && laneWarpIdx < nearestNeighborIdx;

    uint32_t mergeMask = __ballot_sync(kFullMask, merge);
    uint32_t mergeCount = __popc(mergeMask);

    uint32_t baseIdx = kInvalidIndex;
    if (laneWarpIdx == 0) {
        baseIdx = atomicAdd(buildState.nMergedClusters, mergeCount);
    }
    baseIdx = __shfl_sync(kFullMask, baseIdx, 0);

    uint32_t relativeIdx = __popc(mergeMask << (kWarpSize - laneWarpIdx));

    if (merge) {
        LightBounds clusterBounds = buildState.dNodes[clusterIdx].bounds;
        LightBounds neighborBounds = buildState.dNodes[neighborClusterIdx].bounds;
        clusterBounds = Union(clusterBounds, neighborBounds);

        LightBVHConstructionNode node;
        node.bounds = clusterBounds;
        node.left = clusterIdx;
        node.right = neighborClusterIdx;
        clusterIdx = baseIdx + relativeIdx;
        buildState.dNodes[clusterIdx] = node;
    }

    uint32_t validMask = __ballot_sync(kFullMask, merge || !mutual);

    // Compact the clusters by removing duplicate positions of successful merge
    int32_t shift = __fns(validMask, 0, laneWarpIdx + 1);

    clusterIdx = __shfl_sync(kFullMask, clusterIdx, shift);
    if (shift == -1) {
        clusterIdx = kInvalidIndex;
    }

    return nLights - mergeCount;
}

PBRT_GPU void PlocMerge(const uint32_t start, const uint32_t nLeft, const uint32_t nRight, const uint32_t threshold,
    uint32_t clusterIdx, const uint8_t laneWarpIdx, const BuildStateContainer& buildState) {
    
    const uint32_t nLightsInCurrentStep = nLeft + nRight;

    uint32_t nLightsToProcess = nLightsInCurrentStep;
    while (nLightsToProcess > threshold) {
        uint32_t nearestNeighbor = FindNearestNeighbor(nLightsToProcess, clusterIdx, laneWarpIdx, buildState);
        nLightsToProcess = MergeClusters(nLightsToProcess, nearestNeighbor, clusterIdx, laneWarpIdx, buildState);
    }

    // store indices
    if (laneWarpIdx < nLightsInCurrentStep) {
        buildState.dClusterIndices[start + laneWarpIdx] = clusterIdx;
    }

    // memory fence so the later read of clusterIdx is correct
    __threadfence();
}

PBRT_GPU_INLINE bool LoadIndex(uint32_t& clusterIdx, const uint32_t start, const uint32_t end,
    const uint32_t offset, uint32_t* clusterIndices, const uint8_t laneWarpId) {

    uint32_t index = laneWarpId - offset;
    bool validLaneIdx = index < std::min(end - start, kHalfWarp);

    if (validLaneIdx)
        clusterIdx = clusterIndices[start + index];

    return validLaneIdx;
}


__global__ void HplocOuterLoop(BuildStateContainer buildState) {
    uint32_t memStart = blockIdx.x * blockDim.x;
    uint32_t tid = memStart + threadIdx.x;

    uint8_t laneWarpId = threadIdx.x & (kWarpSize - 1);

    uint32_t leftIdx = tid;
    uint32_t rightIdx = tid;
    uint32_t splitIdx = 0; 

    bool laneActive = (tid < buildState.nLights);
    while (__ballot_sync(kFullMask, laneActive)) {
        if (laneActive) {
            uint32_t prevIdx = 0;
            uint32_t sibling = FindParentIdx(leftIdx, rightIdx, buildState.nLights, buildState.dMortonCodes);
            if (sibling == rightIdx) {
                // I am the left child
                prevIdx = atomicExch(&buildState.dParentIndices[rightIdx], leftIdx);

                if (prevIdx != kInvalidIndex) {
                    splitIdx = rightIdx + 1;
                    rightIdx = prevIdx;
                }
            } else {
                // I am the right child
                prevIdx = atomicExch(&buildState.dParentIndices[leftIdx - 1], rightIdx);

                if (prevIdx != kInvalidIndex) {
                    splitIdx = leftIdx;
                    leftIdx = prevIdx;
                }
            }
            if (prevIdx == kInvalidIndex)
                laneActive = false;
        }

        const uint32_t size = rightIdx - leftIdx + 1;
        bool isTreeRoot = laneActive && size == buildState.nLights;

        uint32_t warpMask = __ballot_sync(kFullMask,
            laneActive && (size > kHalfWarp) || isTreeRoot);

        while (warpMask) {
            // index of the first 1 in the integer - 1
            // number of trailing zeros.
            uint8_t selectedLaneIdx = __ffs(warpMask) - 1;

            uint32_t startL = __shfl_sync(kFullMask, leftIdx, selectedLaneIdx);
            uint32_t endR = __shfl_sync(kFullMask, rightIdx, selectedLaneIdx) + 1;
            uint32_t endL = __shfl_sync(kFullMask, splitIdx, selectedLaneIdx);
            uint32_t startR = endL;
            uint32_t threshold =
                __shfl_sync(kFullMask, isTreeRoot, selectedLaneIdx) ? 1 : kHalfWarp;

            uint32_t clusterIdx = kInvalidIndex;
            const bool isLeftValidIndex = LoadIndex(clusterIdx, startL, endL, 0, buildState.dClusterIndices, laneWarpId);
            uint32_t nLeftClusters = __popc(__ballot_sync(kFullMask, isLeftValidIndex && clusterIdx != kInvalidIndex));

            const bool isRightValidIndex = LoadIndex(clusterIdx, startR, endR, nLeftClusters, buildState.dClusterIndices, laneWarpId);
            uint32_t nRightClusters = __popc(__ballot_sync(kFullMask, isRightValidIndex && clusterIdx != kInvalidIndex));

            PlocMerge(startL, nLeftClusters, nRightClusters, threshold, clusterIdx, laneWarpId, buildState);
            warpMask &= (warpMask - 1);
        }
    }
}

void BuildNodes(BuildStateContainer& buildState) {
    const char* description = "Build Nodes";
#ifdef NVTX
    nvtxRangePush(description);
#endif
    
#ifdef PBRT_DEBUG_BUILD
    LOG_VERBOSE("Launching %s", description);
#endif

    int blockSize = GetBlockSize(description, HplocOuterLoop);
    {
        KernelTimerWrapper timer(GetProfilerEvents(description, ProfilerKernelGroup::HPLOC));
        int gridSize = (buildState.nLights + blockSize - 1) / blockSize;
        HplocOuterLoop<<<gridSize, blockSize>>>(buildState);
    }

#ifdef PBRT_DEBUG_BUILD
    GPUWait();
    LOG_VERBOSE("Post-sync %s", description);
#endif

#ifdef NVTX
    nvtxRangePop();
#endif
}

__global__ void AssignMortonCodes(BuildStateContainer buildState, LightBVHBuildContainer* dLightsContainer,
    uint8_t ax0, uint8_t ax1, uint8_t ax2) {
    uint32_t memStart = blockIdx.x * blockDim.x;
    uint32_t tid = memStart + threadIdx.x;

    if (tid < buildState.nLights) {
        LightBVHBuildContainer cont = dLightsContainer[tid];
        LightBVHConstructionNode leaf{cont.bounds, kInvalidIndex, cont.index};

        Point3f centroid = cont.bounds.Centroid();
        Vector3f offset = buildState.allLightBounds.Offset(centroid);

        Point3f position = {offset[ax0], offset[ax1], offset[ax2]};
        Vector3f direction = Normalize(cont.bounds.w);

        buildState.dClusterIndices[tid] = tid;
        buildState.dMortonCodes[tid] = EncodeExtendedMorton5(position, direction);
        buildState.dNodes[tid] = leaf;
    }
}

void Initialize(BuildStateContainer& buildState, LightBVHBuildContainer* dLightsContainer) {

    uint8_t axisOrder[3] = {0, 1, 2};
    const Vector3f diagonal = buildState.allLightBounds.Diagonal();

    if (diagonal[axisOrder[0]] < diagonal[axisOrder[1]])
        pstd::swap(axisOrder[0], axisOrder[1]);
    if (diagonal[axisOrder[1]] < diagonal[axisOrder[2]])
        pstd::swap(axisOrder[1], axisOrder[2]);
    if (diagonal[axisOrder[0]] < diagonal[axisOrder[1]])
        pstd::swap(axisOrder[0], axisOrder[1]);

    const char* description = "Assign Morton Codes";

    #ifdef NVTX
    nvtxRangePush(description);
#endif
    
#ifdef PBRT_DEBUG_BUILD
    LOG_VERBOSE("Launching %s", description);
#endif

    int blockSize = GetBlockSize(description, HplocOuterLoop);
    {
        KernelTimerWrapper timer(GetProfilerEvents(description, ProfilerKernelGroup::HPLOC));
        int gridSize = (buildState.nLights + blockSize - 1) / blockSize;
        AssignMortonCodes<<<gridSize, blockSize>>>(buildState, dLightsContainer, axisOrder[0], axisOrder[1], axisOrder[2]);
    }

#ifdef PBRT_DEBUG_BUILD
    GPUWait();
    LOG_VERBOSE("Post-sync %s", description);
#endif

#ifdef NVTX
    nvtxRangePop();
#endif

    uint64_t* dMortonCodesSorted = GPUAllocAsync<uint64_t>(buildState.nLights);
    uint32_t* dClusterIndicesSorted = GPUAllocAsync<uint32_t>(buildState.nLights);

    void* dTempStorage = nullptr;
    size_t tempStorageBytes = 0;
    uint32_t beginBit = 1, endBit = 64;
    {
        KernelTimerWrapper timer(GetProfilerEvents("Radix Sort Morton keys", ProfilerKernelGroup::HPLOC));
        cub::DeviceRadixSort::SortPairs(dTempStorage, tempStorageBytes, 
            buildState.dMortonCodes, dMortonCodesSorted,
            buildState.dClusterIndices, dClusterIndicesSorted, buildState.nLights, beginBit, endBit);

        dTempStorage = GPUAllocAsync<uint8_t>(tempStorageBytes);
        cub::DeviceRadixSort::SortPairs(dTempStorage, tempStorageBytes, 
            buildState.dMortonCodes, dMortonCodesSorted,
            buildState.dClusterIndices, dClusterIndicesSorted, buildState.nLights, beginBit, endBit);
    }

    GPUFreeAsync(dTempStorage);
    GPUFreeAsync(buildState.dMortonCodes);
    GPUFreeAsync(buildState.dClusterIndices);

    buildState.dClusterIndices = dClusterIndicesSorted;
    buildState.dMortonCodes = dMortonCodesSorted;
}

bool BVHLightSampler::buildBVHGPU(std::vector<LightBVHBuildContainer>& bvhLights) {
    if (bvhLights.size() < 100) {
        return false;
    }

    uint32_t nNodes = bvhLights.size() * 2 - 1;

    BuildStateContainer buildState;
    buildState.nLights = static_cast<uint32_t>(bvhLights.size());
    buildState.nMergedClusters = 0;
    buildState.allLightBounds = m_allLightBounds;
    buildState.dNodes = GPUAllocAsync<LightBVHConstructionNode>(nNodes);
    buildState.dClusterIndices = GPUAllocAsync<uint32_t>(buildState.nLights);
    buildState.dParentIndices = GPUAllocAsync<uint32_t>(buildState.nLights);
    buildState.dMortonCodes = GPUAllocAsync<uint64_t>(buildState.nLights);
    buildState.nMergedClusters = GPUAllocAsync<uint32_t>(1);

    LightBVHBuildContainer* dLightsContainer = GPUAllocAsync<LightBVHBuildContainer>(buildState.nLights);
    GPUCopyToDevice(dLightsContainer, bvhLights.data(), bvhLights.size());
    GPUCopyToDevice(buildState.nMergedClusters, &buildState.nLights, 1);
    GPUMemsetAsync(buildState.dParentIndices, kInvalidIndex, sizeof(uint32_t) * buildState.nLights);

    // uint8_t axisOrder[3] = {0, 1, 2};
    // const Vector3f diagonal = m_allLightBounds.Diagonal();
    // 
    // if (diagonal[axisOrder[0]] < diagonal[axisOrder[1]])
    //     pstd::swap(axisOrder[0], axisOrder[1]);
    // if (diagonal[axisOrder[1]] < diagonal[axisOrder[2]])
    //     pstd::swap(axisOrder[1], axisOrder[2]);
    // if (diagonal[axisOrder[0]] < diagonal[axisOrder[1]])
    //     pstd::swap(axisOrder[0], axisOrder[1]);
    // 
    // std::vector<uint64_t> mortonCodesCPU(bvhLights.size());
    // for (size_t i = 0; i < bvhLights.size(); ++i) {
    //     const LightBVHBuildContainer& cont(bvhLights[i]);
    //     Vector3f offset = m_allLightBounds.Offset(cont.bounds.Centroid());
    // 
    //     Point3f position = {offset[axisOrder[0]], offset[axisOrder[1]], offset[axisOrder[2]]};
    //     Vector3f direction = Normalize(cont.bounds.w);
    // 
    //     mortonCodesCPU[i] = EncodeExtendedMorton5(position, direction);
    // }
    // std::sort(mortonCodesCPU.begin(), mortonCodesCPU.end());

    Initialize(buildState, dLightsContainer);
    GPUFreeAsync(dLightsContainer);
    dLightsContainer = nullptr;

    std::vector<uint64_t> mortonCodes(bvhLights.size());
    GPUCopyToHost(mortonCodes.data(), buildState.dMortonCodes, bvhLights.size()); 

    BuildNodes(buildState);

    GPUFreeAsync(buildState.dMortonCodes);
    GPUFreeAsync(buildState.dClusterIndices);
    GPUFreeAsync(buildState.dParentIndices);
    GPUFreeAsync(buildState.dNodes);
    GPUFreeAsync(buildState.nMergedClusters);

    ReportKernelStats(ProfilerKernelGroup::HPLOC);

    return false;
}

}
