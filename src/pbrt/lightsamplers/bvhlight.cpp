#include "bvhlight.h"

#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>
#include <vector>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/lighttreebuilder.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>

#include <algorithm>
#include <array>

#include <cub/device/device_radix_sort.cuh>
#endif //PBRT_BUILD_GPU_RENDERER

namespace pbrt{

#ifdef PBRT_BUILD_GPU_RENDERER

struct LightBVHCostEvaluator {
    PBRT_GPU Float operator()(const LightBounds &bounds) const {
        return BVHLightSampler::EvaluateCost(bounds);
    }
};

class BVHLightTreeBuilder final : public LightTreeBuilderGPU<uint64_t, LightBVHCostEvaluator> {
  public:
    explicit BVHLightTreeBuilder(const Bounds3f &bounds) : m_allLightBounds(bounds) {}

    bool Build(std::vector<LightBVHBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);
        MortonCodes() = GetSortedMortonCodes(State(), MortonCodes(), lights);
        BuildNodes(LightBVHCostEvaluator());
        return true;
    }

    void FlattenTree(const pstd::vector<Light>& lights, pstd::vector<LightBVHNode>& nodes, HashMap<Light, uint32_t>& bitTrailContainer) {
        const LightTreeBuildState &state(State());
        if (state.nLights == 0)
            return;

        uint32_t nNodes = 0;
        uint32_t rootIndex = 0;
        GPUCopyToHost(&nNodes, state.nMergedClusters, 1);
        GPUCopyToHost(&rootIndex, state.dClusterIndices, 1);
        std::vector<LightTreeConstructionNodeGPU> hostNodes(nNodes);
        GPUCopyToHost(hostNodes.data(), state.dNodes, nNodes);

        nodes.reserve(nNodes);

        FlattenNode(lights, hostNodes, rootIndex, 0, 0, nodes, bitTrailContainer);
    }

    static uint64_t* GetSortedMortonCodes(LightTreeBuildState& buildState, uint64_t* dMortonCodes, const std::vector<LightBVHBuildContainer> &lights) {
        LightTreeBuildState localState = buildState;
        std::array<uint8_t, 3> ax = DetermineAxisOrder(localState.allLightBounds);

        LightBVHBuildContainer* dLightsContainer = GPUAllocAsync<LightBVHBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        GPUParallelFor("Assign Morton Codes", ProfilerKernelGroup::HPLOC, localState.nLights, [=] PBRT_GPU(int idx) {
            LightBVHBuildContainer cont = dLightsContainer[idx];
            LightTreeConstructionNodeGPU leaf{cont.bounds, kInvalidIndex, cont.index};
            Point3f centroid = cont.bounds.Centroid();
            Vector3f offset = buildState.allLightBounds.Offset(centroid);

            Point3f position = {offset[ax[0]], offset[ax[1]], offset[ax[2]]};
            Vector3f direction = Normalize(cont.bounds.w);

            dMortonCodes[idx] = EncodeExtendedMorton5(position, direction);

            localState.dClusterIndices[idx] = idx;
            localState.dNodes[idx] = leaf;
        });

        GPUFreeAsync(dLightsContainer);
        dLightsContainer = nullptr;

        uint64_t *dMortonCodesSorted = GPUAllocAsync<uint64_t>(localState.nLights);
        uint32_t *dClusterIndicesSorted = GPUAllocAsync<uint32_t>(localState.nLights);

        void *dTempStorage = nullptr;
        size_t tempStorageBytes = 0;
        uint32_t beginBit = 1, endBit = 64;

        const char *description = "Radix Sort Morton keys";
        {
            KernelTimerWrapper timer(GetProfilerEvents(description, ProfilerKernelGroup::HPLOC));
            cub::DeviceRadixSort::SortPairs(dTempStorage, tempStorageBytes, dMortonCodes,
                dMortonCodesSorted, localState.dClusterIndices, dClusterIndicesSorted,
                localState.nLights, beginBit, endBit);

            dTempStorage = GPUAllocAsync<uint8_t>(tempStorageBytes);

            cub::DeviceRadixSort::SortPairs(dTempStorage, tempStorageBytes, dMortonCodes,
                dMortonCodesSorted, localState.dClusterIndices, dClusterIndicesSorted,
                localState.nLights, beginBit, endBit);
        }

        GPUFreeAsync(dTempStorage);
        GPUFreeAsync(dMortonCodes);
        GPUFreeAsync(buildState.dClusterIndices);

        buildState.dClusterIndices = dClusterIndicesSorted;

        return dMortonCodesSorted;
    }


  private:
    uint32_t FlattenNode(const pstd::vector<Light>& lights, const std::vector<LightTreeConstructionNodeGPU>& gpuNodes, uint32_t nodeIdx,
        uint32_t bitTrail, uint32_t depth, pstd::vector<LightBVHNode>& nodes, HashMap<Light, uint32_t>& bitTrailContainer) const {
         const LightTreeConstructionNodeGPU &gpuNode = gpuNodes[nodeIdx];
         CompactLightBounds cb(gpuNode.bounds, m_allLightBounds);

         const bool isLeaf = gpuNode.left == kInvalidIndex;
         if (isLeaf) {
             int flatLeafIndex = nodes.size();
             int lightIndex = gpuNode.right;
             nodes.push_back(LightBVHNode::MakeLeaf(lightIndex, cb));
             bitTrailContainer.Insert(lights[lightIndex], bitTrail);
             return flatLeafIndex;
         }

         // Allocate interior _LightBVHNode_ and recursively initialize children
         int flatNodeIndex = nodes.size();
         nodes.push_back(LightBVHNode());
         CHECK_LT(depth, 32);
         uint32_t child0 = FlattenNode(lights, gpuNodes, gpuNode.left, bitTrail, depth + 1, nodes, bitTrailContainer);
         DCHECK_EQ(flatNodeIndex + 1, child0);
         uint32_t child1 = FlattenNode(lights, gpuNodes, gpuNode.right, bitTrail | (1u << depth), depth + 1, nodes, bitTrailContainer);
         
         nodes[flatNodeIndex] = LightBVHNode::MakeInterior(child1, cb);
         return flatNodeIndex;
    }

    Bounds3f m_allLightBounds;
};

#endif  // PBRT_BUILD_GPU_RENDERER

///////////////////////////////////////////////////////////////////////////
// BVHLightSampler

STAT_MEMORY_COUNTER("Memory/Light BVH", lightBVHBytes);
STAT_INT_DISTRIBUTION("Integrator/Lights sampled per lookup", nLightsSampled);

// BVHLightSampler Method Definitions
BVHLightSampler::BVHLightSampler(pstd::span<const Light> lights, Allocator alloc)
    : m_lights(lights.begin(), lights.end(), alloc),
      m_infiniteLights(alloc),
      m_nodes(alloc),
      m_lightToBitTrail(alloc) {
    // Initialize _infiniteLights_ array and light BVH
    std::vector<LightBVHBuildContainer> bvhLights;
    for (size_t i = 0; i < lights.size(); ++i) {
        // Store $i$th light in either _infiniteLights_ or _bvhLights_
        Light light = lights[i];
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        if (!lightBounds)
            m_infiniteLights.push_back(light);
        else if (lightBounds->phi > 0) {
            bvhLights.emplace_back(*lightBounds, i);
            m_allLightBounds = Union(m_allLightBounds, lightBounds->bounds);
        }
    }

    if (!bvhLights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        bool buildOnGPU = buildBVHGPU(bvhLights);
        if (!buildOnGPU)
#endif
            buildBVH(bvhLights, 0, bvhLights.size(), 0, 0);
    }
        
    lightBVHBytes += m_nodes.size() * sizeof(LightBVHNode) +
                     m_lightToBitTrail.capacity() * (sizeof(Light) + sizeof(uint32_t)) +
                     lights.size() * sizeof(Light) +
                     m_infiniteLights.size() * sizeof(Light);
}

LightBVHBuildContainer BVHLightSampler::buildBVH(
    std::vector<LightBVHBuildContainer> &bvhLights, int start, int end,
    uint32_t bitTrail, int depth) {
    DCHECK_LT(start, end);
    // Initialize leaf node if only a single light remains
    if (end - start == 1) {
        int nodeIndex = m_nodes.size();
        CompactLightBounds cb(bvhLights[start].bounds, m_allLightBounds);
        int lightIndex = bvhLights[start].index;
        m_nodes.push_back(LightBVHNode::MakeLeaf(lightIndex, cb));
        m_lightToBitTrail.Insert(m_lights[lightIndex], bitTrail);
        return {bvhLights[start].bounds, nodeIndex};
    }

    // Choose split dimension and position using modified SAH
    // Compute bounds and centroid bounds for lights
    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        const LightBounds &lb = bvhLights[i].bounds;
        bounds = Union(bounds, lb.bounds);
        centroidBounds = Union(centroidBounds, lb.Centroid());
    }

    Float minCost = Infinity;
    int minCostSplitBucket = -1, minCostSplitDim = -1;
    constexpr int nBuckets = 12;
    for (int dim = 0; dim < 3; ++dim) {
        // Compute minimum cost bucket for splitting along dimension _dim_
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim])
            continue;
        // Compute _LightBounds_ for each bucket
        LightBounds bucketLightBounds[nBuckets];
        for (int i = start; i < end; ++i) {
            Point3f pc = bvhLights[i].bounds.Centroid();
            int b = nBuckets * centroidBounds.Offset(pc)[dim];
            if (b == nBuckets)
                b = nBuckets - 1;
            DCHECK_GE(b, 0);
            DCHECK_LT(b, nBuckets);
            bucketLightBounds[b] = Union(bucketLightBounds[b], bvhLights[i].bounds);
        }

        // Compute costs for splitting lights after each bucket
        Float cost[nBuckets - 1];
        for (int i = 0; i < nBuckets - 1; ++i) {
            // Find _LightBounds_ for lights below and above bucket split
            LightBounds b0, b1;
            for (int j = 0; j <= i; ++j)
                b0 = Union(b0, bucketLightBounds[j]);
            for (int j = i + 1; j < nBuckets; ++j)
                b1 = Union(b1, bucketLightBounds[j]);

            // Compute final light split cost for bucket
            cost[i] = EvaluateCost(b0, bounds, dim) + EvaluateCost(b1, bounds, dim);
        }

        // Find light split that minimizes SAH metric
        for (int i = 1; i < nBuckets - 1; ++i) {
            if (cost[i] > 0 && cost[i] < minCost) {
                minCost = cost[i];
                minCostSplitBucket = i;
                minCostSplitDim = dim;
            }
        }
    }

    // Partition lights according to chosen split
    int mid;
    if (minCostSplitDim == -1)
        mid = (start + end) / 2;
    else {
        const auto *pmid = std::partition(
            &bvhLights[start], &bvhLights[end - 1] + 1,
            [=](const LightBVHBuildContainer &cont) {
                int b = nBuckets *
                        centroidBounds.Offset(cont.bounds.Centroid())[minCostSplitDim];
                if (b == nBuckets)
                    b = nBuckets - 1;
                DCHECK_GE(b, 0);
                DCHECK_LT(b, nBuckets);
                return b <= minCostSplitBucket;
            });
        mid = pmid - &bvhLights[0];
        if (mid == start || mid == end)
            mid = (start + end) / 2;
        DCHECK(mid > start && mid < end);
    }

    // Allocate interior _LightBVHNode_ and recursively initialize children
    int nodeIndex = m_nodes.size();
    m_nodes.push_back(LightBVHNode());
    CHECK_LT(depth, 64);
    LightBVHBuildContainer child0 =
        buildBVH(bvhLights, start, mid, bitTrail, depth + 1);
    DCHECK_EQ(nodeIndex + 1, child0.index);
    LightBVHBuildContainer child1 =
        buildBVH(bvhLights, mid, end, bitTrail | (1u << depth), depth + 1);

    // Initialize interior node and return node index and bounds
    LightBounds lb = Union(child0.bounds, child1.bounds);
    CompactLightBounds cb(lb, m_allLightBounds);
    m_nodes[nodeIndex] = LightBVHNode::MakeInterior(child1.index, cb);
    return {lb, nodeIndex};
}

#ifdef PBRT_BUILD_GPU_RENDERER
bool BVHLightSampler::buildBVHGPU(
    std::vector<LightBVHBuildContainer> &bvhLights) {
    if (bvhLights.size() < 100)
        return false;

    BVHLightTreeBuilder builder(m_allLightBounds);
    if (!builder.Build(bvhLights))
        return false;

    builder.FlattenTree(m_lights, m_nodes, m_lightToBitTrail);
    return true;
}
#endif

std::string BVHLightSampler::ToString() const {
    return StringPrintf("[ BVHLightSampler nodes: %s ]", m_nodes);
}

std::string LightBVHNode::ToString() const {
    return StringPrintf(
        "[ LightBVHNode lightBounds: %s childOrLightIndex: %d isLeaf: %d ]", lightBounds,
        childOrLightIndex, isLeaf);
}

}
