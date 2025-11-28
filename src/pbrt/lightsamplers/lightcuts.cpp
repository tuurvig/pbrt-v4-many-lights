#include "lightcuts.h"

#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>

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

struct LightcutsTreeNodeCostEvaluator {
    LightcutsTreeNodeCostEvaluator(Bounds3f bounds, bool isPoint) :
        sceneBoundsDiagonalSqr(LengthSquared(bounds.Diagonal())),
        isPoint(isPoint) {}

    PBRT_GPU Float operator()(const LightBounds &bounds) const {
        return LightcutsLightSampler::EvaluateCost(bounds, sceneBoundsDiagonalSqr, isPoint);
    }

    Float sceneBoundsDiagonalSqr;
    bool isPoint;
};

class LightcutsTreeBuilderGPU final : public LightTreeBuilderGPU<uint32_t, LightcutsTreeNodeCostEvaluator> {
  public:
    explicit LightcutsTreeBuilderGPU(const Bounds3f &bounds, bool isPoint) : m_allLightBounds(bounds), m_isPoint(isPoint) {}

    bool Build(std::vector<LightBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);
        MortonCodes() = GetSortedMortonCodes(State(), MortonCodes(), lights);
        BuildNodes(LightcutsTreeNodeCostEvaluator(m_allLightBounds, m_isPoint));
        return true;
    }

    void FlattenTree(LightcutsTree& tree, HashMap<Light, LightLocation>& bitTrailContainer, float& u) {
        const LightTreeBuildState &state(State());
        if (state.nLights == 0)
            return;

        uint32_t nNodes = 0;
        uint32_t rootIndex = 0;
        GPUCopyToHost(&nNodes, state.nMergedClusters, 1);
        GPUCopyToHost(&rootIndex, state.dClusterIndices, 1);
        std::vector<LightTreeConstructionNodeGPU> hostNodes(nNodes);
        GPUCopyToHost(hostNodes.data(), state.dNodes, nNodes);

        tree.nodes.reserve(nNodes);

        uint32_t rootRepresentant = 0;
        FlattenNode(tree, hostNodes, bitTrailContainer, rootIndex, 0, 0, rootRepresentant, u);
    }

    static uint32_t* GetSortedMortonCodes(LightTreeBuildState& buildState, uint32_t* dMortonCodes, const std::vector<LightBuildContainer>& lights) {
        LightTreeBuildState localState = buildState;
        std::array<uint8_t, 3> ax = DetermineAxisOrder(localState.allLightBounds);

        LightBuildContainer* dLightsContainer = GPUAllocAsync<LightBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        GPUParallelFor("Assign Morton Codes", ProfilerKernelGroup::HPLOC, localState.nLights, [=] PBRT_GPU(int idx) {
            LightBuildContainer cont = dLightsContainer[idx];
            LightTreeConstructionNodeGPU leaf{cont.bounds, kInvalidIndex, kInvalidIndex};
            Point3f centroid = cont.bounds.Centroid();
            Vector3f offset = buildState.allLightBounds.Offset(centroid);

            Point3f position = {offset[ax[0]], offset[ax[1]], offset[ax[2]]};

            Float x = QuantizeUnitToBitRange(position.x, 10);
            Float y = QuantizeUnitToBitRange(position.y, 10);
            Float z = QuantizeUnitToBitRange(position.z, 10);

            dMortonCodes[idx] = EncodeMorton3(x, y, z);
            localState.dClusterIndices[idx] = idx;
            localState.dNodes[idx] = leaf;
        });

        GPUFreeAsync(dLightsContainer);
        dLightsContainer = nullptr;

        uint32_t *dMortonCodesSorted = GPUAllocAsync<uint32_t>(localState.nLights);
        uint32_t *dClusterIndicesSorted = GPUAllocAsync<uint32_t>(localState.nLights);

        void *dTempStorage = nullptr;
        size_t tempStorageBytes = 0;
        uint32_t beginBit = 1, endBit = 32;

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
    uint32_t FlattenNode(LightcutsTree& tree, const std::vector<LightTreeConstructionNodeGPU>& gpuNodes,
         HashMap<Light, LightLocation>& bitTrailContainer, uint32_t nodeIdx, uint32_t bitTrail, uint32_t depth, uint32_t& representantIdx, float& u) const {

         const LightTreeConstructionNodeGPU &gpuNode = gpuNodes[nodeIdx];
         CompactLightBounds cb(gpuNode.bounds, m_allLightBounds);

         const bool isLeaf = gpuNode.left == kInvalidIndex;
         if (isLeaf) {
             int flatLeafIndex = tree.nodes.size();
             int lightIndex = nodeIdx;
             representantIdx = lightIndex;
             tree.nodes.push_back(LightcutsTreeNode::MakeLeaf(lightIndex, representantIdx, cb));
             bitTrailContainer.Insert(tree.lights[lightIndex], {tree.isPoint, bitTrail});
             return flatLeafIndex;
         }


         // Allocate interior and recursively initialize children
         size_t flatNodeIndex = tree.nodes.size();
         tree.nodes.emplace_back();
         CHECK_LT(depth, 32);
         uint32_t representantLeftIdx = 0, representantRightIdx = 0;
         uint32_t child0 = FlattenNode(tree, gpuNodes, bitTrailContainer, gpuNode.left, bitTrail, depth + 1, representantLeftIdx, u);
         DCHECK_EQ(flatNodeIndex + 1, child0);
         uint32_t child1 = FlattenNode(tree, gpuNodes, bitTrailContainer, gpuNode.right, bitTrail | (1u << depth), depth + 1, representantRightIdx, u);
         
         Float intensities[2] = {gpuNode.bounds.phi, gpuNode.bounds.phi};
         Float nodePMF;
         int child = SampleDiscrete(intensities, u, &nodePMF, &u);
         representantIdx = (child == 0) ? representantLeftIdx : representantRightIdx;
         
         tree.nodes[flatNodeIndex] = LightcutsTreeNode::MakeInterior(child1, representantIdx, cb);
         return flatNodeIndex;
    }

private:
    Bounds3f m_allLightBounds;
    bool m_isPoint;
};

#endif  // PBRT_BUILD_GPU_RENDERER
///////////////////////////////////////////////////////////////////////////
// LightcutsLightSampler

STAT_MEMORY_COUNTER("Memory/Lightcuts LightTree", lightCutsLightTreeBytes);

constexpr uint32_t infiniteLightsIndex = 2;
constexpr uint32_t otherLightsIndex = 3;

LightcutsTree::LightcutsTree(bool isPoint, Allocator alloc) 
    : lights(alloc), nodes(alloc), isPoint(isPoint) {}

LightcutsLightSampler::LightcutsLightSampler(pstd::span<const Light> lights, Allocator alloc, Float threshold) 
    : m_pointTree(true, alloc), m_spotTree(false, alloc), m_otherLights(alloc), m_lightToLocation(alloc), m_threshold(threshold) {
    
    // Initialize infiniteLights array and lightcuts lights
    std::vector<LightBuildContainer> pointLights, spotLights;

    for (size_t i = 0; i < lights.size(); ++i) {
        Light light = lights[i];
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        
        if (!lightBounds) {
            uint32_t index = m_infiniteLights.size();
            m_infiniteLights.push_back(light);
            m_lightToLocation.Insert(light, {infiniteLightsIndex, index});

        } else if (lightBounds->phi > 0) {
            if (light.Is<PointLight>()) {
                pointLights.emplace_back(*lightBounds, light);
                m_pointTree.allLightBounds = Union(m_pointTree.allLightBounds, lightBounds->bounds);

            } else if (light.Is<SpotLight>()) {
                spotLights.emplace_back(*lightBounds, light);
                m_spotTree.allLightBounds = Union(m_spotTree.allLightBounds, lightBounds->bounds);

            } else {
                uint32_t index = m_otherLights.size();
                m_otherLights.push_back(light);
                m_lightToLocation.Insert(light, {otherLightsIndex, index});
                m_otherLightsPower += lightBounds->phi;
            }
        }
    }

    RNG rng;
    Float u = rng.Uniform<Float>();
    if (!m_pointTree.lights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        bool buildOnGPU = buildLightTreeGPU(pointLights, m_pointTree, m_lightToLocation, u);
        if (!buildOnGPU)
#endif
            buildLightTree(pointLights, m_pointTree, 0, pointLights.size(), 0, 0, u);
    }

    if (!m_spotTree.lights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        bool buildOnGPU = buildLightTreeGPU(spotLights, m_spotTree, m_lightToLocation, u);
        if (!buildOnGPU)
#endif
            buildLightTree(spotLights, m_spotTree, 0, spotLights.size(), 0, 0, u);
    }

    lightCutsLightTreeBytes += (m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size() + m_infiniteLights.size()) * sizeof(Light) + 
                               (m_pointTree.nodes.size() + m_spotTree.nodes.size()) * sizeof(LightcutsTreeNode) +
                               m_lightToLocation.capacity() * (sizeof(Light) + sizeof(LightLocation));
}

TreeNodeBuildSuccess LightcutsLightSampler::buildLightTree(std::vector<LightBuildContainer>& lightcutsLights,
    LightcutsTree& tree, int start, int end, uint32_t bitTrail, int depth, float& u) {

    DCHECK_LT(start, end);

    if (end - start == 1) {
        const LightBuildContainer& leaf(lightcutsLights[start]);
        CompactLightBounds cb(leaf.bounds, tree.allLightBounds);

        int nodeIndex = tree.nodes.size();
        int lightIndex = tree.lights.size();

        tree.lights.emplace_back(leaf.light);
        tree.nodes.emplace_back(LightcutsTreeNode::MakeLeaf(lightIndex, nodeIndex, cb));
        m_lightToLocation.Insert(leaf.light, {tree.isPoint, bitTrail});
        return {leaf.bounds, nodeIndex, nodeIndex};
    }

    // Choose split dimension and position using Similarity Metric
    // Compute bounds and centroid bounds for lights
    Bounds3f centroidBounds;
    for (int i = start; i < end; ++i) {
        const LightBounds& lb(lightcutsLights[i].bounds);
        centroidBounds = Union(centroidBounds, lb.Centroid());
    }

    Float minCost = Infinity;
    int minCostSplitBucket = -1;
    int minCostSplitDim = -1;

    constexpr int nBuckets = 16;
    for (int dim = 0; dim < 3; ++dim) {
        // Compute minimum cost bucket for splitting along dim
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]){
            continue;
        }

        LightBounds bucketLightBounds[nBuckets];
        for (int i = start; i < end; ++i) {
            Point3f pc = lightcutsLights[i].bounds.Centroid();
            int b = nBuckets * centroidBounds.Offset(pc)[dim];
            if (b == nBuckets){
                b = nBuckets - 1;
            }
            DCHECK_GE(b, 0);
            DCHECK_LT(b, nBuckets);
            bucketLightBounds[b] = Union(bucketLightBounds[b], lightcutsLights[i].bounds);
        }

        LightBounds leftBoundsSum[nBuckets], rightBoundsSum[nBuckets];
        leftBoundsSum[0] = bucketLightBounds[0];
        rightBoundsSum[nBuckets - 1] = bucketLightBounds[nBuckets - 1];

        for (int lower = 1, upper = nBuckets - 2; lower < nBuckets; ++lower, --upper) {
            LightBounds& leftBoundsPrefix(leftBoundsSum[lower]);
            LightBounds& rightBoundsPrefix(rightBoundsSum[upper]);

            const LightBounds& prevLeftBoundsPrefix(leftBoundsSum[lower - 1]);
            const LightBounds& prevRightBoundsPrefix(rightBoundsSum[upper + 1]);

            leftBoundsPrefix = Union(leftBoundsPrefix, prevLeftBoundsPrefix);
            rightBoundsPrefix = Union(rightBoundsPrefix, prevRightBoundsPrefix);
        }

        Float diagonalLenSqr = LengthSquared(tree.allLightBounds.Diagonal());
        for (int i = 0, max = nBuckets - 1; i < max; ++i) {
            const Float leftCost = EvaluateCost(leftBoundsSum[i], diagonalLenSqr, tree.isPoint);
            const Float rightCost = EvaluateCost(rightBoundsSum[i + 1], diagonalLenSqr, tree.isPoint);

            const Float cost = rightCost + leftCost;

            if (cost > 0 && cost < minCost) {
                minCost = cost;
                minCostSplitBucket = i;
                minCostSplitDim = dim;
            }
        }
    }

    // Partition lights according to chosen split
    int mid;
    if (minCostSplitDim == -1) {
        mid = (start + end) / 2;
    } else {
        const auto* pmid = std::partition(&lightcutsLights[start], &lightcutsLights[end - 1] + 1,
            [=](const LightBuildContainer& container) {
                int b = nBuckets * centroidBounds.Offset(container.bounds.Centroid())[minCostSplitDim];
                if (b == nBuckets) {
                    b = nBuckets - 1;
                }
                DCHECK_GE(b, 0);
                DCHECK_LT(b, nBuckets);
                return b <= minCostSplitBucket;
            });
        mid = pmid - &lightcutsLights[0];
        if (mid == start || mid == end) {
            mid = (start + end) / 2;
        }
        DCHECK(mid > start && mid < end);
    }

    // Allocate interior and recursively initialize children
    size_t nodeIndex = tree.nodes.size();
    tree.nodes.emplace_back();
    CHECK_LT(depth, 64);
    TreeNodeBuildSuccess left = buildLightTree(lightcutsLights, tree, start, mid, bitTrail, depth + 1, u);
    DCHECK_EQ(nodeIndex + 1, left.nodeIdx);
    TreeNodeBuildSuccess right = buildLightTree(lightcutsLights, tree, mid, end, bitTrail | (1u << depth), depth + 1, u);

    Float intensities[2] = {left.bounds.phi, right.bounds.phi};
    Float nodePMF;
    int child = SampleDiscrete(intensities, u, &nodePMF, &u);
    int successorIdx = (child == 0) ? left.representantIdx : right.representantIdx;

    LightBounds lb = Union(left.bounds, right.bounds);
    CompactLightBounds cb(lb, tree.allLightBounds);
    tree.nodes[nodeIndex] = LightcutsTreeNode::MakeInterior(right.nodeIdx, successorIdx, cb);
    return {lb, successorIdx, static_cast<int>(nodeIndex)};
}

pstd::optional<SampledLight> LightcutsLightSampler::SampleLightTree(const LightSampleContext& ctx, const LightcutsTree& tree, const BSDF* bsdf, Float pmf, Float u) const {
    int nodeIndex = 0;
    Point3f p = ctx.p();
    Normal3f n = ctx.ns;
    const LightcutsTreeNode* node = &tree.nodes[nodeIndex];

    Float treeDiagonal = LengthSquared(tree.allLightBounds.Diagonal());
    while (!node->isLeaf) {
        int childrenIndices[2] = {nodeIndex + 1, node->childOrLightIndex};
        const LightcutsTreeNode *children[2] = {&tree.nodes[nodeIndex + 1],
                                                &tree.nodes[node->childOrLightIndex]};
        
        Float errBounds[2] = {ComputeErrorBounds(children[0], tree.allLightBounds, bsdf, p),
                              ComputeErrorBounds(children[1], tree.allLightBounds, bsdf, p)};

        if (errBounds[0] == 0 && errBounds[1] == 0) {
            return {};
        }

        // Randomly sample a children node
        Float nodePMF;
        int child = SampleDiscrete(errBounds, u, &nodePMF, &u);
        pmf *= nodePMF;
        nodeIndex = (child == 0) ? (nodeIndex + 1) : node->childOrLightIndex;
        node = &tree.nodes[nodeIndex];

        if (errBounds[child] < m_threshold) {
            int representantLightIndex = tree.nodes[node->representantIdx].childOrLightIndex;
            return SampledLight{tree.lights[representantLightIndex], pmf};
        }
    }

    return SampledLight{tree.lights[node->childOrLightIndex], pmf};
}

pstd::optional<SampledLight> LightcutsLightSampler::SampleInfiniteLight(size_t nLights, Float &pmf, Float &u) const {
    // Compute infinite light sampling probability _pInfinite_
    Float pInfinite = Float(m_infiniteLights.size()) /
                      Float(m_infiniteLights.size() + (nLights == 0 ? 0 : 1));

    if (u < pInfinite) {
        // Sample infinite lights with uniform probability
        u /= pInfinite;
        int index =
            std::min<int>(u * m_infiniteLights.size(), m_infiniteLights.size() - 1);
        Float pmf = pInfinite / m_infiniteLights.size();
        return SampledLight{m_infiniteLights[index], pmf};
    }

    u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);
    pmf = 1 - pInfinite;

    return {};
}

#ifdef PBRT_BUILD_GPU_RENDERER
bool LightcutsLightSampler::buildLightTreeGPU(std::vector<LightBuildContainer> &lights, LightcutsTree& tree, HashMap<Light, LightLocation>& lightToLocation, float& u) {
    if (lights.size() < 100)
        return false;

    LightcutsTreeBuilderGPU builder(tree.allLightBounds, tree.isPoint);
    if (!builder.Build(lights))
        return false;

    builder.FlattenTree(tree, m_lightToLocation, u);
    return true;
}
#endif

std::string LightcutsLightSampler::ToString() const {
    return StringPrintf("[ LightcutsLightSampler point tree nodes: %s spot tree nodes: %s ]", m_pointTree.nodes, m_spotTree.nodes);
}

std::string LightcutsTreeNode::ToString() const {
    return StringPrintf(
        "[ LightcutsLightSampler lightBounds: %s childOrLightIndex: %d isLeaf: %d ]", compactLightBounds, childOrLightIndex, isLeaf);
}

}
