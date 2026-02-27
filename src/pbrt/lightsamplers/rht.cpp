// rht.cpp - RHTLightSampler class is Copyright(c) 2025-2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include "rht.h"
#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/hash.h>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/lighttreebuilder.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>

#include <algorithm>
#include <array>

#endif //PBRT_BUILD_GPU_RENDERER

namespace pbrt {

///////////////////////////////////////////////////////////////////////////
// Resampled Hierarchic Tree Light LightSampler

STAT_MEMORY_COUNTER("Memory/Resampled Hierarchic Tree", RHTLightTreeBytes);

RHTLightSampler::RHTLightSampler(pstd::span<const Light> lights, Allocator alloc, Float gamma) :
    m_tree(alloc), m_infiniteLights(alloc), m_lightToBitTrail(alloc), gamma(gamma) {
    std::vector<RHTBuildContainer> treeLights;
    {
        std::vector<LightBVHBuildContainer> lightsForLeaves;
        lightsForLeaves.reserve(lights.size());
        for (size_t i = 0; i < lights.size(); ++i) {
            // Store $i$th light in either _infiniteLights_ or _treeLights_
            Light light = lights[i];
            pstd::optional<LightBounds> lightBounds = light.Bounds();
            if (!lightBounds) {
                m_infiniteLights.push_back(light);
            }
            else if (lightBounds->phi > 0) {
                lightsForLeaves.emplace_back(*lightBounds, i);
                m_tree.allLightBounds = Union(m_tree.allLightBounds, lightBounds->bounds);
            }
        }

        m_tree.leaves.reserve(lightsForLeaves.size());
        treeLights.reserve(lightsForLeaves.size());
        for (size_t i = 0; i < lightsForLeaves.size(); ++i) {
            const LightBVHBuildContainer& container(lightsForLeaves[i]);
            // For build, use SphericalLightBounds derived from LightBounds
            treeLights.emplace_back(SphericalLightBounds(container.bounds.bounds, container.bounds.phi), i);
            Light light = lights[container.index];
            // Store detailed CompactLightBounds for leaves
            m_tree.leaves.emplace_back(container.bounds, container.bounds.phi, m_tree.allLightBounds, light);
        }
    }

    if (!treeLights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        bool buildOnGPU = buildLightTreeGPU(treeLights);
        if (!buildOnGPU)
#endif
        {
            Float u = 0; // dummy
            RHTNodeEmitter emitter(m_tree, m_lightToBitTrail);
            BuildLightTree<16, RHTBuildContainer, SphericalBoundsCostEvaluator, RHTNodeEmitter>(treeLights, 0, treeLights.size(), 0, 0, SphericalBoundsCostEvaluator(), emitter, u);
        }
    }

    RHTLightTreeBytes += (m_infiniteLights.size()) * sizeof(Light) + 
                          m_tree.leaves.size() * sizeof(CompactLight) +
                          m_tree.innerNodes.size() * sizeof(ResampledTreeNode) +
                          m_lightToBitTrail.capacity() * (sizeof(Light) + sizeof(uint32_t));
}

#ifdef PBRT_BUILD_GPU_RENDERER
class RHTreeBuilderGPU final : public LightTreeBuilderGPU<SphericalLightBounds, uint32_t, SphericalBoundsCostEvaluator> {
  public:
    explicit RHTreeBuilderGPU(const Bounds3f &bounds) : m_allLightBounds(bounds) {}

    bool Build(std::vector<RHTBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);

        LightTreeBuildState<SphericalLightBounds> buildState(State());

        RHTBuildContainer* dLightsContainer = GPUAllocAsync<RHTBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        const Float largestRadius = Length(buildState.allLightBounds.Diagonal()) * Float(0.5);
        const Float sqrtLargestRadius = std::sqrt(largestRadius);

        uint32_t* dMortonCodes = MortonCodes();
        MortonCodes() = SortNodesMorton(State(), MortonCodes(), [buildState, sqrtLargestRadius, dLightsContainer, dMortonCodes] PBRT_GPU(int idx) {
            const RHTBuildContainer cont = dLightsContainer[idx];
            const LightTreeConstructionNodeGPU<SphericalLightBounds> leaf(cont.bounds, kInvalidIndex, idx);
            const Point3f centroid = cont.bounds.Centroid();
            const Vector3f offset = buildState.allLightBounds.Offset(centroid);

            const Float x = QuantizeUnitToBitRange(offset.x, 10);
            const Float y = QuantizeUnitToBitRange(offset.y, 10);
            const Float z = QuantizeUnitToBitRange(offset.z, 10);

            dMortonCodes[idx] = EncodeMorton3(x, y, z);
            buildState.dClusterIndices[idx] = idx;
            buildState.dNodes[idx] = leaf;
        });
        
        GPUFreeAsync(dLightsContainer);
        dLightsContainer = nullptr;

        BuildNodes(SphericalBoundsCostEvaluator());
        
        return true;
    }

    void FlattenTree(ResampledTree& tree, std::vector<RHTBuildContainer> &lights, HashMap<Light, uint32_t>& bitTrailContainer, Float& u) {
        const LightTreeBuildState<SphericalLightBounds> &state(State());
        if (state.nLights == 0)
            return;

        uint32_t nNodes = 0;
        uint32_t rootIndex = 0;
        GPUCopyToHost(&nNodes, state.nMergedClusters, 1);
        GPUCopyToHost(&rootIndex, state.dClusterIndices, 1);
        std::vector<LightTreeConstructionNodeGPU<SphericalLightBounds>> hostNodes(nNodes);
        GPUCopyToHost(hostNodes.data(), state.dNodes, nNodes);

        tree.innerNodes.reserve(nNodes);

        RHTNodeEmitter emitter(tree, bitTrailContainer);
        GPUToRHTLeaf adapter(hostNodes, lights);
        
        FlattenLightTree<GPUToRHTLeaf, RHTNodeEmitter>(adapter, rootIndex, 0, 0, emitter, u);
    }

private:
    Bounds3f m_allLightBounds;
};

bool RHTLightSampler::buildLightTreeGPU(std::vector<RHTBuildContainer> &lights) {
    if (lights.size() < 100 || !Options->useGPU)
        return false;

    RHTreeBuilderGPU builder(m_tree.allLightBounds);
    if (!builder.Build(lights))
        return false;

    Float u = 0;
    builder.FlattenTree(m_tree, lights, m_lightToBitTrail, u);
    return true;
}
#endif

#define PBRT_RHT_MAX_STACK 64
struct alignas(16) TraversalState {
    uint32_t nodeIndex;
    Float PsParent; // probability of splitting C_parent
    Float T; // accumulated traversal state T(C)
    Float uNode; // random variable for traversal
};

PBRT_CPU_GPU
void RHTLightSampler::CollectLightCandidates(HeuristicHReservoirSet& reservoirSet, const LightSampleContext& ctx, uint32_t seed, const Float u, const Float uSplit, const Float pmf) const {
    TraversalState stack[PBRT_RHT_MAX_STACK];
    int stackHead = 0;

    const Float startingSplitProbability = std::max(u, Float(1) - MathEpsilon);
    stack[stackHead] = {0, startingSplitProbability, Float(1), u};

    Point3f p = ctx.p();
    Normal3f n = ctx.ns;

    while (stackHead >= 0) {
        const TraversalState state = stack[stackHead];
        --stackHead;

        const ResampledTreeNode* node = &m_tree.innerNodes[state.nodeIndex];
        if (node->isLeaf) {
            const Float pdf = state.PsParent + (1 - state.PsParent) * state.T;
            const LightCandidate candidate{node->childOrLightIndex, pmf * pdf};

            const CompactLight& cl(m_tree.leaves[candidate.lightIdx]);
            const Float hWeight = cl.bounds.Importance(p, n, m_tree.allLightBounds);

            reservoirSet.Add(candidate, hWeight);
            continue;
        }

        const uint32_t childrenIndices[2] = {static_cast<uint32_t>(state.nodeIndex + 1), node->childOrLightIndex};

        const Float PsNode = std::min(node->bounds.SplitProbability(p, gamma), state.PsParent); // Ps(C)
        const Float PsHatNode = 1 - PsNode; // Ps_hat(C)

        // propability of splitting C_parent given that C has not been split
        const Float Pns = (state.PsParent - PsNode) / PsHatNode; // Pns(C)

        const Float T_node = Pns + (1 - Pns) * state.T;
        
        if (uSplit <= PsNode) {
            stackHead += 2;
            DCHECK_LT(stackHead, PBRT_RHT_MAX_STACK);
            Float uLeft  = state.uNode + HashFloat(seed, childrenIndices[0]);
            Float uRight = state.uNode + HashFloat(seed, childrenIndices[1]);

            if (uLeft > 1) uLeft -= 1;
            if (uRight > 1) uRight -= 1;

            stack[stackHead]     = {childrenIndices[0], PsNode, T_node, uLeft};
            stack[stackHead - 1] = {childrenIndices[1], PsNode, T_node, uRight};
            continue;
        }
        
        const ResampledTreeNode *children[2] = {&m_tree.innerNodes[childrenIndices[0]],
                                                &m_tree.innerNodes[childrenIndices[1]]};

        const Float ci[2] = {children[0]->bounds.Importance(p, n),
                             children[1]->bounds.Importance(p, n)};

        if (ci[0] == 0 && ci[1] == 0) {
            continue;
        }

        // Pick child to traverse
        Float nodePMF = 0;
        Float uWarped = 0;
        int child = SampleDiscrete(ci, state.uNode, &nodePMF, &uWarped);
        uWarped += HashFloat(seed, childrenIndices[child]);
        if (uWarped > 1) uWarped -= 1;

        ++stackHead;
        DCHECK_LT(stackHead, PBRT_RHT_MAX_STACK);
        stack[stackHead] = {childrenIndices[child], PsNode, T_node * nodePMF, uWarped};
    }
}


std::string RHTLightSampler::ToString() const {
    return StringPrintf("[ RHTLightSampler innerNodes: %s leaves: %s ]", m_tree.innerNodes, m_tree.leaves);
}

} // namespace pbrt
