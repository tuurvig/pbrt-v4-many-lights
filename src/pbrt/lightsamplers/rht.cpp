// Copyright(c) 2026 Richard Kvasnica.
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

/// @brief Builds the RHT hierarchy from lights provided by integrator.
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
            // Build phase uses spherical bounds (cheap split/importance evaluation).
            treeLights.emplace_back(SphericalLightBounds(container.bounds.bounds, container.bounds.phi), i);
            Light light = lights[container.index];
            // Leaf payload keeps tighter compact bounds + original Light handle.
            m_tree.leaves.emplace_back(container.bounds, container.bounds.phi, m_tree.allLightBounds, light);
        }
    }

    if (!treeLights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        // Prefer GPU build for larger trees when GPU rendering is enabled.
        bool buildOnGPU = buildLightTreeGPU(treeLights);
        if (!buildOnGPU)
#endif
        {
            RHTNodeEmitter emitter(m_tree, m_lightToBitTrail);
            BuildLightTree<16, RHTBuildContainer, SphericalBoundsCostEvaluator, RHTNodeEmitter>(treeLights, 0, treeLights.size(), 0, 0, SphericalBoundsCostEvaluator(), emitter);
        }
    }

    RHTLightTreeBytes += (m_infiniteLights.size()) * sizeof(Light) + 
                          m_tree.leaves.size() * sizeof(CompactLight) +
                          m_tree.innerNodes.size() * sizeof(ResampledTreeNode) +
                          m_lightToBitTrail.capacity() * (sizeof(Light) + sizeof(uint32_t));
}

#ifdef PBRT_BUILD_GPU_RENDERER
/// @brief GPU builder for RHT using spherical bounds and energy-weighted SAH.
class RHTreeBuilderGPU final : public LightTreeBuilderGPU<SphericalLightBounds, uint32_t, SphericalBoundsCostEvaluator> {
  public:
    /// @brief Creates a builder for the provided scene bounds.
    explicit RHTreeBuilderGPU(const Bounds3f &bounds) : m_allLightBounds(bounds) {}

    /// @brief Builds temporary GPU hierarchy and merges nodes via spherical cost metric.
    bool Build(std::vector<RHTBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);

        LightTreeBuildState<SphericalLightBounds> buildState(State());

        RHTBuildContainer* dLightsContainer = GPUAllocAsync<RHTBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        // Kept for parity with related builders; not used directly in the current Morton key.
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

    /// @brief Copies merged GPU nodes and emits a flattened `ResampledTree`.
    void FlattenTree(ResampledTree& tree, std::vector<RHTBuildContainer> &lights, HashMap<Light, uint32_t>& bitTrailContainer) {
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
        GPUToRHTLeaf adapter(hostNodes);
        
        FlattenLightTree<GPUToRHTLeaf, RHTNodeEmitter>(adapter, rootIndex, 0, 0, emitter);
    }

private:
    Bounds3f m_allLightBounds;
};

/// @brief Builds the RHT hierarchy on GPU and flattens it to host storage.
bool RHTLightSampler::buildLightTreeGPU(std::vector<RHTBuildContainer> &lights) {
    if (lights.size() < 100 || !Options->useGPU)
        return false;

    RHTreeBuilderGPU builder(m_tree.allLightBounds);
    if (!builder.Build(lights))
        return false;

    builder.FlattenTree(m_tree, lights, m_lightToBitTrail);
    return true;
}
#endif

#define PBRT_RHT_MAX_STACK 32

/// @brief Compact stack element used during candidate collection traversal.
/// Stores the node index together with normalized traversal scalars in packed form.
struct alignas(8) PackedTraversalState {
    PackedTraversalState() = default;

    PBRT_CPU_GPU
    PackedTraversalState(uint32_t nodeIndex, Float T, Float PsParent) :
        nodeIndex(nodeIndex), T(PackNormalizedFloat(T)), PsParent(PackNormalizedFloat(PsParent)) {}

    uint32_t nodeIndex;
    uint16_t T;
    uint16_t PsParent;
};

/// @brief Expanded traversal state used while processing one active branch.
struct alignas(16) TraversalState {
    PBRT_CPU_GPU
    TraversalState(uint32_t nodeIndex, Float T, Float PsParent) :
        nodeIndex(nodeIndex), T(T), PsParent(PsParent) {}

    PBRT_CPU_GPU
    void operator=(PackedTraversalState state) {
        nodeIndex = state.nodeIndex;
        T = UnpackToFloat(state.T);
        PsParent = UnpackToFloat(state.PsParent);
    }

    uint32_t nodeIndex;
    Float T; // accumulated traversal state T(C)
    Float PsParent; // probability of splitting C_parent
};

PBRT_CPU_GPU 
/// @brief Traverses the RHT and populates heuristic-H reservoirs with light candidates.
/// The traversal follows the split/no-split model from the paper and evaluates candidate weights.
void RHTLightSampler::CollectLightCandidates(HeuristicHReservoirSet& reservoirSet, const LightSampleContext& ctx, uint32_t seed, Float u, const Float uSplit, const Float pmf) const {
    PackedTraversalState stack[PBRT_RHT_MAX_STACK];
    int stackHead = -1;

    // Start from the root with full traversal weight.
    TraversalState state(0, Float(1), Float(1));

    Point3f p = ctx.p();
    Normal3f n = ctx.ns;

    while (true) {
        const ResampledTreeNode* node = &m_tree.innerNodes[state.nodeIndex];

        if (node->isLeaf) {
            // Final proposal density for this leaf under split/no-split process.
            const Float pdf = (state.PsParent + (1 - state.PsParent) * state.T) * pmf;

            const uint32_t lightIdx = node->childOrLightIndex;
            const CompactLight &cl(m_tree.leaves[lightIdx]);
            const Float hImportance = cl.bounds.Importance(p, n, m_tree.allLightBounds) / pdf;

            if (pdf > 0 && hImportance > 0) {
                const LightCandidate candidate{lightIdx, pdf};
                reservoirSet.Add(candidate, hImportance);
            }

            if (stackHead < 0) break;

            state = stack[stackHead];
            --stackHead;
            continue;
        }

        const uint32_t childIdxLeft = static_cast<uint32_t>(state.nodeIndex + 1);
        const uint32_t childIdxRight = node->childOrLightIndex;

        const Float splitProb = node->bounds.SplitProbability(p, gamma);
        Float PsNode = state.PsParent; // Ps(C)
        Float T_node = state.T;
        if (state.PsParent > splitProb) {
            PsNode = splitProb;
            const Float PsHatNode = 1 - PsNode; // Ps_hat(C)

            // Probability of splitting parent given that current node has not split.
            const Float Pns = std::min((state.PsParent - PsNode) / PsHatNode, OneMinusEpsilon); // Pns(C)
            T_node = Pns + (1 - Pns) * state.T;
        }

        const ResampledTreeNode *childLeft = &m_tree.innerNodes[childIdxLeft];
        const ResampledTreeNode *childRight = &m_tree.innerNodes[childIdxRight];

        const Float importanceLeft = childLeft->bounds.Importance(p, n);
        const Float importanceRight = childRight->bounds.Importance(p, n);

        const Float wSum = importanceLeft + importanceRight;

        if (wSum == 0) {
            if (stackHead < 0) break;
            
            state = stack[stackHead];
            --stackHead;
            continue;
        }

        const Float pLeft = std::min(importanceLeft / wSum, OneMinusEpsilon);
        const Float pRight = 1 - pLeft;

        if (uSplit <= PsNode) {
            DCHECK_LT(stackHead, PBRT_RHT_MAX_STACK - 1);

            // Branch split: process left now, defer right on the explicit stack.
            state = TraversalState(childIdxLeft, T_node * pLeft, PsNode);
            stackHead++;
            stack[stackHead] = PackedTraversalState(childIdxRight, T_node * pRight, PsNode);
            continue;
        }

        // No split: stochastically select exactly one child and continue.
        if (u <= pLeft) {
            u /= pLeft;
            state = TraversalState(childIdxLeft, T_node * pLeft, PsNode);
        } else {
            u = (u - pLeft) / pRight;
            state = TraversalState(childIdxRight, T_node * pRight, PsNode);
        }

        // Scramble traversal sample after each decision to reduce structured correlation.
        u += HashFloat(seed, state.nodeIndex);
        if (u >= 1) u -= 1;
    }
}


std::string RHTLightSampler::ToString() const {
    return StringPrintf("[ RHTLightSampler innerNodes: %s leaves: %s ]", m_tree.innerNodes, m_tree.leaves);
}

} // namespace pbrt
