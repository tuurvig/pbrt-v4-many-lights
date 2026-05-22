// Copyright(c) 2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include "hslc.h"

#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/timer.h>

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
// Hierarchic Stochastic Lightcuts LightSampler

STAT_MEMORY_COUNTER("Memory/Hierarchic Stochastic Lightcuts LightTree", HSLCLightTreeBytes);
STAT_COUNTER("Time/CPU Construction", constructionMicroseconds);

HSLCLightSampler::HSLCLightSampler(pstd::span<const Light> lights, Allocator alloc) :
    m_tree(alloc), m_infiniteLights(alloc), m_lightToBitTrail(alloc) {
    std::vector<LightcutsBuildContainer> treeLights;
    for (size_t i = 0; i < lights.size(); ++i) {
        // Store $i$th light in either _infiniteLights_ or _treeLights_
        Light light = lights[i];
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        if (!lightBounds) {
            m_infiniteLights.push_back(light);
        }
        else if (lightBounds->phi > 0) {
            treeLights.emplace_back(*lightBounds, light);
            m_tree.allLightBounds = Union(m_tree.allLightBounds, lightBounds->bounds);
        }
    }

    //RNG rng;
    //Float u = rng.Uniform<Float>();
    if (!treeLights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        // Prefer GPU build for larger trees when GPU rendering is enabled.
        bool buildOnGPU = buildLightTreeGPU(treeLights);
        if (!buildOnGPU)
#endif
        {
            Timer constructionTimer;
            SLCNodeEmitter emitter(m_tree, m_lightToBitTrail);
            BuildLightTree<16, LightcutsBuildContainer, SAOHCostEvaluator, SLCNodeEmitter>(treeLights, 0, treeLights.size(), 0, 0, SAOHCostEvaluator(), emitter);
            constructionMicroseconds += constructionTimer.ElapsedMicroseconds();
        }
    }

    HSLCLightTreeBytes += (m_tree.lights.size() + m_infiniteLights.size()) * sizeof(Light) + 
                          m_tree.nodes.size() * sizeof(LightcutsTreeNode) +
                          m_lightToBitTrail.capacity() * (sizeof(Light) + sizeof(uint32_t));
}

#ifdef PBRT_BUILD_GPU_RENDERER
/// @brief GPU builder for HSLC trees based on SAOH ordering and Morton sorting.
class HSLCTreeBuilderGPU final : public LightTreeBuilderGPU<LightBounds, uint64_t, SAOHCostEvaluator> {
  public:
    /// @brief Creates a builder for the provided scene light bounds.
    explicit HSLCTreeBuilderGPU(const Bounds3f &bounds) : m_allLightBounds(bounds) {}

    /// @brief Builds temporary GPU hierarchy and merges nodes with SAOH.
    bool Build(std::vector<LightcutsBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);

        LightTreeBuildState<LightBounds> buildState(State());
        std::array<uint8_t, 3> ax = DetermineAxisOrder(buildState.allLightBounds);

        LightcutsBuildContainer* dLightsContainer = GPUAllocAsync<LightcutsBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        uint64_t* dMortonCodes = MortonCodes();
        MortonCodes() = SortNodesMorton(State(), MortonCodes(), [buildState, ax, dLightsContainer, dMortonCodes] PBRT_GPU(int idx) {
            LightcutsBuildContainer cont = dLightsContainer[idx];
            LightTreeConstructionNodeGPU<LightBounds> leaf(cont.bounds, kInvalidIndex, idx);
            Point3f centroid = cont.bounds.Centroid();
            Vector3f offset = buildState.allLightBounds.Offset(centroid);

            Point3f position = {offset[ax[0]], offset[ax[1]], offset[ax[2]]};
            Vector3f direction = Normalize(cont.bounds.w);

            dMortonCodes[idx] = EncodeExtendedMorton5(position, direction);
            buildState.dClusterIndices[idx] = idx;
            buildState.dNodes[idx] = leaf;
        });
        
        GPUFreeAsync(dLightsContainer);
        dLightsContainer = nullptr;

        BuildNodes(SAOHCostEvaluator());
        
        return true;
    }

    /// @brief Copies merged GPU nodes and emits a flattened `LightcutsTree`.
    void FlattenTree(LightcutsTree& tree, std::vector<LightcutsBuildContainer> &lights, HashMap<Light, uint32_t>& bitTrailContainer) {
        const LightTreeBuildState<LightBounds> &state(State());
        if (state.nLights == 0)
            return;

        uint32_t nNodes = 0;
        uint32_t rootIndex = 0;
        GPUCopyToHost(&nNodes, state.nMergedClusters, 1);
        GPUCopyToHost(&rootIndex, state.dClusterIndices, 1);
        std::vector<LightTreeConstructionNodeGPU<LightBounds>> hostNodes(nNodes);
        GPUCopyToHost(hostNodes.data(), state.dNodes, nNodes);

        tree.nodes.reserve(nNodes);

        SLCNodeEmitter emitter(tree, bitTrailContainer);
        GPUToLightcutsLeaf adapter(hostNodes, lights);

        Timer flattenTimer;
        FlattenLightTree<GPUToLightcutsLeaf, SLCNodeEmitter>(adapter, rootIndex, 0, 0, emitter);
        constructionMicroseconds += flattenTimer.ElapsedMicroseconds();
    }

private:
    Bounds3f m_allLightBounds;
};

/// @brief Builds the HSLC tree on GPU and flattens it to host memory.
bool HSLCLightSampler::buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights) {
    if (lights.size() < 100 || !Options->useGPU)
        return false;

    HSLCTreeBuilderGPU builder(m_tree.allLightBounds);
    if (!builder.Build(lights))
        return false;

    builder.FlattenTree(m_tree, lights, m_lightToBitTrail);
    return true;
}
#endif

std::string HSLCLightSampler::ToString() const {
    return StringPrintf("[ HSLCLightSampler nodes: %s ]", m_tree.nodes);
}


}
