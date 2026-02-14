#include "hslc.h"

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
// Hierarchic Stochastic Lightcuts LightSampler

STAT_MEMORY_COUNTER("Memory/Hierarchic Stochastic Lightcuts LightTree", HSLCLightTreeBytes);

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

    RNG rng;
    Float u = rng.Uniform<Float>();
    if (!treeLights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        bool buildOnGPU = buildLightTreeGPU(treeLights, u);
        if (!buildOnGPU)
#endif
        {
            SLCNodeEmitter emitter(m_tree, m_lightToBitTrail);
            BuildLightTree<16, LightcutsBuildContainer, SAOHCostEvaluator, SLCNodeEmitter>(treeLights, 0, treeLights.size(), 0, 0, SAOHCostEvaluator(), emitter, u);
        }
    }

    HSLCLightTreeBytes += (m_tree.lights.size() + m_infiniteLights.size()) * sizeof(Light) + 
                          m_tree.nodes.size() * sizeof(LightcutsTreeNode) +
                          m_lightToBitTrail.capacity() * (sizeof(Light) + sizeof(uint32_t));
}

#ifdef PBRT_BUILD_GPU_RENDERER
class HSLCTreeBuilderGPU final : public LightTreeBuilderGPU<LightBounds, uint64_t, SAOHCostEvaluator> {
  public:
    explicit HSLCTreeBuilderGPU(const Bounds3f &bounds) : m_allLightBounds(bounds) {}

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

    void FlattenTree(LightcutsTree& tree, std::vector<LightcutsBuildContainer> &lights, HashMap<Light, uint32_t>& bitTrailContainer, Float& u) {
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
        
        FlattenLightTree<GPUToLightcutsLeaf, SLCNodeEmitter>(adapter, rootIndex, 0, 0, emitter, u);
    }

private:
    Bounds3f m_allLightBounds;
};

bool HSLCLightSampler::buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights, Float& u) {
    if (lights.size() < 100 || !Options->useGPU)
        return false;

    HSLCTreeBuilderGPU builder(m_tree.allLightBounds);
    if (!builder.Build(lights))
        return false;

    builder.FlattenTree(m_tree, lights, m_lightToBitTrail, u);
    return true;
}
#endif

std::string HSLCLightSampler::ToString() const {
    return StringPrintf("[ HSLCLightSampler nodes: %s ]", m_tree.nodes);
}


}
