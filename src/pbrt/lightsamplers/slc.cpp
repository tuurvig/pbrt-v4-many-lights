#include "slc.h"

#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>

#include <pbrt/util/hash.h>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/lighttreebuilder.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>

#include <algorithm>
#include <array>

#include <cub/device/device_radix_sort.cuh>
#endif //PBRT_BUILD_GPU_RENDERER

namespace pbrt {

///////////////////////////////////////////////////////////////////////////
// Stochastic Lightcuts LightSampler

STAT_MEMORY_COUNTER("Memory/Stochastic Lightcuts LightTree", SLCLightTreeBytes);

SLCLightSampler::SLCLightSampler(pstd::span<const Light> lights, Allocator alloc, Float threshold) :
    m_tree(alloc), m_infiniteLights(alloc), m_lightToBitTrail(alloc), m_threshold(threshold) {
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

    SLCLightTreeBytes += (m_tree.lights.size() + m_infiniteLights.size()) * sizeof(Light) + 
                         m_tree.nodes.size() * sizeof(LightcutsTreeNode) +
                         m_lightToBitTrail.capacity() * (sizeof(Light) + sizeof(uint32_t));
}

class SLCTreeBuilderGPU final : public LightTreeBuilderGPU<uint64_t, SAOHCostEvaluator> {
  public:
    explicit SLCTreeBuilderGPU(const Bounds3f &bounds) : m_allLightBounds(bounds) {}

    bool Build(std::vector<LightcutsBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);
        MortonCodes() = GetSortedMortonCodes(State(), MortonCodes(), lights);
        BuildNodes(SAOHCostEvaluator());
        return true;
    }

    void FlattenTree(LightcutsTree& tree, std::vector<LightcutsBuildContainer> &lights, HashMap<Light, uint32_t>& bitTrailContainer, Float& u) {
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

        SLCNodeEmitter emitter(tree, bitTrailContainer);
        GPUToLightcutsLeaf adapter(hostNodes, lights);
        
        FlattenLightTree<GPUToLightcutsLeaf, SLCNodeEmitter>(adapter, rootIndex, 0, 0, emitter, u);
    }

    static uint64_t* GetSortedMortonCodes(LightTreeBuildState& buildState, uint64_t* dMortonCodes, const std::vector<LightcutsBuildContainer>& lights) {
        LightTreeBuildState localState = buildState;
        std::array<uint8_t, 3> ax = DetermineAxisOrder(localState.allLightBounds);

        LightcutsBuildContainer* dLightsContainer = GPUAllocAsync<LightcutsBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        GPUParallelFor("Assign Morton Codes", ProfilerKernelGroup::HPLOC, localState.nLights, [=] PBRT_GPU(int idx) {
            LightcutsBuildContainer cont = dLightsContainer[idx];
            LightTreeConstructionNodeGPU leaf(cont.bounds, kInvalidIndex, idx);
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
        
        constexpr uint32_t beginBit = 1;
        constexpr uint32_t endBit = 64;

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

private:
    Bounds3f m_allLightBounds;
};

#ifdef PBRT_BUILD_GPU_RENDERER
bool SLCLightSampler::buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights, Float& u) {
    if (lights.size() < 100 || !Options->useGPU)
        return false;

    SLCTreeBuilderGPU builder(m_tree.allLightBounds);
    if (!builder.Build(lights))
        return false;

    builder.FlattenTree(m_tree, lights, m_lightToBitTrail, u);
    return true;
}
#endif

std::string SLCLightSampler::ToString() const {
    return StringPrintf("[ SLCLightSampler nodes: %s ]", m_tree.nodes);
}


}
