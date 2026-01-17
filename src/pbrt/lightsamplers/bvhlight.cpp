#include "bvhlight.h"

#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>
#include <vector>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/lighttreebuilder.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/options.h>

#include <algorithm>
#include <array>

#include <cub/device/device_radix_sort.cuh>
#endif //PBRT_BUILD_GPU_RENDERER

namespace pbrt{

#ifdef PBRT_BUILD_GPU_RENDERER

class BVHLightTreeBuilder final : public LightTreeBuilderGPU<uint64_t, SAOHCostEvaluator> {
  public:
    explicit BVHLightTreeBuilder(const Bounds3f &bounds) : m_allLightBounds(bounds) {}

    bool Build(std::vector<LightBVHBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);

        LightTreeBuildState buildState(State());
        std::array<uint8_t, 3> ax = DetermineAxisOrder(buildState.allLightBounds);

        LightBVHBuildContainer* dLightsContainer = GPUAllocAsync<LightBVHBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        uint64_t* dMortonCodes = MortonCodes();
        MortonCodes() = SortNodesMorton(State(), MortonCodes(), [buildState, ax, dLightsContainer, dMortonCodes] PBRT_GPU(int idx) {
            LightBVHBuildContainer cont = dLightsContainer[idx];
            LightTreeConstructionNodeGPU leaf{cont.bounds, kInvalidIndex, static_cast<uint32_t>(cont.index)};
            Point3f centroid = cont.bounds.Centroid();
            Vector3f offset = buildState.allLightBounds.Offset(centroid);

            Point3f position = {offset[ax[0]], offset[ax[1]], offset[ax[2]]};
            Vector3f direction = Normalize(cont.bounds.w);

            dMortonCodes[idx] = EncodeExtendedMorton5(position, direction);

            buildState.dClusterIndices[idx] = idx;
            buildState.dNodes[idx] = leaf;
        });

        BuildNodes(SAOHCostEvaluator());
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

        LightHierarchyNodeEmitter emitter(nodes, bitTrailContainer, lights, m_allLightBounds);
        GPUToLightBVHLeaf adapter(hostNodes);
        Float u = 0;

        FlattenLightTree<GPUToLightBVHLeaf, LightHierarchyNodeEmitter>(adapter, rootIndex, 0, 0, emitter, u);
    }

    static uint64_t* UploadSortedLeaves(LightTreeBuildState& buildState, uint64_t* dMortonCodes, const std::vector<LightBVHBuildContainer> &lights) {
        LightTreeBuildState localState = buildState;
        std::array<uint8_t, 3> ax = DetermineAxisOrder(localState.allLightBounds);

        LightBVHBuildContainer* dLightsContainer = GPUAllocAsync<LightBVHBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        GPUParallelFor("Assign Morton Codes", ProfilerKernelGroup::HPLOC, localState.nLights, [=] PBRT_GPU(int idx) {
            LightBVHBuildContainer cont = dLightsContainer[idx];
            LightTreeConstructionNodeGPU leaf{cont.bounds, kInvalidIndex, static_cast<uint32_t>(cont.index)};
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
        {
            LightHierarchyNodeEmitter nodeEmitter(m_nodes, m_lightToBitTrail, m_lights, m_allLightBounds);
            Float u = 0;
            BuildLightTree<16, LightBVHBuildContainer, SAOHCostEvaluator, LightHierarchyNodeEmitter>(bvhLights, 0, bvhLights.size(), 0, 0, SAOHCostEvaluator(), nodeEmitter, u);
        }
    }
    
    lightBVHBytes += m_nodes.size() * sizeof(LightBVHNode) +
                     m_lightToBitTrail.capacity() * (sizeof(Light) + sizeof(uint32_t)) +
                     lights.size() * sizeof(Light) +
                     m_infiniteLights.size() * sizeof(Light);
}

#ifdef PBRT_BUILD_GPU_RENDERER
bool BVHLightSampler::buildBVHGPU(
    std::vector<LightBVHBuildContainer> &bvhLights) {
    if (bvhLights.size() < 100 || !Options->useGPU)
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

}
