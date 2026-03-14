#include "lightcuts.h"

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
#endif // PBRT_BUILD_GPU_RENDERER

namespace pbrt {
#ifdef PBRT_BUILD_GPU_RENDERER

class LightcutsTreeBuilderGPU final : public LightTreeBuilderGPU<LightBounds, uint32_t, LightcutsCostEvaluator> {
  public:
    explicit LightcutsTreeBuilderGPU(const Bounds3f &bounds, bool isPoint) : m_allLightBounds(bounds), m_isPoint(isPoint) {}

    bool Build(std::vector<LightcutsBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);

        LightTreeBuildState<LightBounds> buildState = State();
        std::array<uint8_t, 3> ax = DetermineAxisOrder(buildState.allLightBounds);

        LightcutsBuildContainer* dLightsContainer = GPUAllocAsync<LightcutsBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        uint32_t* dMortonCodes = MortonCodes();
        MortonCodes() = SortNodesMorton(State(), MortonCodes(), [buildState, dLightsContainer, ax, dMortonCodes] PBRT_GPU(int idx) mutable {
            LightcutsBuildContainer cont = dLightsContainer[idx];
            LightTreeConstructionNodeGPU<LightBounds> leaf(cont.bounds, kInvalidIndex, idx);
            Point3f centroid = cont.bounds.Centroid();
            Vector3f offset = buildState.allLightBounds.Offset(centroid);

            Point3f position = {offset[ax[0]], offset[ax[1]], offset[ax[2]]};

            Float x = QuantizeUnitToBitRange(position.x, 10);
            Float y = QuantizeUnitToBitRange(position.y, 10);
            Float z = QuantizeUnitToBitRange(position.z, 10);

            dMortonCodes[idx] = EncodeMorton3(x, y, z);
            buildState.dClusterIndices[idx] = idx;
            buildState.dNodes[idx] = leaf;
        });

        GPUFreeAsync(dLightsContainer);
        dLightsContainer = nullptr;

        BuildNodes(LightcutsCostEvaluator(m_allLightBounds, m_isPoint));
        return true;
    }

    void FlattenTree(LightcutsTree& tree, std::vector<LightcutsBuildContainer> &lights, HashMap<Light, LightLocation>& bitTrailContainer, Float& u) {
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

        LightcutsNodeEmitter emitter(tree, bitTrailContainer, m_isPoint);
        GPUToLightcutsLeaf adapter(hostNodes, lights);
        
        FlattenLightTree<GPUToLightcutsLeaf, LightcutsNodeEmitter>(adapter, rootIndex, 0, 0, emitter, u);
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

LightcutsLightSampler::LightcutsLightSampler(pstd::span<const Light> lights, Allocator alloc, Float threshold) :
    m_pointTree(alloc), m_spotTree(alloc), m_otherLights(alloc), m_infiniteLights(alloc),
    m_lightToLocation(alloc), m_otherLightIntensities(0), m_threshold(threshold) {
    
    // Initialize infiniteLights array and lightcuts lights
    std::vector<LightcutsBuildContainer> pointLights, spotLights;

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

            } else if (light.Is<SpotLight>() || light.Is<CosineSpotLight>()) {
                spotLights.emplace_back(*lightBounds, light);
                m_spotTree.allLightBounds = Union(m_spotTree.allLightBounds, lightBounds->bounds);

            } else {
                uint32_t index = m_otherLights.size();
                m_otherLights.push_back(light);
                m_lightToLocation.Insert(light, {otherLightsIndex, index});
                m_otherLightIntensities += lightBounds->phi;
            }
        }
    }

    RNG rng;
    Float u = rng.Uniform<Float>();
    if (!pointLights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        bool buildOnGPU = buildLightTreeGPU(pointLights, m_pointTree, true, u);
        if (!buildOnGPU)
#endif
        {
            LightcutsNodeEmitter emitter(m_pointTree, m_lightToLocation, true);
            BuildLightTree<16, LightcutsBuildContainer, LightcutsCostEvaluator, LightcutsNodeEmitter>(pointLights, 0, pointLights.size(), 0, 0, LightcutsCostEvaluator(m_pointTree.allLightBounds, true), emitter, u);
        }
    }

    if (!spotLights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        bool buildOnGPU = buildLightTreeGPU(spotLights, m_spotTree, false, u);
        if (!buildOnGPU)
#endif
        {
            LightcutsNodeEmitter emitter(m_spotTree, m_lightToLocation, false);
            BuildLightTree<16, LightcutsBuildContainer, LightcutsCostEvaluator, LightcutsNodeEmitter>(spotLights, 0, spotLights.size(), 0, 0, LightcutsCostEvaluator(m_spotTree.allLightBounds, false), emitter, u);
        }
    }

    lightCutsLightTreeBytes += (m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size() + m_infiniteLights.size()) * sizeof(Light) + 
                               (m_pointTree.nodes.size() + m_spotTree.nodes.size()) * sizeof(LightcutsTreeNode) +
                               m_lightToLocation.capacity() * (sizeof(Light) + sizeof(LightLocation));
}

PBRT_CPU_GPU
pstd::optional<SampledLight> LightcutsLightSampler::SampleLightTree(const LightSampleContext& ctx, const LightcutsTree& tree, bool isPoint, const BSDF* bsdf, Float pmf, Float u) const {
    const Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));
    Float errBounds[PBRT_LIGHTCUTS_CUT_SIZE] = {0};
    CutData data[PBRT_LIGHTCUTS_CUT_SIZE];
    
    uint32_t bitTrail = 0; // dummy
    int cutSize = ComputeLightcutsTreeCut<PBRT_LIGHTCUTS_CUT_SIZE>(errBounds, data, bitTrail, ctx, tree.nodes, tree.allLightBounds, shadingFrame, bsdf, m_threshold, !isPoint);

    if (cutSize == 0) {
        return {};
    }

    WeightedReservoirSampler<CutData> reservoir(Hash(u));
    for (int i = 0, max = PBRT_LIGHTCUTS_CUT_SIZE; i < max; ++i) {
        Float errBound = errBounds[i];
        if (errBound <= 0) continue;
        
        CutData nodeData = data[i];
        const LightcutsTreeNode* node = &tree.nodes[nodeData.nodeIndex];

        reservoir.Add(data[i], node->compactLightBounds.PhiOrI());
    }

    pmf *= reservoir.SampleProbability();

    constexpr uint32_t indexMask = std::numeric_limits<uint32_t>::max() >> 1;

    const LightcutsTreeNode* node = &tree.nodes[reservoir.GetSample().nodeIndex & indexMask];
    const LightcutsTreeNode* representant = &tree.nodes[node->representantIdx];
    
    const Float nodeIntensity = node->compactLightBounds.PhiOrI();
    const Float repIntensity = representant->compactLightBounds.PhiOrI();

    int representantLightIndex = representant->childOrLightIndex;
    return SampledLight(tree.lights[representantLightIndex], pmf, FloatToBits(nodeIntensity / repIntensity));
}

#ifdef PBRT_BUILD_GPU_RENDERER
bool LightcutsLightSampler::buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights, LightcutsTree& tree, bool isPoint, Float& u) {
    if (lights.size() < 100 || !Options->useGPU)
        return false;

    LightcutsTreeBuilderGPU builder(tree.allLightBounds, isPoint);
    if (!builder.Build(lights))
        return false;

    builder.FlattenTree(tree, lights, m_lightToLocation, u);
    return true;
}
#endif

std::string LightcutsLightSampler::ToString() const {
    return StringPrintf("[ LightcutsLightSampler point tree nodes: %s spot tree nodes: %s ]", m_pointTree.nodes, m_spotTree.nodes);
}

}
