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
#endif //PBRT_BUILD_GPU_RENDERER

namespace pbrt{
#ifdef PBRT_BUILD_GPU_RENDERER

class LightcutsTreeBuilderGPU final : public LightTreeBuilderGPU<uint32_t, LightcutsCostEvaluator> {
  public:
    explicit LightcutsTreeBuilderGPU(const Bounds3f &bounds, bool isPoint) : m_allLightBounds(bounds), m_isPoint(isPoint) {}

    bool Build(std::vector<LightcutsBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);

        LightTreeBuildState buildState = State();
        std::array<uint8_t, 3> ax = DetermineAxisOrder(buildState.allLightBounds);

        LightcutsBuildContainer* dLightsContainer = GPUAllocAsync<LightcutsBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        uint32_t* dMortonCodes = MortonCodes();
        MortonCodes() = SortNodesMorton(State(), MortonCodes(), [buildState, dLightsContainer, ax, dMortonCodes] PBRT_GPU(int idx) mutable {
            LightcutsBuildContainer cont = dLightsContainer[idx];
            LightTreeConstructionNodeGPU leaf(cont.bounds, kInvalidIndex, idx);
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

LightcutsTree::LightcutsTree(Allocator alloc) 
    : lights(alloc), nodes(alloc) {}

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

            } else if (light.Is<SpotLight>()) {
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
    Point3f p = ctx.p();
    Vector3f wo = ctx.wo;
    Normal3f n = ctx.n;

    BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
    Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.n));

    Float estL = 0;
    Float estParent = 0;
    Float clusterEst[2] = {};

    constexpr Float floatUintMax = 0x1p32f;
    uint32_t currentU = static_cast<uint32_t>(u * floatUintMax);

    int nodeIndex = 0;
    const LightcutsTreeNode* node = &tree.nodes[nodeIndex];

    while (!node->isLeaf) {
        uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

        const LightcutsTreeNode *children[2] = {&tree.nodes[childrenIndices[0]],
                                                &tree.nodes[childrenIndices[1]]};
        
        const Float nodeIntensities[2] = {children[0]->compactLightBounds.PhiOrI(),
                                          children[1]->compactLightBounds.PhiOrI()};

        const LightcutsTreeNode *representants[2] = {&tree.nodes[children[0]->representantIdx],
                                                     &tree.nodes[children[1]->representantIdx]};

        const Float clusterEst[2] = {
            ComputeClusterEstimate(bsdf, bsdfFlags, representants[0]->compactLightBounds.Bound(tree.allLightBounds, false), p, n, wo, nodeIntensities[0]),
            ComputeClusterEstimate(bsdf, bsdfFlags, representants[1]->compactLightBounds.Bound(tree.allLightBounds, false), p, n, wo, nodeIntensities[1])
        };

        Float errBounds[2] = {1, 1};
        
        constexpr Float minLengthSqr = 1e-6f;

        if (nodeIntensities[0] != 0 && nodeIntensities[1] != 0) {
            const Bounds3f nodeBound0 = children[0]->compactLightBounds.Bounds(tree.allLightBounds);
            const Bounds3f nodeBound1 = children[1]->compactLightBounds.Bounds(tree.allLightBounds);

            Float geomBound0 = ComputeGeometricBound(children[0], nodeBound0, shadingFrame, !isPoint, p, wo, bsdf && IsTransmissive(bsdfFlags));
            Float geomBound1 = ComputeGeometricBound(children[1], nodeBound1, shadingFrame, !isPoint, p, wo, bsdf && IsTransmissive(bsdfFlags));

            if (geomBound0 > MachineEpsilon && geomBound1 > MachineEpsilon) {
                Float ub0 = geomBound0 * nodeIntensities[0];
                Float ub1 = geomBound1 * nodeIntensities[1];

                Float matBound0 = 1;
                Float matBound1 = 1;

                if (bsdf) {
                    matBound0 = bsdf->Max_f(wo, nodeBound0, p);
                    matBound1 = bsdf->Max_f(wo, nodeBound1, p);
                }

                if ((matBound0 > MachineEpsilon && matBound1 > MachineEpsilon)) {
                    ub0 *= matBound0;
                    ub1 *= matBound1;

                    const Float diagonalLengthSqr0 = std::max(LengthSquared(nodeBound0.Diagonal()), minLengthSqr);
                    const Float diagonalLengthSqr1 = std::max(LengthSquared(nodeBound1.Diagonal()), minLengthSqr);

                    Float dist2Min0 = DistanceSquared(p, ClosestPoint(p, nodeBound0));
                    Float dist2Min1 = DistanceSquared(p, ClosestPoint(p, nodeBound1));

                    if (dist2Min0 > diagonalLengthSqr0 && dist2Min1 > diagonalLengthSqr1) {
                        Float dBoundMin0 = 1 / dist2Min0;
                        Float dBoundMin1 = 1 / dist2Min1;
                    
                        errBounds[0] = dBoundMin0 * ub0;
                        errBounds[1] = dBoundMin1 * ub1;
                    }
                    else {
                        errBounds[0] = ub0;
                        errBounds[1] = ub1;
                    }
                } else {
                    if (matBound0 < MachineEpsilon && matBound1 < MachineEpsilon) {
                        return {};
                    }

                    // weight of the first child will be 1 or 0 based on whether the other child is 0.
                    errBounds[0] = static_cast<Float>(matBound1 < MachineEpsilon);
                    errBounds[1] = 1 - errBounds[0];
                }
            } else {
                if (geomBound0 < MachineEpsilon && geomBound1 < MachineEpsilon) {
                    return {};
                }
                // weight of the first child will be 1 or 0 based on whether the other child is 0.
                errBounds[0] = static_cast<Float>(geomBound1 < MachineEpsilon);
                errBounds[1] = 1 - errBounds[0];
            }
        } else {
            if (nodeIntensities[0] == 0 && nodeIntensities[1] == 0) {
                return {};
            }
            // weight of the first child will be 1 or 0 based on whether the other child is 0.
            errBounds[0] = static_cast<Float>(nodeIntensities[1] == 0);
            errBounds[1] = 1 - errBounds[0];
        }

        if (errBounds[0] < MachineEpsilon) {
            
            if (errBounds[1] < MachineEpsilon) {
                return {};
            }
            errBounds[0] = MachineEpsilon;
        } else if (errBounds[1] < MachineEpsilon){
            errBounds[1] = MachineEpsilon;
        }

        Float weights[2] = {0};
        weights[0] = std::min(OneMinusEpsilon, errBounds[0] / (errBounds[0] + errBounds[1]));
        weights[1] = 1 - weights[0];
        
        uint32_t threshold = static_cast<uint32_t>(weights[0] * floatUintMax);

        // Randomly sample a children node
        int child = 0;
        if (currentU < threshold) {
            currentU = static_cast<uint32_t>((static_cast<float>(currentU) / weights[0]));
        } else {
            child = 1;

            currentU -= threshold;
            currentU = static_cast<uint32_t>((static_cast<float>(currentU) / weights[1]));
        }
        
        currentU ^= FastIntegerHash(nodeIndex);
        pmf *= weights[child];
        nodeIndex = childrenIndices[child];
        node = &tree.nodes[nodeIndex];

        estL = estL - estParent + clusterEst[0] + clusterEst[1];
        estParent = clusterEst[child];

        if (errBounds[child] < m_threshold * estL) {
            int representantLightIndex = representants[child]->childOrLightIndex;
            Float repIntensity = representants[child]->compactLightBounds.PhiOrI();
            return SampledLight(tree.lights[representantLightIndex], pmf, nodeIntensities[child] / repIntensity);
        }
    }

    return SampledLight(tree.lights[node->childOrLightIndex], pmf);
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
