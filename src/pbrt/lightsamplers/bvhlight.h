// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_BVH_LIGHTSAMPLER_H
#define  PBRT_BVH_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>

#include <pbrt/util/manylights.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/containers.h>

namespace pbrt {

// BVHLightSampler Definition
class BVHLightSampler {
  public:
    // BVHLightSampler Public Methods
    BVHLightSampler(pstd::span<const Light> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* /*bsdf*/, Float u) const {
        Float pmf = 1;
        if (!m_infiniteLights.empty()) {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_nodes.size(), pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        // Traverse light BVH to sample light
        if (m_nodes.empty())
            return {};

        // Declare common variables for light BVH traversal
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        int nodeIndex = 0;

        constexpr Float floatUintMax = 0x1p32f;
        uint32_t currentU = static_cast<uint32_t>(u * floatUintMax);

        Float importance = 0.f;
        while (true) {
            // Process light BVH node for light sampling
            LightBVHNode node = m_nodes[nodeIndex];
            if (!node.isLeaf) {
                uint32_t childrenIndices[2] = {nodeIndex + 1, node.childOrLightIndex};

                // Compute light BVH child node importances
                const LightBVHNode *children[2] = {&m_nodes[childrenIndices[0]],
                                                   &m_nodes[childrenIndices[1]]};
                Float ci[2] = {
                    children[0]->lightBounds.Importance(p, n, m_allLightBounds),
                    children[1]->lightBounds.Importance(p, n, m_allLightBounds)};

                if (ci[0] < MachineEpsilon) {
                    if (ci[1] < MachineEpsilon) {
                        return {};
                    }
                    nodeIndex = childrenIndices[1];
                    importance = ci[1];
                    continue;
                } else if (ci[1] < MachineEpsilon){
                    nodeIndex = childrenIndices[1];
                    importance = ci[0];
                    continue;
                }

                Float weights[2] = {0};
                weights[0] = std::min(OneMinusEpsilon, ci[0] / (ci[0] + ci[1]));
                weights[1] = 1 - weights[0];

                uint32_t threshold = static_cast<uint32_t>(weights[0] * floatUintMax);
                int child = 0;
                if (currentU < threshold) {
                    currentU = static_cast<uint32_t>((static_cast<float>(currentU) / weights[0]));
                } else {
                    child = 1;

                    currentU -= threshold;
                    currentU = static_cast<uint32_t>((static_cast<float>(currentU) / weights[1]));
                }

                nodeIndex = childrenIndices[child];
                importance = ci[child];
                currentU ^= FastIntegerHash(nodeIndex);
                pmf *= weights[child];

            } else {
                //Confirm light has nonzero importance before returning light sample
                //Float imp = node.lightBounds.Importance(p, n, m_allLightBounds);
                if (nodeIndex > 0)
                    DCHECK_GT(importance, 0);
                if (nodeIndex > 0 || importance > 0)
                    return SampledLight(m_lights[node.childOrLightIndex], pmf);
                return {};
            }
        }
    }

    PBRT_CPU_GPU
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* /*bsdf*/, Light light) const {
        // Handle infinite _light_ PMF computation
        if (!m_lightToBitTrail.HasKey(light))
            return InfiniteLightSimplePMF(m_infiniteLights, m_nodes.size());

        // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = m_lightToBitTrail[light];
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_nodes.empty() ? 0 : 1));

        Float pmf = 1 - pInfinite;
        int nodeIndex = 0;

        // Compute light's PMF by walking down tree nodes to the light
        while (true) {
            const LightBVHNode *node = &m_nodes[nodeIndex];
            if (node->isLeaf) {
                DCHECK_EQ(light, m_lights[node->childOrLightIndex]);
                return pmf;
            }
            // Compute child importances and update PMF for current node
            const LightBVHNode *child0 = &m_nodes[nodeIndex + 1];
            const LightBVHNode *child1 = &m_nodes[node->childOrLightIndex];
            Float ci[2] = {child0->lightBounds.Importance(p, n, m_allLightBounds),
                           child1->lightBounds.Importance(p, n, m_allLightBounds)};
            DCHECK_GT(ci[bitTrail & 1], 0);
            pmf *= ci[bitTrail & 1] / (ci[0] + ci[1]);

            // Use _bitTrail_ to find next node index and update its value
            nodeIndex = (bitTrail & 1) ? node->childOrLightIndex : (nodeIndex + 1);
            bitTrail >>= 1;
        }
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (m_lights.empty())
            return {};
        int lightIndex = std::min<int>(u * m_lights.size(), m_lights.size() - 1);
        return SampledLight{m_lights[lightIndex], 1.f / m_lights.size()};
    }

    PBRT_CPU_GPU
    LightPMF PMF(Light light) const {
        if (m_lights.empty())
            return 0;
        return 1.f / m_lights.size();
    }

    std::string ToString() const;

  private:
    // BVHLightSampler Private Methods
    LightBVHBuildContainer buildBVH(
        std::vector<LightBVHBuildContainer> &bvhLights, int start, int end,
        uint32_t bitTrail, int depth);

#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildBVHGPU(std::vector<LightBVHBuildContainer> &bvhLights);
#endif

    PBRT_CPU_GPU
    Float EvaluateCost(const LightBounds &b, const Bounds3f &bounds, int dim) const {
        // Return complete cost estimate for _LightBounds_
        Float Kr = MaxComponentValue(bounds.Diagonal()) / bounds.Diagonal()[dim];
        return CostSAOH(b) * Kr;
    }

    // BVHLightSampler Private Members
    pstd::vector<Light> m_lights;
    pstd::vector<Light> m_infiniteLights;
    Bounds3f m_allLightBounds;
    pstd::vector<LightBVHNode> m_nodes;
    HashMap<Light, uint32_t> m_lightToBitTrail;
};

}

#endif // PBRT_BVH_LIGHTSAMPLER_H
