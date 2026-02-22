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

    PBRT_CPU_GPU PBRT_NOINLINE
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* /*bsdf*/, uint32_t seed, Float u) const {
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

        while (true) {
            // Process light BVH node for light sampling
            LightBVHNode node = m_nodes[nodeIndex];
            if (!node.isLeaf) {
                // Compute light BVH child node importances
                const LightBVHNode *children[2] = {&m_nodes[nodeIndex + 1],
                                                   &m_nodes[node.childOrLightIndex]};
                Float ci[2] = {
                    children[0]->lightBounds.Importance(p, n, m_allLightBounds),
                    children[1]->lightBounds.Importance(p, n, m_allLightBounds)};
                if (ci[0] == 0 && ci[1] == 0)
                    return {};

                // Randomly sample light BVH child node
                Float nodePMF;
                int child = SampleDiscrete(ci, u, &nodePMF, &u);
                pmf *= nodePMF;
                nodeIndex = (child == 0) ? (nodeIndex + 1) : node.childOrLightIndex;

                const Float scrambleOffset = HashFloat(nodeIndex, seed);
                u += scrambleOffset;
                if (u >= 1) u -= 1;
            } else {
                // Confirm light has nonzero importance before returning light sample
                if (nodeIndex > 0)
                    DCHECK_GT(node.lightBounds.Importance(p, n, m_allLightBounds), 0);
                if (nodeIndex > 0 ||
                    node.lightBounds.Importance(p, n, m_allLightBounds) > 0)
                    return SampledLight(m_lights[node.childOrLightIndex], pmf);
                return {};
            }
        }
    }

    PBRT_CPU_GPU PBRT_NOINLINE
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* /*bsdf*/, uint32_t /*seed*/, Light light) const {
        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return InfiniteLightSimplePMF(m_infiniteLights, m_nodes.size());

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_nodes.empty() ? 0 : 1));

        // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = m_lightToBitTrail[light];
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        
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
        if (m_nodes.empty() || m_lights.empty())
            return {};

        int lightIndex = std::min<int>(u * m_lights.size(), m_lights.size() - 1);
        Light light = m_lights[lightIndex];
        LightPMF lpmf = PMF(light);
        return SampledLight{light, lpmf.pmf, lpmf.scale};
    }

    PBRT_CPU_GPU
    LightPMF PMF(Light light) const {
        Float pInfinite = InfiniteLightSimplePMF(m_infiniteLights, m_nodes.size());

        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return pInfinite;

        Float pmf = 1 - pInfinite;

        if (m_lights.empty())
            return 0;

        return pmf / (m_lights.size() - m_infiniteLights.size());
    }

    template <typename ScatterEval>
    PBRT_CPU_GPU PBRT_NOINLINE pstd::optional<SampledLd> SampleLd(const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const {
        pstd::optional<SampledLight> sampledLight = Sample(ctx, bsdf, seed, u);
        if (!sampledLight) {
            return {};
        }

        Light light = sampledLight->light;
        DCHECK(light && sampledLight->p != 0);
        pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
        if (!ls || !ls->L || ls->pdf == 0)
            return {};

        Float lightPDF = sampledLight->p * ls->pdf;
        ls->L *= sampledLight->scale;
        
        Float scatterPDF = 0;
        SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));
        return SampledLd(f_hat * ls->L, ls->pLight, lightPDF, scatterPDF);
    }

    std::string ToString() const;

  private:
    // BVHLightSampler Private Methods
#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildBVHGPU(std::vector<LightBVHBuildContainer> &bvhLights);
#endif

    // BVHLightSampler Private Members
    pstd::vector<Light> m_lights;
    pstd::vector<Light> m_infiniteLights;
    Bounds3f m_allLightBounds;
    pstd::vector<LightBVHNode> m_nodes;
    HashMap<Light, uint32_t> m_lightToBitTrail;
};

}

#endif // PBRT_BVH_LIGHTSAMPLER_H
