// rht.h - RHTLightSampler class is Copyright(c) 2025-2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt and lightcuts.h source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_RHT_LIGHTSAMPLER_H
#define  PBRT_RHT_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/bsdf.h>

#include <pbrt/util/manylights.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/math.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/containers.h>

namespace pbrt {

// Resampled Hierarchic Tree Light Sampler Definition
class RHTLightSampler {
  public:
    // Resampled Hierarchic Tree Light Sampler Public Methods
    RHTLightSampler(pstd::span<const Light> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Float u) const {
        Float pmf = 1;
        if (!m_infiniteLights.empty()) {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.leaves.size(), pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }
        
        if (m_tree.innerNodes.empty())
            return {};

        // Declare common variables for light BVH traversal
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;

        int nodeIndex = 0;

        return Sample(u);
    }

    PBRT_CPU_GPU
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, Light light) const {
        return PMF(light);
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        Float pmf = 1;
        {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.leaves.size(), pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        uint32_t index = std::min<uint32_t>(u * static_cast<Float>(m_tree.leaves.size()), m_tree.leaves.size() - 1);
        pmf /= m_tree.leaves.size();
        return SampledLight{m_tree.leaves[index].light, pmf};
    }

    PBRT_CPU_GPU
    LightPMF PMF(Light light) const {
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = InfiniteLightSimplePMF(m_infiniteLights, m_tree.leaves.size());
        
        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return pInfinite;

        if (m_tree.leaves.empty())
            return 0;

        Float pmf = 1 - pInfinite;
        return pmf / m_tree.leaves.size(); 
    }

    std::string ToString() const;

  private:
#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildLightTreeGPU(std::vector<RHTBuildContainer> &lights);
#endif

    // Resampled Hierarchic Tree Light Sampler Private Members
    ResampledTree m_tree;
    pstd::vector<Light> m_infiniteLights;
    HashMap<Light, uint32_t> m_lightToBitTrail;
};

}
#endif // PBRT_RHT_LIGHTSAMPLER_H
