// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_EXHAUSTIVE_LIGHTSAMPLER_H
#define  PBRT_EXHAUSTIVE_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/containers.h>

namespace pbrt {

// ExhaustiveLightSampler Definition
class ExhaustiveLightSampler {
  public:
    ExhaustiveLightSampler(pstd::span<const Light> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, Float u) const;

    PBRT_CPU_GPU
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, Light light) const;

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (lights.empty())
            return {};

        int lightIndex = std::min<int>(u * lights.size(), lights.size() - 1);
        return SampledLight{lights[lightIndex], 1.f / lights.size()};
    }

    PBRT_CPU_GPU
    LightPMF PMF(Light light) const {
        if (lights.empty())
            return 0;
        return 1.f / lights.size();
    }

    std::string ToString() const;

  private:
    pstd::vector<Light> lights, boundedLights, infiniteLights;
    pstd::vector<LightBounds> lightBounds;
    HashMap<Light, size_t> lightToBoundedIndex;
};
}

#endif // PBRT_EXHAUSTIVE_LIGHTSAMPLER_H
