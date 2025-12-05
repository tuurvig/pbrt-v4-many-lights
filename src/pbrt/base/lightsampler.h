// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_LIGHTSAMPLER_H
#define PBRT_BASE_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

// SampledLight Definition
struct SampledLight {
    PBRT_CPU_GPU
    SampledLight(Light light, Float p = 0, Float scale = 1) :
        light(light), p(p), scale(scale) {}

    Light light;
    Float p = 0;
    Float scale = 1;
    std::string ToString() const;
};

struct LightPMF {
    PBRT_CPU_GPU
    LightPMF(Float pmf, Float scale = 1) : pmf(pmf), scale(scale) {}
    Float pmf = 0;
    Float scale = 1;
};

class UniformLightSampler;
class PowerLightSampler;
class BVHLightSampler;
class LightcutsLightSampler;
class ExhaustiveLightSampler;

// LightSampler Definition
class LightSampler : public TaggedPointer<UniformLightSampler,
                                          PowerLightSampler,
                                          ExhaustiveLightSampler,
                                          BVHLightSampler,
                                          LightcutsLightSampler> {
  public:
    // LightSampler Interface
    using TaggedPointer::TaggedPointer;

    static LightSampler Create(const std::string &name, pstd::span<const Light> lights, bool discretizedLights,
                               Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(Float u) const;
    PBRT_CPU_GPU inline LightPMF PMF(Light light) const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, Float u) const;
    PBRT_CPU_GPU inline LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, Light light) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_LIGHTSAMPLER_H
