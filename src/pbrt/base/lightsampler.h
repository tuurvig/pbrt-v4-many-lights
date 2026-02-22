// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_LIGHTSAMPLER_H
#define PBRT_BASE_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>
#include <pbrt/interaction.h>

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

struct SampledLd {
    PBRT_CPU_GPU SampledLd(const SampledSpectrum& s, const Interaction& intr, Float lightPDF, Float scatterPDF) :
        Ld(s), pLight(intr), lightPDF(lightPDF), scatterPDF(scatterPDF) {}
    SampledSpectrum Ld;
    Interaction pLight;
    Float lightPDF;
    Float scatterPDF;
    
};

class UniformLightSampler;
class PowerLightSampler;
class ExhaustiveLightSampler;

class BVHLightSampler;
class LightcutsLightSampler;
class SLCLightSampler;
class HSLCLightSampler;
class RHTLightSampler;

// LightSampler Definition
class LightSampler : public TaggedPointer<UniformLightSampler,
                                          PowerLightSampler,
                                          ExhaustiveLightSampler,
                                          BVHLightSampler,
                                          LightcutsLightSampler,
                                          SLCLightSampler,
                                          HSLCLightSampler,
                                          RHTLightSampler> {
  public:
    // LightSampler Interface
    using TaggedPointer::TaggedPointer;

    static LightSampler Create(const std::string &name, pstd::span<const Light> lights, bool discretizedLights,
                               Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(Float u) const;
    PBRT_CPU_GPU inline LightPMF PMF(Light light) const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Float u) const;
    PBRT_CPU_GPU inline LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Light light) const;

    template <typename ScatterEval>
    PBRT_CPU_GPU inline pstd::optional<SampledLd> SampleLd(const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_LIGHTSAMPLER_H
