// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Contributions Copyright(c) 2026 Richard Kvasnica.
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
    SampledLight(Light light, Float p = 0, uint32_t hint = std::numeric_limits<uint32_t>::max(), Float pLearning = 1) :
        light(light), p(p), hint(hint), pLearning(pLearning) {}

    Light light;
    Float p;
    uint32_t hint;
    Float pLearning;

    std::string ToString() const;
};

struct LightPMF {
    PBRT_CPU_GPU
    LightPMF(Float pmf, Float scale = 1) : pmf(pmf), scale(scale) {}
    Float pmf = 0;
    Float scale = 1;
};

struct SampledLd {
    SampledLd() = default;

    PBRT_CPU_GPU SampledLd(const SampledSpectrum& s, Light light, const Interaction& intr, Float lightPDF, Float scatterPDF, uint32_t lightSamplerHint = std::numeric_limits<uint32_t>::max(), Float pdfCancellationFactor = 0) :
        Ld(s), pLight(intr.pi), nLight(intr.n), lightPDF(lightPDF), scatterPDF(scatterPDF), light(light), lightSamplerHint(lightSamplerHint), pdfCancellationFactor(pdfCancellationFactor)  {}

    PBRT_CPU_GPU
    Ray SpawnShadowRay(const Interaction &from) const {
        Ray r = pbrt::SpawnRayTo(from.pi, from.n, from.time, pLight, nLight);
        r.medium = from.GetMedium(r.d);
        return r;
    }

    PBRT_CPU_GPU
    Ray SpawnShadowRay(Point3fi pFrom, Normal3f nFrom, Float time) const {
        return pbrt::SpawnRayTo(pFrom, nFrom, time, pLight, nLight);
    }

    SampledSpectrum Ld;
    Point3fi pLight;
    Normal3f nLight;
    Float lightPDF;
    Float scatterPDF;
    Light light;
    uint32_t lightSamplerHint;
    Float pdfCancellationFactor;
};

#ifndef PBRT_RHT_F_SAMPLES
#define PBRT_RHT_F_SAMPLES 2
#endif

enum ERequiresShadowRays : int {
    E_DEFAULT_SHADOW_RAYS = 1,
    E_RHT_SHADOW_RAYS = PBRT_RHT_F_SAMPLES,
    E_LIGHTCUTS_SHADOW_RAYS = PBRT_LIGHTCUTS_CUT_SIZE
};

class UniformLightSampler;
class PowerLightSampler;
class ExhaustiveLightSampler;

class BVHLightSampler;
class LightcutsLightSampler;
class SLCLightSampler;
class HSLCLightSampler;
class RHTLightSampler;
class LTCLightSampler;

// LightSampler Definition
class LightSampler : public TaggedPointer<UniformLightSampler,
                                          PowerLightSampler,
                                          ExhaustiveLightSampler,
                                          BVHLightSampler,
                                          LightcutsLightSampler,
                                          SLCLightSampler,
                                          HSLCLightSampler,
                                          RHTLightSampler,
                                          LTCLightSampler> {
  public:
    // LightSampler Interface
    using TaggedPointer::TaggedPointer;

    static LightSampler Create(ERequiresShadowRays& nShadowRays, const std::string &name, pstd::span<const Light> lights, bool discretizedLights,
                               Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(Float u) const;
    PBRT_CPU_GPU inline LightPMF PMF(Light light) const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Float u) const;
    PBRT_CPU_GPU inline LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Light light) const;

    template <int NSamples, typename ScatterEval>
    PBRT_CPU_GPU inline void SampleLd(CountedArray<SampledLd, NSamples>& samples, const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_LIGHTSAMPLER_H
