// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_POWER_LIGHTSAMPLER_H
#define  PBRT_POWER_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/containers.h>

namespace pbrt {

// PowerLightSampler Definition
class PowerLightSampler {
  public:
    // PowerLightSampler Public Methods
    PowerLightSampler(pstd::span<const Light> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (!m_aliasTable.size()) {
            return {};
        }
        Float pmf;
        int lightIndex = m_aliasTable.Sample(u, &pmf);
        return SampledLight(m_lights[lightIndex], pmf);
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &/*ctx*/, const BSDF* /*bsdf*/, uint32_t /*seed*/, Float u) const {
        return Sample(u);
    }

    PBRT_CPU_GPU
    LightPMF PMF(Light light) const {
        if (!m_aliasTable.size()) {
            return 0;
        }

        return LightPMF(m_aliasTable.PMF(m_lightToIndex[light]));
    }

    PBRT_CPU_GPU
    LightPMF PMF(const LightSampleContext &/*ctx*/, const BSDF* /*bsdf*/, uint32_t /*seed*/, Light light) const { return PMF(light); }
    
    template <int NSamples, typename ScatterEval>
    PBRT_CPU_GPU PBRT_NOINLINE void SampleLd(CountedArray<SampledLd, NSamples>& samples, const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const {
        pstd::optional<SampledLight> sampledLight = Sample(u);
        if (!sampledLight) {
            return;
        }

        Light light = sampledLight->light;
        DCHECK(light && sampledLight->p != 0);
        pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
        if (!ls || !ls->L || ls->pdf == 0)
            return;

        Float lightPDF = sampledLight->p * ls->pdf;
        ls->L *= sampledLight->scale;
        
        Float scatterPDF = 0;
        SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));

        samples.Add(SampledLd(f_hat * ls->L, ls->pLight, lightPDF, scatterPDF));
    }

    std::string ToString() const;

  private:
    // PowerLightSampler Private Members
    pstd::vector<Light> m_lights;
    HashMap<Light, size_t> m_lightToIndex;
    AliasTable m_aliasTable;
};

}

#endif  // PBRT_POWER_LIGHTSAMPLER_H
