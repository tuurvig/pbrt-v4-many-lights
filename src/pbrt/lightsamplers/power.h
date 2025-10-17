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
        return SampledLight{m_lights[lightIndex], pmf};
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        if (!m_aliasTable.size()) {
            return 0;
        }

        return m_aliasTable.PMF(m_lightToIndex[light]);
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
        return Sample(u);
    }

    PBRT_CPU_GPU
    Float PMF(const LightSampleContext &ctx, Light light) const { return PMF(light); }

    std::string ToString() const;

  private:
    // PowerLightSampler Private Members
    pstd::vector<Light> m_lights;
    HashMap<Light, size_t> m_lightToIndex;
    AliasTable m_aliasTable;
};

}

#endif  // PBRT_POWER_LIGHTSAMPLER_H
