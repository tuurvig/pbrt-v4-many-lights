// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_UNIFORM_LIGHTSAMPLER_H
#define  PBRT_UNIFORM_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>

#include <pbrt/util/pstd.h>

namespace pbrt {

// UniformLightSampler Definition
class UniformLightSampler {
public:
    UniformLightSampler(pstd::span<const Light> lights, Allocator alloc)
        : m_lights(lights.begin(), lights.end(), alloc) {}

    PBRT_CPU_GPU pstd::optional<SampledLight> Sample(Float u) const {
        if (m_lights.empty()) {
            return {};
        }
        int lightIndex = std::min<int>(u * m_lights.size(), m_lights.size() - 1);
        return SampledLight(m_lights[lightIndex], 1.f / static_cast<float>(m_lights.size()));
    }

    PBRT_CPU_GPU pstd::optional<SampledLight> Sample(const LightSampleContext & /*ctx*/, const BSDF* /*bsdf*/, Float u) const {
        return Sample(u);
    }

    PBRT_CPU_GPU LightPMF PMF(Light light) const {
        return LightPMF(m_lights.empty() ? 0 : 1.f / m_lights.size());
    }

    PBRT_CPU_GPU LightPMF PMF(const LightSampleContext & /*ctx*/, const BSDF* /*bsdf*/, Light light) const { 
        return PMF(light);
    }

    std::string ToString() const { return "UniformLightSampler"; }

private:
    // UniformLightSampler Private Members
    pstd::vector<Light> m_lights;
};

}

#endif  // PBRT_UNIFORM_LIGHTSAMPLER_H
