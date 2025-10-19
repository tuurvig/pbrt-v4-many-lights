// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_LIGHTSAMPLERS_H
#define PBRT_LIGHTSAMPLERS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>  // LightBounds. Should that live elsewhere?
#include <pbrt/util/containers.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/vecmath.h>

#include <pbrt/lightsamplers/uniform.h>
#include <pbrt/lightsamplers/power.h>
#include <pbrt/lightsamplers/bvhlight.h>
#include <pbrt/lightsamplers/exhaustive.h>

#include <pbrt/lightsamplers/lightcuts.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace pbrt {

PBRT_CPU_GPU inline pstd::optional<SampledLight> LightSampler::Sample(const LightSampleContext &ctx,
                                                         Float u) const {
    auto s = [&](auto ptr) { return ptr->Sample(ctx, u); };
    return Dispatch(s);
}

PBRT_CPU_GPU inline Float LightSampler::PMF(const LightSampleContext &ctx, Light light) const {
    auto pdf = [&](auto ptr) { return ptr->PMF(ctx, light); };
    return Dispatch(pdf);
}

PBRT_CPU_GPU inline pstd::optional<SampledLight> LightSampler::Sample(Float u) const {
    auto sample = [&](auto ptr) { return ptr->Sample(u); };
    return Dispatch(sample);
}

PBRT_CPU_GPU inline Float LightSampler::PMF(Light light) const {
    auto pdf = [&](auto ptr) { return ptr->PMF(light); };
    return Dispatch(pdf);
}

}  // namespace pbrt

#endif  // PBRT_LIGHTSAMPLERS_H
