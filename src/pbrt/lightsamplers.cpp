// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Contributions Copyright(c) 2026 Richard Kvasnica.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/lightsamplers.h>

#include <pbrt/paramdict.h>
#include <pbrt/options.h>

#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>

namespace pbrt {

std::string SampledLight::ToString() const {
    return StringPrintf("[ SampledLight light: %s p: %f ]",
                        light ? light.ToString().c_str() : "(nullptr)", p);
}

LightSampler LightSampler::Create(ERequiresShadowRays &nShadowRays, pstd::span<const Light> lights, const bool discretizedLights,
                                  const ParameterDictionary& params, Allocator alloc) {
    nShadowRays = E_DEFAULT_SHADOW_RAYS;
    std::string name = params.GetOneString("lightsampler", "bvh");
    if (lights.size() == 1)
        name = "uniform";

    if (name == "uniform")
        return alloc.new_object<UniformLightSampler>(lights, alloc);
    if (name == "power")
        return alloc.new_object<PowerLightSampler>(lights, alloc);
    if (name == "bvh")
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    if (name == "lightcuts") {
        nShadowRays = E_LIGHTCUTS_SHADOW_RAYS;
        if (discretizedLights) {
            return alloc.new_object<LightcutsLightSampler>(lights, alloc);
        }
        Error(R"(Cannot use lightcuts lightsampler without discretizing area lights. Using "slc" stochastic lightcuts.)");
        name = "slc";
    }
    if (name == "slc") {
        nShadowRays = E_LIGHTCUTS_SHADOW_RAYS;
        const Float threshold = params.GetOneFloat("lsParam1", 0.02f);
        return alloc.new_object<SLCLightSampler>(lights, alloc, threshold);
    }
    if (name == "hslc") {
        return alloc.new_object<HSLCLightSampler>(lights, alloc);
    }
    if (name == "rht") {
        nShadowRays = E_RHT_SHADOW_RAYS;
        const Float gamma = params.GetOneFloat("lsParam1", 0.2f);
        return alloc.new_object<RHTLightSampler>(lights, alloc, gamma);
    }
    if (name == "ltc") {
        const Float beta = params.GetOneFloat("lsParam1", 2.0f);
        const Float omega = params.GetOneFloat("lsParam2", Float(6) / 7);
        const Float gamma = params.GetOneFloat("lsParam3", 128.0f);
        return alloc.new_object<LTCLightSampler>(lights, alloc, beta, omega, gamma);
    }
    else if (name == "exhaustive")
        return alloc.new_object<ExhaustiveLightSampler>(lights, alloc);
    else {
        Error(R"(Light sample distribution type "%s" unknown. Using "bvh".)",
              name.c_str());
    }
    
    return alloc.new_object<BVHLightSampler>(lights, alloc);
}

std::string LightSampler::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

}  // namespace pbrt
