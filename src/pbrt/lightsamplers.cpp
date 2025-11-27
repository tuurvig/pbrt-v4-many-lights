// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/lightsamplers.h>

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
#include <pbrt/options.h>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>

namespace pbrt {

std::string SampledLight::ToString() const {
    return StringPrintf("[ SampledLight light: %s p: %f ]",
                        light ? light.ToString().c_str() : "(nullptr)", p);
}

LightSampler LightSampler::Create(const std::string &name, pstd::span<const Light> lights, bool discretizedLights,
                                  Allocator alloc) {
    if (name == "uniform")
        return alloc.new_object<UniformLightSampler>(lights, alloc);
    else if (name == "power")
        return alloc.new_object<PowerLightSampler>(lights, alloc);
    else if (name == "bvh")
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    else if (name == "lightcuts") {
        if (discretizedLights) {
            return alloc.new_object<LightcutsLightSampler>(lights, alloc);
        }
        Error(R"(Cannot use lightcuts lightsampler without discretizing area lights. Using "bvh".)");
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
