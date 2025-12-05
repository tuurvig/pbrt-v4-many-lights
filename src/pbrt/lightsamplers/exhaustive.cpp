#include "exhaustive.h"

namespace pbrt{
// ExhaustiveLightSampler Method Definitions
ExhaustiveLightSampler::ExhaustiveLightSampler(pstd::span<const Light> lights,
                                               Allocator alloc)
    : lights(lights.begin(), lights.end(), alloc),
      boundedLights(alloc),
      infiniteLights(alloc),
      lightBounds(alloc),
      lightToBoundedIndex(alloc) {
    for (const auto &light : lights) {
        if (pstd::optional<LightBounds> lb = light.Bounds(); lb) {
            lightToBoundedIndex.Insert(light, boundedLights.size());
            lightBounds.push_back(*lb);
            boundedLights.push_back(light);
        } else
            infiniteLights.push_back(light);
    }
}

PBRT_CPU_GPU pstd::optional<SampledLight> ExhaustiveLightSampler::Sample(const LightSampleContext &ctx,
                                                            const BSDF* /*bsdf*/, Float u) const {
    Float pInfinite = Float(infiniteLights.size()) /
                      Float(infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));

    // Note: shared with BVH light sampler...
    if (u < pInfinite) {
        u /= pInfinite;
        int index = std::min<int>(u * infiniteLights.size(), infiniteLights.size() - 1);
        Float pdf = pInfinite * 1.f / infiniteLights.size();
        return SampledLight{infiniteLights[index], pdf};
    } else {
        u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);

        uint64_t seed = MixBits(FloatToBits(u));
        WeightedReservoirSampler<Light> wrs(seed);

        for (size_t i = 0; i < boundedLights.size(); ++i)
            wrs.Add(boundedLights[i], lightBounds[i].Importance(ctx.p(), ctx.n));

        if (!wrs.HasSample())
            return {};

        Float pdf = (1.f - pInfinite) * wrs.SampleProbability();
        return SampledLight{wrs.GetSample(), pdf};
    }
}

PBRT_CPU_GPU LightPMF ExhaustiveLightSampler::PMF(const LightSampleContext &ctx, const BSDF* /*bsdf*/, Light light) const {
    if (!lightToBoundedIndex.HasKey(light))
        return 1.f / (infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));

    Float importanceSum = 0;
    Float lightImportance = 0;
    for (size_t i = 0; i < boundedLights.size(); ++i) {
        Float importance = lightBounds[i].Importance(ctx.p(), ctx.n);
        importanceSum += importance;
        if (light == boundedLights[i])
            lightImportance = importance;
    }
    Float pInfinite = Float(infiniteLights.size()) /
                      Float(infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));
    Float pdf = lightImportance / importanceSum * (1. - pInfinite);
    return pdf;
}

std::string ExhaustiveLightSampler::ToString() const {
    return StringPrintf("[ ExhaustiveLightSampler lightBounds: %s]", lightBounds);
}

}
