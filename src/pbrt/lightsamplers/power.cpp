#include "power.h"

#include <pbrt/util/spectrum.h>

namespace pbrt{

///////////////////////////////////////////////////////////////////////////
// PowerLightSampler

// PowerLightSampler Method Definitions
PowerLightSampler::PowerLightSampler(pstd::span<const Light> lights, Allocator alloc)
    : m_lights(lights.begin(), lights.end(), alloc),
      m_lightToIndex(alloc),
      m_aliasTable(alloc) {
    if (lights.empty())
        return;

    // Initialize _lightToIndex_ hash table
    for (size_t i = 0; i < lights.size(); ++i)
        m_lightToIndex.Insert(lights[i], i);

    // Compute lights' power and initialize alias table
    pstd::vector<Float> lightPower;
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.5f);
    for (const auto &light : lights) {
        SampledSpectrum phi = SafeDiv(light.Phi(lambda), lambda.PDF());
        lightPower.push_back(phi.Average());
    }
    if (std::accumulate(lightPower.begin(), lightPower.end(), 0.f) == 0.f)
        std::fill(lightPower.begin(), lightPower.end(), 1.f);
    m_aliasTable = AliasTable(lightPower, alloc);
}

std::string PowerLightSampler::ToString() const {
    return StringPrintf("[ PowerLightSampler aliasTable: %s ]", m_aliasTable);
}

}
