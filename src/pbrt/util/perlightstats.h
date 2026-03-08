// perlightstats.h is Copyright(c) 2026 Richard Kvasnica.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_PERLIGHTSTATS_H
#define PBRT_UTIL_PERLIGHTSTATS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/util/pstd.h>

#include <string>
#include <vector>

namespace pbrt {
void StatsEnablePerLightStatistics(pstd::span<const Light> lights,
                                   const std::string &outputBaseName);

void ReportLightSampleBeforeShadow(Light light);
void ReportLightSampleAfterShadowVisible(Light light);

void StatsWritePerLightStatistics();

}  // namespace pbrt

#endif  // PBRT_UTIL_PERLIGHTSTATS_H
