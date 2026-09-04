// Copyright(c) 2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/perlightstats.h>

#include <pbrt/util/file.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>

#include <atomic>
#include <cinttypes>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace pbrt {

struct LightHash {
    size_t operator()(Light light) const noexcept { return Hash(light.ptr()); }
};

struct LightStatsGlobalState {
    std::mutex mtx;
    std::vector<Light> lights;
    std::unordered_map<Light, size_t, LightHash> lightToIndex;
    std::string outputBaseName;
    bool enabled = false;

    std::vector<uint64_t> totalBeforeShadow;
    std::vector<uint64_t> totalAfterShadow;
    std::vector<double> totalContribBeforeShadow;
    std::vector<double> totalContribAfterShadow;

    void Accumulate(const std::vector<uint64_t>& threadBefore,
                    const std::vector<uint64_t>& threadAfter,
                    const std::vector<double>& threadContribBefore,
                    const std::vector<double>& threadContribAfter) {
        std::lock_guard<std::mutex> lock(mtx);
        if (totalBeforeShadow.size() < threadBefore.size()) {
            totalBeforeShadow.resize(threadBefore.size(), 0);
            totalAfterShadow.resize(threadAfter.size(), 0);
            totalContribBeforeShadow.resize(threadContribBefore.size(), 0.0);
            totalContribAfterShadow.resize(threadContribAfter.size(), 0.0);
        }
        for (size_t i = 0; i < threadBefore.size(); ++i) {
            totalBeforeShadow[i] += threadBefore[i];
            totalAfterShadow[i] += threadAfter[i];
            totalContribBeforeShadow[i] += threadContribBefore[i];
            totalContribAfterShadow[i] += threadContribAfter[i];
        }
    }
};

static LightStatsGlobalState statsState;

struct ThreadLightStats {
    std::vector<uint64_t> beforeShadow;
    std::vector<uint64_t> afterShadow;
    std::vector<double> contribBeforeShadow;
    std::vector<double> contribAfterShadow;

    void ResizeIfNecessary(size_t size) {
        if (beforeShadow.size() < size) {
            beforeShadow.resize(size, 0);
            afterShadow.resize(size, 0);
            contribBeforeShadow.resize(size, 0.0);
            contribAfterShadow.resize(size, 0.0);
        }
    }

    void Clear() {
        std::fill(beforeShadow.begin(), beforeShadow.end(), 0);
        std::fill(afterShadow.begin(), afterShadow.end(), 0);
        std::fill(contribBeforeShadow.begin(), contribBeforeShadow.end(), 0.0);
        std::fill(contribAfterShadow.begin(), contribAfterShadow.end(), 0.0);
    }
};

static thread_local ThreadLightStats threadStats;

// This callback is triggered automatically when the main thread calls ReportThreadStats
static StatRegisterer lightStatRegisterer([](StatsAccumulator& accum) {
    if (statsState.enabled) {
        statsState.Accumulate(threadStats.beforeShadow, threadStats.afterShadow,
                              threadStats.contribBeforeShadow,
                              threadStats.contribAfterShadow);
        threadStats.Clear();
    }
});

void StatsEnablePerLightStatistics(pstd::span<const Light> lights,
                                   const std::string &outputBaseName) {
    std::lock_guard<std::mutex> lock(statsState.mtx);

    statsState.lights.assign(lights.begin(), lights.end());
    statsState.lightToIndex.clear();
    statsState.lightToIndex.reserve(statsState.lights.size());
    for (size_t i = 0; i < statsState.lights.size(); ++i)
        statsState.lightToIndex.emplace(statsState.lights[i], i);

    statsState.outputBaseName = outputBaseName;
    statsState.totalBeforeShadow.assign(lights.size(), 0);
    statsState.totalAfterShadow.assign(lights.size(), 0);
    statsState.totalContribBeforeShadow.assign(lights.size(), 0.0);
    statsState.totalContribAfterShadow.assign(lights.size(), 0.0);
    statsState.enabled = true;
}

void ReportLightSampleBeforeShadow(Light light, Float contribution) {
    if (!statsState.enabled) return;

    auto iter = statsState.lightToIndex.find(light);
    if (iter == statsState.lightToIndex.end())
        return;

    const size_t index = iter->second;
    threadStats.ResizeIfNecessary(statsState.lights.size());
    ++threadStats.beforeShadow[index];
    if (IsFinite(contribution))
        threadStats.contribBeforeShadow[index] += contribution;
}

void ReportLightSampleAfterShadowVisible(Light light, Float contribution) {
    if (!statsState.enabled) return;

    auto iter = statsState.lightToIndex.find(light);
    if (iter == statsState.lightToIndex.end())
        return;

    const size_t index = iter->second;
    threadStats.ResizeIfNecessary(statsState.lights.size());
    ++threadStats.afterShadow[index];
    if (IsFinite(contribution))
        threadStats.contribAfterShadow[index] += contribution;
}

// Assumes ReportThreadStats was called for all worker threads before this point
void StatsWritePerLightStatistics() {
    std::lock_guard<std::mutex> lock(statsState.mtx);
    if (!statsState.enabled) return;

    uint64_t sumBeforeShadow = 0;
    uint64_t sumAfterShadow = 0;
    double sumContribBeforeShadow = 0;
    double sumContribAfterShadow = 0;
    uint64_t neverSampledLights = 0;
    uint64_t sampledNeverVisitedLights = 0;

    for (size_t i = 0; i < statsState.lights.size(); ++i) {
        sumBeforeShadow += statsState.totalBeforeShadow[i];
        sumAfterShadow += statsState.totalAfterShadow[i];
        sumContribBeforeShadow += statsState.totalContribBeforeShadow[i];
        sumContribAfterShadow += statsState.totalContribAfterShadow[i];
        if (statsState.totalBeforeShadow[i] == 0) {
            ++neverSampledLights;
        }
        else if (statsState.totalAfterShadow[i] == 0) {
            ++sampledNeverVisitedLights;
        }
    }

    std::string filename = statsState.outputBaseName.empty() ?
        "lightstats.csv" : statsState.outputBaseName + "-lightstats.csv";
    FILE *fp = FOpenWrite(filename);
    if (!fp) {
        LOG_ERROR("%s: unable to write light statistics file", filename.c_str());
    } else {
        const double visibilityRate =
            sumBeforeShadow > 0 ? double(sumAfterShadow) / double(sumBeforeShadow)
                                : 0.0;
        fprintf(fp, "# total_lights: %" PRIu64 "\n",
                uint64_t(statsState.lights.size()));
        fprintf(fp, "# total_before_shadow: %" PRIu64 "\n", sumBeforeShadow);
        fprintf(fp, "# total_after_shadow_visible: %" PRIu64 "\n",
                sumAfterShadow);
        fprintf(fp, "# global_visibility_rate: %.9f\n", visibilityRate);
        fprintf(fp, "# total_contrib_before_shadow: %.9g\n", sumContribBeforeShadow);
        fprintf(fp, "# total_contrib_after_shadow_visible: %.9g\n",
                sumContribAfterShadow);
        fprintf(fp, "# global_contrib_visibility_rate: %.9f\n",
                sumContribBeforeShadow > 0
                    ? sumContribAfterShadow / sumContribBeforeShadow
                    : 0.0);
        fprintf(fp, "# never_sampled_lights: %" PRIu64 "\n", neverSampledLights);
        fprintf(fp, "# sampled_but_never_visible_lights: %" PRIu64 "\n",
                sampledNeverVisitedLights);
        fprintf(fp,"light_index;light_tag;before_shadow;after_shadow_visible;visibility_rate;contrib_before_shadow;contrib_after_shadow_visible;contrib_visibility_rate\n");

        for (size_t i = 0; i < statsState.lights.size(); ++i) {
            const Light light = statsState.lights[i];
            const double lightVisibilityRate = statsState.totalBeforeShadow[i] > 0 ?
                    static_cast<double>(statsState.totalAfterShadow[i]) / static_cast<double>(statsState.totalBeforeShadow[i]) : 0.0;
            const double contribBefore = statsState.totalContribBeforeShadow[i];
            const double contribAfter = statsState.totalContribAfterShadow[i];
            const double contribVisibilityRate =
                contribBefore > 0 ? contribAfter / contribBefore : 0.0;

            fprintf(fp, "%" PRIu64 ";%u;%" PRIu64 ";%" PRIu64 ";%.9f;%.9g;%.9g;%.9f\n",
                    uint64_t(i), light.Tag(), statsState.totalBeforeShadow[i], statsState.totalAfterShadow[i], lightVisibilityRate,
                    contribBefore, contribAfter, contribVisibilityRate);
        }

        fclose(fp);
    }

    statsState.enabled = false;
    statsState.lights.clear();
    statsState.lightToIndex.clear();
    statsState.outputBaseName.clear();
    statsState.totalBeforeShadow.clear();
    statsState.totalAfterShadow.clear();
    statsState.totalContribBeforeShadow.clear();
    statsState.totalContribAfterShadow.clear();
}

}  // namespace pbrt
