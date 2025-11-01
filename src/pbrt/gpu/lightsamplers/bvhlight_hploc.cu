#include <pbrt/lightsamplers/bvhlight.h>

#include <pbrt/gpu/util.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace pbrt {
static constexpr uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();
static constexpr int kHPLOCSearchRadius = 12;

bool BVHLightSampler::buildBVHGPU(std::vector<std::pair<int, LightBounds>>& bvhLights) {
    int nLights = static_cast<int>(bvhLights.size());
    if (nLights < 100) {
        return false;
    }

    

    return false;
}

}