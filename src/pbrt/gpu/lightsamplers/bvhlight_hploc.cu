#include <pbrt/lightsamplers/bvhlight.h>

#include <pbrt/gpu/util.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/vecmath.h>

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <limits>
#include <vector>

namespace pbrt {
static constexpr uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();
static constexpr int kHPLOCSearchRadius = 12;

struct BuildStateContainer {
    Bounds3f allLightsBounds;
    size_t nLights;
    size_t nMergedClusters;

    LightBVHBuildContainer* deviceLightsContainer;
    LightBVHNode* nodes;
    uint32_t* clusterIndices;
    uint32_t* parentIndices;
};

uint64_t* GetSortedMortonCodes(BuildStateContainer& buildState) {
    const BuildStateContainer localState = buildState;
    
    uint64_t* deviceMortonCodes = GPUAllocAsync<uint64_t>(localState.nLights);
    uint64_t* deviceMortonCodesSorted = GPUAllocAsync<uint64_t>(localState.nLights);
    uint32_t* deviceClusterIndicesSorted = GPUAllocAsync<uint32_t>(localState.nLights);

    GPUParallelFor("Assign Morton Codes", ProfilerKernelGroup::HPLOC, localState.nLights,
    [=] PBRT_GPU(int idx) {
        LightBVHBuildContainer cont = localState.deviceLightsContainer[idx];
        Point3f centroid = cont.bounds.Centroid();
        Vector3f normal = cont.bounds.w;

        localState.clusterIndices[idx] = idx;
        deviceMortonCodes[idx] = EncodeExtendedMorton5(centroid, localState.allLightsBounds, normal);
    });

    void* tempStorage = nullptr;
    size_t tempStorageBytes = 0;
    uint32_t beginBit = 1, endBit = 64;
    {
        KernelTimerWrapper timer(GetProfilerEvents("Radix Sort Morton keys", ProfilerKernelGroup::HPLOC));
        cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, 
            deviceMortonCodes, deviceMortonCodesSorted,
            localState.clusterIndices, deviceClusterIndicesSorted, localState.nLights, beginBit, endBit);

        tempStorage = GPUAllocAsync<uint8_t>(tempStorageBytes);
        cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, 
            deviceMortonCodes, deviceMortonCodesSorted,
            localState.clusterIndices, deviceClusterIndicesSorted, localState.nLights, beginBit, endBit);
    }

    GPUFreeAsync(deviceMortonCodes);
    GPUFreeAsync(tempStorage);
    GPUFreeAsync(localState.clusterIndices);

    buildState.clusterIndices = deviceClusterIndicesSorted;
    return deviceMortonCodesSorted;
}

bool BVHLightSampler::buildBVHGPU(std::vector<LightBVHBuildContainer>& bvhLights) {
    if (bvhLights.size() < 100) {
        return false;
    }

    BuildStateContainer buildState;
    buildState.nLights = bvhLights.size();
    buildState.nMergedClusters = 0;
    buildState.allLightsBounds = m_allLightBounds;

    buildState.deviceLightsContainer = GPUAllocAsync<LightBVHBuildContainer>(buildState.nLights);
    buildState.clusterIndices = GPUAllocAsync<uint32_t>(buildState.nLights);
    GPUCopyToDevice(buildState.deviceLightsContainer, bvhLights.data(), bvhLights.size());

    // std::vector<uint64_t> mortonCodesCPU(bvhLights.size());
    // for (size_t i = 0; i < bvhLights.size(); ++i) {
    //     const LightBVHBuildContainer& cont(bvhLights[i]);
    //     Point3f centroid = cont.bounds.Centroid();
    //     Vector3f normal = cont.bounds.w;
    //     mortonCodesCPU[i] = EncodeExtendedMorton5(centroid, m_allLightBounds, normal);
    // }
    // std::sort(mortonCodesCPU.begin(), mortonCodesCPU.end());

    uint64_t* deviceSortedMortonCodes = GetSortedMortonCodes(buildState);

    std::vector<uint64_t> mortonCodes(bvhLights.size());
    GPUCopyToHost(mortonCodes.data(), deviceSortedMortonCodes, bvhLights.size());    

    GPUFreeAsync(deviceSortedMortonCodes);
    GPUFreeAsync(buildState.deviceLightsContainer);
    GPUFreeAsync(buildState.clusterIndices);

    ReportKernelStats(ProfilerKernelGroup::HPLOC);

    return false;
}

}
