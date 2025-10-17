#include "bvhlight.h"

namespace pbrt{
///////////////////////////////////////////////////////////////////////////
// BVHLightSampler

STAT_MEMORY_COUNTER("Memory/Light BVH", lightBVHBytes);
STAT_INT_DISTRIBUTION("Integrator/Lights sampled per lookup", nLightsSampled);

// BVHLightSampler Method Definitions
BVHLightSampler::BVHLightSampler(pstd::span<const Light> lights, Allocator alloc)
    : m_lights(lights.begin(), lights.end(), alloc),
      m_infiniteLights(alloc),
      m_nodes(alloc),
      m_lightToBitTrail(alloc) {
    // Initialize _infiniteLights_ array and light BVH
    std::vector<std::pair<int, LightBounds>> bvhLights;
    for (size_t i = 0; i < lights.size(); ++i) {
        // Store $i$th light in either _infiniteLights_ or _bvhLights_
        Light light = lights[i];
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        if (!lightBounds)
            m_infiniteLights.push_back(light);
        else if (lightBounds->phi > 0) {
            bvhLights.push_back(std::make_pair(i, *lightBounds));
            m_allLightBounds = Union(m_allLightBounds, lightBounds->bounds);
        }
    }
    if (!bvhLights.empty())
        buildBVH(bvhLights, 0, bvhLights.size(), 0, 0);
    lightBVHBytes += m_nodes.size() * sizeof(LightBVHNode) +
                     m_lightToBitTrail.capacity() * sizeof(uint32_t) +
                     lights.size() * sizeof(Light) +
                     m_infiniteLights.size() * sizeof(Light);
}

std::pair<int, LightBounds> BVHLightSampler::buildBVH(
    std::vector<std::pair<int, LightBounds>> &bvhLights, int start, int end,
    uint32_t bitTrail, int depth) {
    DCHECK_LT(start, end);
    // Initialize leaf node if only a single light remains
    if (end - start == 1) {
        int nodeIndex = m_nodes.size();
        CompactLightBounds cb(bvhLights[start].second, m_allLightBounds);
        int lightIndex = bvhLights[start].first;
        m_nodes.push_back(LightBVHNode::MakeLeaf(lightIndex, cb));
        m_lightToBitTrail.Insert(m_lights[lightIndex], bitTrail);
        return {nodeIndex, bvhLights[start].second};
    }

    // Choose split dimension and position using modified SAH
    // Compute bounds and centroid bounds for lights
    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        const LightBounds &lb = bvhLights[i].second;
        bounds = Union(bounds, lb.bounds);
        centroidBounds = Union(centroidBounds, lb.Centroid());
    }

    Float minCost = Infinity;
    int minCostSplitBucket = -1, minCostSplitDim = -1;
    constexpr int nBuckets = 12;
    for (int dim = 0; dim < 3; ++dim) {
        // Compute minimum cost bucket for splitting along dimension _dim_
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim])
            continue;
        // Compute _LightBounds_ for each bucket
        LightBounds bucketLightBounds[nBuckets];
        for (int i = start; i < end; ++i) {
            Point3f pc = bvhLights[i].second.Centroid();
            int b = nBuckets * centroidBounds.Offset(pc)[dim];
            if (b == nBuckets)
                b = nBuckets - 1;
            DCHECK_GE(b, 0);
            DCHECK_LT(b, nBuckets);
            bucketLightBounds[b] = Union(bucketLightBounds[b], bvhLights[i].second);
        }

        // Compute costs for splitting lights after each bucket
        Float cost[nBuckets - 1];
        for (int i = 0; i < nBuckets - 1; ++i) {
            // Find _LightBounds_ for lights below and above bucket split
            LightBounds b0, b1;
            for (int j = 0; j <= i; ++j)
                b0 = Union(b0, bucketLightBounds[j]);
            for (int j = i + 1; j < nBuckets; ++j)
                b1 = Union(b1, bucketLightBounds[j]);

            // Compute final light split cost for bucket
            cost[i] = EvaluateCost(b0, bounds, dim) + EvaluateCost(b1, bounds, dim);
        }

        // Find light split that minimizes SAH metric
        for (int i = 1; i < nBuckets - 1; ++i) {
            if (cost[i] > 0 && cost[i] < minCost) {
                minCost = cost[i];
                minCostSplitBucket = i;
                minCostSplitDim = dim;
            }
        }
    }

    // Partition lights according to chosen split
    int mid;
    if (minCostSplitDim == -1)
        mid = (start + end) / 2;
    else {
        const auto *pmid = std::partition(
            &bvhLights[start], &bvhLights[end - 1] + 1,
            [=](const std::pair<int, LightBounds> &l) {
                int b = nBuckets *
                        centroidBounds.Offset(l.second.Centroid())[minCostSplitDim];
                if (b == nBuckets)
                    b = nBuckets - 1;
                DCHECK_GE(b, 0);
                DCHECK_LT(b, nBuckets);
                return b <= minCostSplitBucket;
            });
        mid = pmid - &bvhLights[0];
        if (mid == start || mid == end)
            mid = (start + end) / 2;
        DCHECK(mid > start && mid < end);
    }

    // Allocate interior _LightBVHNode_ and recursively initialize children
    int nodeIndex = m_nodes.size();
    m_nodes.push_back(LightBVHNode());
    CHECK_LT(depth, 64);
    std::pair<int, LightBounds> child0 =
        buildBVH(bvhLights, start, mid, bitTrail, depth + 1);
    DCHECK_EQ(nodeIndex + 1, child0.first);
    std::pair<int, LightBounds> child1 =
        buildBVH(bvhLights, mid, end, bitTrail | (1u << depth), depth + 1);

    // Initialize interior node and return node index and bounds
    LightBounds lb = Union(child0.second, child1.second);
    CompactLightBounds cb(lb, m_allLightBounds);
    m_nodes[nodeIndex] = LightBVHNode::MakeInterior(child1.first, cb);
    return {nodeIndex, lb};
}

std::string BVHLightSampler::ToString() const {
    return StringPrintf("[ BVHLightSampler nodes: %s ]", m_nodes);
}

std::string LightBVHNode::ToString() const {
    return StringPrintf(
        "[ LightBVHNode lightBounds: %s childOrLightIndex: %d isLeaf: %d ]", lightBounds,
        childOrLightIndex, isLeaf);
}

}