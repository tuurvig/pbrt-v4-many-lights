#include "lightcuts.h"

#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>

namespace pbrt{
///////////////////////////////////////////////////////////////////////////
// LightcutsLightSampler

STAT_MEMORY_COUNTER("Memory/Lightcuts LightTree", lightCutsLightTreeBytes);

LightcutsLightSampler::LightcutsLightSampler(pstd::span<const Light> lights, Allocator alloc) 
    : m_lights(alloc), m_infiniteLights(alloc), m_nodes(alloc) {

    // Initialize infiniteLights array and lightcuts lights
    std::vector<LightBuildContainer> lightcutsLights;
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.5f);
    for (size_t i = 0; i < lights.size(); ++i) {
        Light light = lights[i];
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        
        if (!lightBounds) {
            m_infiniteLights.push_back(light);
        } else if (lightBounds->phi > 0) {
            // SampledSpectrum phi = SafeDiv(light.Phi(lambda), lambda.PDF());
            // Float avgPower = phi.Average();
            lightcutsLights.emplace_back(*lightBounds, light);
            m_allLightBounds = Union(m_allLightBounds, lightBounds->bounds);
        }
    }

    if (!lightcutsLights.empty()) {
        float u = 0.5;
        buildLightTree(lightcutsLights, 0, lightcutsLights.size(), 0, u);
    }

    lightCutsLightTreeBytes += m_lights.size() * sizeof(Light);
}

LightcutsLightSampler::TreeNodeBuildSuccess LightcutsLightSampler::buildLightTree(
    std::vector<LightBuildContainer>& lightcutsLights, int start, int end, int depth, float& u) {
    DCHECK_LT(start, end);

    if (end - start == 1) {
        const LightBuildContainer& leaf(lightcutsLights[start]);
        CompactLightBounds cb(leaf.bounds, m_allLightBounds);

        int nodeIndex = m_nodes.size();
        int lightIndex = m_lights.size();

        m_lights.emplace_back(leaf.light);
        m_nodes.emplace_back(LightcutsTreeNode::MakeLeaf(lightIndex, nodeIndex, cb));
        
        return {leaf.bounds, nodeIndex, nodeIndex};
    }

    // Choose split dimension and position using Similarity Metric
    // Compute bounds and centroid bounds for lights
    Bounds3f centroidBounds;
    for (int i = start; i < end; ++i) {
        const LightBounds& lb(lightcutsLights[i].bounds);
        centroidBounds = Union(centroidBounds, lb.Centroid());
    }

    Float minCost = Infinity;
    int minCostSplitBucket = -1;
    int minCostSplitDim = -1;

    constexpr int nBuckets = 16;
    for (int dim = 0; dim < 3; ++dim) {
        // Compute minimum cost bucket for splitting along dim
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]){
            continue;
        }

        LightBounds bucketLightBounds[nBuckets];
        for (int i = start; i < end; ++i) {
            Point3f pc = lightcutsLights[i].bounds.Centroid();
            int b = nBuckets * centroidBounds.Offset(pc)[dim];
            if (b == nBuckets){
                b = nBuckets - 1;
            }
            DCHECK_GE(b, 0);
            DCHECK_LT(b, nBuckets);
            bucketLightBounds[b] = Union(bucketLightBounds[b], lightcutsLights[i].bounds);
        }

        LightBounds leftBoundsSum[nBuckets], rightBoundsSum[nBuckets];
        leftBoundsSum[0] = bucketLightBounds[0];
        rightBoundsSum[nBuckets - 1] = bucketLightBounds[nBuckets - 1];

        for (int lower = 1, upper = nBuckets - 2; lower < nBuckets; ++lower, --upper) {
            LightBounds& leftBoundsPrefix(leftBoundsSum[lower]);
            LightBounds& rightBoundsPrefix(rightBoundsSum[upper]);

            const LightBounds& prevLeftBoundsPrefix(leftBoundsSum[lower - 1]);
            const LightBounds& prevRightBoundsPrefix(rightBoundsSum[upper + 1]);

            leftBoundsPrefix = Union(leftBoundsPrefix, prevLeftBoundsPrefix);
            rightBoundsPrefix = Union(rightBoundsPrefix, prevRightBoundsPrefix);
        }

        for (int i = 0, max = nBuckets - 1; i < max; ++i) {
            const Float leftCost = EvaluateCost(leftBoundsSum[i], true);
            const Float rightCost = EvaluateCost(rightBoundsSum[i + 1], true);

            const Float cost = rightCost + leftCost;

            if (cost > 0 && cost < minCost) {
                minCost = cost;
                minCostSplitBucket = i;
                minCostSplitDim = dim;
            }
        }
    }

    // Partition lights according to chosen split
    int mid;
    if (minCostSplitDim == -1) {
        mid = (start + end) / 2;
    } else {
        const auto* pmid = std::partition(&lightcutsLights[start], &lightcutsLights[end - 1] + 1,
            [=](const LightBuildContainer& container) {
                int b = nBuckets * centroidBounds.Offset(container.bounds.Centroid())[minCostSplitDim];
                if (b == nBuckets) {
                    b = nBuckets - 1;
                }
                DCHECK_GE(b, 0);
                DCHECK_LT(b, nBuckets);
                return b <= minCostSplitBucket;
            });
        mid = pmid - &lightcutsLights[0];
        if (mid == start || mid == end) {
            mid = (start + end) / 2;
        }
        DCHECK(mid > start && mid < end);
    }

    // Allocate interior and recursively initialize children
    size_t nodeIndex = m_nodes.size();
    m_nodes.emplace_back();
    CHECK_LT(depth, 64);
    TreeNodeBuildSuccess left = buildLightTree(lightcutsLights, start, mid, depth + 1, u);
    DCHECK_EQ(nodeIndex + 1, left.nodeIdx);
    TreeNodeBuildSuccess right = buildLightTree(lightcutsLights, mid, end, depth + 1, u);

    Float intensities[2] = {left.bounds.phi, right.bounds.phi};
    Float nodePMF;
    int child = SampleDiscrete(intensities, u, &nodePMF, &u);
    int successorIdx = (child == 0) ? left.representantIdx : right.representantIdx;
    
    // if (left.bounds.phi == 0)
    //     successorIdx = right.representantIdx;
    // else if (right.bounds.phi == 0)
    //     successorIdx = left.representantIdx;
    // else {
    //     Float totalPower = left.bounds.phi + right.bounds.phi;
    //     Float probLeft = left.bounds.phi / totalPower;
    //     if (u < probLeft) {
    //         successorIdx = left.representantIdx;
    //         u /= probLeft;
    //     } else {
    //         successorIdx = right.representantIdx;
    //         u = (u - probLeft) / (1 - probLeft);
    //     }
    // }

    LightBounds lb = Union(left.bounds, right.bounds);
    CompactLightBounds cb(lb, m_allLightBounds);
    m_nodes[nodeIndex] = LightcutsTreeNode::MakeInterior(right.nodeIdx, successorIdx, cb);
    return {lb, successorIdx, static_cast<int>(nodeIndex)};
}

std::string LightcutsLightSampler::ToString() const {
    return StringPrintf("[ LightcutsLightSampler nodes: %s ]", m_nodes);
}

std::string LightcutsTreeNode::ToString() const {
    return StringPrintf(
        "[ LightcutsLightSampler lightBounds: %s childOrLightIndex: %d isLeaf: %d ]", compactLightBounds, childOrLightIndex, isLeaf);
}

}
