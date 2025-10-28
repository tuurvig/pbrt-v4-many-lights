#include "lightcuts.h"

#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>

namespace pbrt{
///////////////////////////////////////////////////////////////////////////
// LightcutsLightSampler

STAT_MEMORY_COUNTER("Memory/Lightcuts LightTree", lightCutsLightTreeBytes);

constexpr uint32_t infiniteLightsIndex = 2;
constexpr uint32_t otherLightsIndex = 3;

LightcutsTree::LightcutsTree(bool isPoint, Allocator alloc) 
    : lights(alloc), nodes(alloc), isPoint(isPoint) {}

LightcutsLightSampler::LightcutsLightSampler(pstd::span<const Light> lights, Allocator alloc, Float threshold) 
    : m_pointTree(true, alloc), m_spotTree(false, alloc), m_otherLights(alloc), m_lightToLocation(alloc), m_threshold(threshold) {
    
    // Initialize infiniteLights array and lightcuts lights
    std::vector<LightBuildContainer> pointLights, spotLights;

    for (size_t i = 0; i < lights.size(); ++i) {
        Light light = lights[i];
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        
        if (!lightBounds) {
            uint32_t index = m_infiniteLights.size();
            m_infiniteLights.push_back(light);
            m_lightToLocation.Insert(light, {infiniteLightsIndex, index});

        } else if (lightBounds->phi > 0) {
            if (light.Is<PointLight>()) {
                pointLights.emplace_back(*lightBounds, light);
                m_pointTree.allLightBounds = Union(m_pointTree.allLightBounds, lightBounds->bounds);

            } else if (light.Is<SpotLight>()) {
                spotLights.emplace_back(*lightBounds, light);
                m_spotTree.allLightBounds = Union(m_spotTree.allLightBounds, lightBounds->bounds);

            } else {
                uint32_t index = m_otherLights.size();
                m_otherLights.push_back(light);
                m_lightToLocation.Insert(light, {otherLightsIndex, index});
                m_otherLightsPower += lightBounds->phi;
            }
        }
    }

    RNG rng;
    Float u = rng.Uniform<Float>();
    if (!m_pointTree.lights.empty()) {
        buildLightTree(pointLights, m_pointTree, 0, pointLights.size(), 0, 0, u);
    }

    if (!m_spotTree.lights.empty()) {
        buildLightTree(spotLights, m_spotTree, 0, spotLights.size(), 0, 0, u);
    }

    lightCutsLightTreeBytes += (m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size() + m_infiniteLights.size()) * sizeof(Light) + 
                               (m_pointTree.nodes.size() + m_spotTree.nodes.size()) * sizeof(LightcutsTreeNode) +
                               m_lightToLocation.capacity() * (sizeof(Light) + sizeof(LightLocation));
}

LightcutsLightSampler::TreeNodeBuildSuccess LightcutsLightSampler::buildLightTree(
    std::vector<LightBuildContainer>& lightcutsLights, LightcutsTree& tree, int start, int end, uint32_t bitTrail, int depth, float& u) {
    DCHECK_LT(start, end);

    if (end - start == 1) {
        const LightBuildContainer& leaf(lightcutsLights[start]);
        CompactLightBounds cb(leaf.bounds, tree.allLightBounds);

        int nodeIndex = tree.nodes.size();
        int lightIndex = tree.lights.size();

        tree.lights.emplace_back(leaf.light);
        tree.nodes.emplace_back(LightcutsTreeNode::MakeLeaf(lightIndex, nodeIndex, cb));
        m_lightToLocation.Insert(leaf.light, {tree.isPoint, bitTrail});
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

        Float diagonalLenSqr = LengthSquared(tree.allLightBounds.Diagonal());
        for (int i = 0, max = nBuckets - 1; i < max; ++i) {
            const Float leftCost = EvaluateCost(leftBoundsSum[i], diagonalLenSqr, tree.isPoint);
            const Float rightCost = EvaluateCost(rightBoundsSum[i + 1], diagonalLenSqr, tree.isPoint);

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
    size_t nodeIndex = tree.nodes.size();
    tree.nodes.emplace_back();
    CHECK_LT(depth, 64);
    TreeNodeBuildSuccess left = buildLightTree(lightcutsLights, tree, start, mid, bitTrail, depth + 1, u);
    DCHECK_EQ(nodeIndex + 1, left.nodeIdx);
    TreeNodeBuildSuccess right = buildLightTree(lightcutsLights, tree, mid, end, bitTrail | (1u << depth), depth + 1, u);

    Float intensities[2] = {left.bounds.phi, right.bounds.phi};
    Float nodePMF;
    int child = SampleDiscrete(intensities, u, &nodePMF, &u);
    int successorIdx = (child == 0) ? left.representantIdx : right.representantIdx;

    LightBounds lb = Union(left.bounds, right.bounds);
    CompactLightBounds cb(lb, tree.allLightBounds);
    tree.nodes[nodeIndex] = LightcutsTreeNode::MakeInterior(right.nodeIdx, successorIdx, cb);
    return {lb, successorIdx, static_cast<int>(nodeIndex)};
}

pstd::optional<SampledLight> LightcutsLightSampler::SampleLightTree(const LightSampleContext& ctx, const LightcutsTree& tree, Float pmf, Float u) const {
    int nodeIndex = 0;
    Point3f p = ctx.p();
    Normal3f n = ctx.ns;
    const LightcutsTreeNode* node = &tree.nodes[nodeIndex];

    Float treeDiagonal = LengthSquared(tree.allLightBounds.Diagonal());
    while (!node->isLeaf) {
        int childrenIndices[2] = {nodeIndex + 1, node->childOrLightIndex};
        const LightcutsTreeNode *children[2] = {&tree.nodes[nodeIndex + 1],
                                                &tree.nodes[node->childOrLightIndex]};
        
        Float errBounds[2] = {ComputeErrorBounds(children[0], tree.allLightBounds, p), ComputeErrorBounds(children[1], tree.allLightBounds, p)};
        if (errBounds[0] == 0 && errBounds[1] == 0) {
            return {};
        }

        // Randomly sample a children node
        Float nodePMF;
        int child = SampleDiscrete(errBounds, u, &nodePMF, &u);
        pmf *= nodePMF;
        nodeIndex = (child == 0) ? (nodeIndex + 1) : node->childOrLightIndex;
        node = &tree.nodes[nodeIndex];

        if (errBounds[child] < m_threshold) {
            int representantLightIndex = tree.nodes[node->representantIdx].childOrLightIndex;
            return SampledLight{tree.lights[representantLightIndex], pmf};
        }
    }

    return SampledLight{tree.lights[node->childOrLightIndex], pmf};
}

pstd::optional<SampledLight> LightcutsLightSampler::SampleInfiniteLight(size_t nLights, Float &pmf, Float &u) const {
    // Compute infinite light sampling probability _pInfinite_
    Float pInfinite = Float(m_infiniteLights.size()) /
                      Float(m_infiniteLights.size() + (nLights == 0 ? 0 : 1));

    if (u < pInfinite) {
        // Sample infinite lights with uniform probability
        u /= pInfinite;
        int index =
            std::min<int>(u * m_infiniteLights.size(), m_infiniteLights.size() - 1);
        Float pmf = pInfinite / m_infiniteLights.size();
        return SampledLight{m_infiniteLights[index], pmf};
    }

    u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);
    pmf = 1 - pInfinite;

    return {};
}

std::string LightcutsLightSampler::ToString() const {
    return StringPrintf("[ LightcutsLightSampler point tree nodes: %s spot tree nodes: %s ]", m_pointTree.nodes, m_spotTree.nodes);
}

std::string LightcutsTreeNode::ToString() const {
    return StringPrintf(
        "[ LightcutsLightSampler lightBounds: %s childOrLightIndex: %d isLeaf: %d ]", compactLightBounds, childOrLightIndex, isLeaf);
}

}
