// pbrt/util/lighttree_generic.h

#ifndef PBRT_UTIL_LIGHTTREE_GENERIC_H
#define PBRT_UTIL_LIGHTTREE_GENERIC_H

#include <pbrt/pbrt.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/vecmath.h>

#include <vector>
#include <algorithm>

namespace pbrt {

template <typename InputTypeT, typename ResultTypeT>
struct NodeEmitterInterface {
    using ResultType = ResultTypeT;
    using InputType = InputTypeT;
    
    virtual int ReserveInterior() = 0;

    virtual ResultType EmitLeaf(const InputType& item, uint32_t bitTrail) = 0;

    // assumes the reservationIndex is consistent with the current vector state.
    // Since BuildLightTree is recursive and single-threaded the node at reservationIndex should be valid and waiting.
    // We reserve before recursion and finalize after both recursions are finished.
    // This ordering ensures the index of left child node is always at the increment index of current node.
    virtual ResultType FinalizeInterior(int reservationIndex, const ResultType& left, const ResultType& right, Float& u) = 0;
};

struct BuildContainerInterface {
    BuildContainerInterface(const LightBounds& bounds) : bounds(bounds) {}
    LightBounds bounds;
};

// Generic Light Tree Builder
// NBuckets: Number of buckets for SAH/Cost evaluation
// BuildContainer: Type of the object holding light info during build
// CostEvaluator: Functor to calculate split cost
// NodeEmitter: Class handling the actual creation of Leaf/Interior nodes and return types
template <int NBuckets, typename BuildContainer, typename CostEvaluator, typename NodeEmitter>
typename NodeEmitter::ResultType BuildLightTree(std::vector<BuildContainer>& items,
                                                int start, int end,
                                                uint32_t bitTrail,
                                                int depth,
                                                const CostEvaluator& costEval,
                                                NodeEmitter& emitter,
                                                Float& u) {
    DCHECK_LT(start, end);

    // 1. Base Case: Emit Leaf
    if (end - start == 1) {
        return emitter.EmitLeaf(items[start], bitTrail);
    }

    // 2. Compute bounds for split heuristics
    Bounds3f parentBounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        centroidBounds = Union(centroidBounds, items[i].bounds.Centroid());
        parentBounds = Union(parentBounds, items[i].bounds.bounds);
    }

    // 3. Find Best Split
    Float minCost = Infinity;
    int minCostSplitBucket = -1;
    int minCostSplitDim = -1;

    for (int dim = 0; dim < 3; ++dim) {
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim])
            continue;

        LightBounds bucketLightBounds[NBuckets];
        for (int i = start; i < end; ++i) {
            Point3f pc = items[i].bounds.Centroid();
            int b = NBuckets * centroidBounds.Offset(pc)[dim];
            if (b == NBuckets) b = NBuckets - 1;
            bucketLightBounds[b] = Union(bucketLightBounds[b], items[i].bounds);
        }

        LightBounds leftBoundsSum[NBuckets];
        LightBounds rightBoundsSum[NBuckets];
        
        // Simultaneous forward and backward scan
        leftBoundsSum[0] = bucketLightBounds[0];
        rightBoundsSum[NBuckets - 1] = bucketLightBounds[NBuckets - 1];
        for (int lower = 1, upper = NBuckets - 2; lower < NBuckets; ++lower, --upper) {
            leftBoundsSum[lower] = Union(bucketLightBounds[lower], leftBoundsSum[lower - 1]);
            rightBoundsSum[upper] = Union(bucketLightBounds[upper], rightBoundsSum[upper + 1]);
        }

        // Evaluate cost
        Vector3f boundsDiagonal = parentBounds.Diagonal();
        Float Kr = MaxComponentValue(boundsDiagonal) / boundsDiagonal[dim]; 
        for (int i = 0; i < NBuckets - 1; ++i) {
            Float leftCost = costEval(leftBoundsSum[i]);
            Float rightCost = costEval(rightBoundsSum[i]);
            Float cost = Kr * (leftCost + rightCost);
            if (cost > 0 && cost < minCost) {
                minCost = cost;
                minCostSplitBucket = i;
                minCostSplitDim = dim;
            }
        }
    }

    // 4. Partition
    int mid;
    if (minCostSplitDim == -1) {
        mid = (start + end) / 2;
    } else {
        const auto* pmid = std::partition(
            &items[start], &items[end - 1] + 1,
            [=](const BuildContainer& item) {
                int b = NBuckets * centroidBounds.Offset(item.bounds.Centroid())[minCostSplitDim];
                if (b == NBuckets) b = NBuckets - 1;
                return b <= minCostSplitBucket;
            });
        mid = pmid - &items[0];
        if (mid == start || mid == end)
            mid = (start + end) / 2;
    }

    // Reserve space for interior node
    auto reservation = emitter.ReserveInterior();

    // 5. Recursion
    auto leftRes  = BuildLightTree<NBuckets, BuildContainer, CostEvaluator, NodeEmitter>(items, start, mid, bitTrail, depth + 1, costEval, emitter, u);
    auto rightRes = BuildLightTree<NBuckets, BuildContainer, CostEvaluator, NodeEmitter>(items, mid, end, bitTrail | (1u << depth), depth + 1, costEval, emitter, u);

    // 6. Finalize Interior
    return emitter.FinalizeInterior(reservation, leftRes, rightRes, u);
}
}

#endif //PBRT_UTIL_LIGHTTREE_GENERIC_H