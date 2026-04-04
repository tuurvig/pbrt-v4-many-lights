// lighttree_generic is Copyright(c) 2026 Richard Kvasnica.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_LIGHTTREE_GENERIC_H
#define PBRT_UTIL_LIGHTTREE_GENERIC_H

#include <pbrt/pbrt.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/vecmath.h>

#include <vector>
#include <algorithm>

namespace pbrt {

/// @brief Interface for objects holding bounding information druing tree construction.
/// Provides a unified way to access spatial and directional bounds regardless of the specific bounding volume type.
/// @tparam LightBoundsTypeT The underlying type used to represent bounds (e.g., AABB, Spherical Bounds).
template<typename LightBoundsTypeT>
struct BuildContainerInterface {
    using LightBoundsType = LightBoundsTypeT;
    BuildContainerInterface() = default;

    PBRT_CPU_GPU
    BuildContainerInterface(const LightBoundsTypeT& bounds) : bounds(bounds) {}
    
    LightBoundsTypeT bounds; ///< The spatial/directional bounds of the associated node or cluster
};

/// @brief Abstract interface defining the lifecycle of emitting tree ndoes into a flat memory layout.
/// The construction follows the reservation phase and then the finalization of interior nodes on single threaded,
/// recursive tree construction where child nodes must be processed before their parent.
/// @tparam BuildContainerTypeT The type of the container holding input data (leaf bounds).
/// @tparam ResultTypeT The type returned after emitting a node, usually containing its final bounds and index.
template <typename BuildContainerTypeT, typename ResultTypeT>
struct NodeEmitterInterface {
    using ResultType = ResultTypeT;
    using BuildContainerType = BuildContainerTypeT;
    
    /// @brief Allocates an empty slot for an interior node in the underlying storage.
    /// @return The index of the reserved slot.
    virtual int ReserveInterior() = 0;

    /// @brief Constructs and stores a leaf node.
    /// @param item The Bounds and data associated with the leaf.
    /// @param bitTrail A bitmask tracking the traversal path to this leaf. (Used in hierarchic sampling).
    /// @return The resulting bounds and storage index of the emitted leaf.
    virtual ResultType EmitLeaf(const BuildContainerType& item, uint32_t bitTrail) = 0;

    /// @brief Populates a previously reserved interior node after its children have been processed.
    /// Assumes a single-threaded execution where `reservationIndex` correctly corresponds to the current node.
    /// By reserving before recursion, the left child's index is placed immediately after the parent.
    /// @param reservationIndex The storage index previously obtained via ReserveInterior().
    /// @param left The build result of the left child sub-tree
    /// @param right The build result of the right child sub-tree
    /// @return The resulting merged bounds and storage index of the finalized interior node.
    virtual ResultType FinalizeInterior(int reservationIndex, const ResultType& left, const ResultType& right) = 0;
};

/// @brief An adapter interface for traversing and converting tree nodes between different representations.
/// Primarily used for adapting GPU-constructed nodes for algorithm-specific tree nodes.
/// @tparam InputTypeT The source node format.
/// @tparam OutputTypeT The target node format, the source node should be converted into.
template <typename InputTypeT, typename OutputTypeT>
struct TreeLeafAdapterInterface {
using InputType = InputTypeT;
using OutputType = OutputTypeT;

    virtual const InputType& At(uint32_t idx) const = 0;

    uint32_t Left(uint32_t idx) const {return Left(At(idx));}
    uint32_t Right(uint32_t idx) const {return Right(At(idx));}
    bool IsLeaf(uint32_t idx) const {return IsLeaf(At(idx));}

    virtual uint32_t Left(const InputType& node) const = 0;
    virtual uint32_t Right(const InputType& node) const = 0;
    virtual bool IsLeaf(const InputType& node) const = 0;

    /// @brief Converts the generic input node into the algorithm-specific output format.
    /// @param node The input node to convert.
    /// @return The newly constructed output node.
    virtual OutputTypeT Convert(const InputType& node) const = 0;
};

/// @brief Recursively build a generic light tree using a binned heuristic (SAH/SAOH).
/// This function employs a top-down recursive approach. It evaluates potential split
/// planes by discretizing the spatial bounds into a fixed number of buckets
/// and calculating the cost of splitting at each bucket boundary.
/// 
/// @tparam NBuckets Number of spatial bins used to approximate the continuous split plane search
/// @tparam BuildContainer The data structure holding the light bounds during construction
/// @tparam CostEvaluator Functor type computing the heuristic cost of bounding volume.
/// @tparam NodeEmitter Type handling the allocation and initialization of tree nodes.
/// 
/// @param items Vector of light containers to be spatially partitioned.
/// @param start Starting index of the current partition range.
/// @param end Ending index (exclusive) of the current partition range.
/// @param bitTrail A bitmask tracking the binary path from the root to the current node.
/// @param depth Current recursion depth in the tree.
/// @param costEval Instance of the cost evaluator functor.
/// @param emitter Instance of the node emitter responsible for storing the constructed nodes.
/// @return The result type defined by the NodeEmitter
template <int NBuckets, typename BuildContainer, typename CostEvaluator, typename NodeEmitter>
typename NodeEmitter::ResultType BuildLightTree(std::vector<BuildContainer>& items,
                                                int start, int end,
                                                uint32_t bitTrail,
                                                int depth,
                                                const CostEvaluator& costEval,
                                                NodeEmitter& emitter) {
    DCHECK_LT(start, end);

    // 1. Base Case: Emit Leaf
    if (end - start == 1) {
        return emitter.EmitLeaf(items[start], bitTrail);
    }

    // 2. Compute the bounding box of cluster centroids (for split planes)
    // and the overall spatial bounds of all items in the current range
    Bounds3f centroidBounds;
    typename BuildContainer::LightBoundsType::BoundsType parentBounds;
    for (int i = start; i < end; ++i) {
        centroidBounds = Union(centroidBounds, items[i].bounds.Centroid());
        parentBounds = Union(parentBounds, items[i].bounds.Bounds());
    }

    // 3. Find Best Split: Evaluate potatial split planes across all 3 dimensions
    // using the binned heuristic approach to find the minimum cost split.
    Float minCost = Infinity;
    int minCostSplitBucket = -1;
    int minCostSplitDim = -1;

    for (int dim = 0; dim < 3; ++dim) {
        // Skip dimensions where all centroids lie on the exact same plane.
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim])
            continue;

        // Initialize buckets and accumulate the bounds of all items falling into each bucket.
        typename BuildContainer::LightBoundsType bucketLightBounds[NBuckets];
        for (int i = start; i < end; ++i) {
            Point3f pc = items[i].bounds.Centroid();
            int b = NBuckets * centroidBounds.Offset(pc)[dim];
            if (b == NBuckets) b = NBuckets - 1;
            bucketLightBounds[b] = Union(bucketLightBounds[b], items[i].bounds);
        }

        // Arrays to hold the cumulative bounds from the left and right sides of any split.
        typename BuildContainer::LightBoundsType leftBoundsSum[NBuckets];
        typename BuildContainer::LightBoundsType rightBoundsSum[NBuckets];
        
        // Simultaneous forward and backward scan to compute prefix and suffix sums of bounds.
        // This allows us to evaluate the split cost at any bucket boundary in O(1) time.
        leftBoundsSum[0] = bucketLightBounds[0];
        rightBoundsSum[NBuckets - 1] = bucketLightBounds[NBuckets - 1];
        for (int lower = 1, upper = NBuckets - 2; lower < NBuckets; ++lower, --upper) {
            leftBoundsSum[lower] = Union(bucketLightBounds[lower], leftBoundsSum[lower - 1]);
            rightBoundsSum[upper] = Union(bucketLightBounds[upper], rightBoundsSum[upper + 1]);
        }

        // Evaluate cost for each bucket boundary.
        // Kr is a spatial regulatization factor that penalizes splitting along shorter axes,
        // encouraging a more geometrically balanced tree.
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

    // 4. Partition the items array based on the best discored split.
    int mid;
    if (minCostSplitDim == -1) {
        // Fallback: If no valid split was found, force a median split.
        mid = (start + end) / 2;
    } else {
        // Reorder the items so that elements going to the left child precede those going right.
        const auto* pmid = std::partition(
            &items[start], &items[end - 1] + 1,
            [=](const BuildContainer& item) {
                int b = NBuckets * centroidBounds.Offset(item.bounds.Centroid())[minCostSplitDim];
                if (b == NBuckets) b = NBuckets - 1;
                return b <= minCostSplitBucket;
            });
        mid = pmid - &items[0];

        // Edge case fallback: If the heuristic placed all items on one side, force a median split
        // to guarantee tree depth progression and prevent infinite recursion.
        if (mid == start || mid == end)
            mid = (start + end) / 2;
    }

    // 5. Reserve space for interior node. This maintains the flat tree layout.
    auto reservation = emitter.ReserveInterior();

    // 6. Recursion: Build left and right subtrees.
    // The `bitTrail` is updated by setting the bit on the position of `depth` to 1 for the right child branch.
    auto leftRes  = BuildLightTree<NBuckets, BuildContainer, CostEvaluator, NodeEmitter>(items, start, mid, bitTrail, depth + 1, costEval, emitter);
    auto rightRes = BuildLightTree<NBuckets, BuildContainer, CostEvaluator, NodeEmitter>(items, mid, end, bitTrail | (1u << depth), depth + 1, costEval, emitter);

    // 6. Finalize Interior: Merge the results and populate the reserved node.
    return emitter.FinalizeInterior(reservation, leftRes, rightRes);
}

/// @brief Recursively flattens an intermediate light tree representation into a linear memory layout.
/// 
/// @tparam TreeNodesAdapter Adapter Type used to abstract the reading of the input tree structure.
/// @tparam NodeEmitter Builder type responsible for writing formatted nodes into the final array.
/// 
/// @param nodes The adapter instance providing unified access to the input tree nodes.
/// @param nodeIdx The index of the current node being processed in the input tree.
/// @param bitTrail A bitmask recording the traversal path from the root.
/// @param depth The current recursion depth, used to place the correct bit into the bitTrail.
/// @param emitter The emitter instance managing the destination storage and final node formatting.
/// @return The result of the emitted node.
template <typename TreeNodesAdapter, typename NodeEmitter>
typename NodeEmitter::ResultType FlattenLightTree(const TreeNodesAdapter& nodes,
                                                  uint32_t nodeIdx,
                                                  uint32_t bitTrail,
                                                  int depth,
                                                  NodeEmitter& emitter) {
    // 1. Fetch abstract node via the adapter.
    const auto& node(nodes.At(nodeIdx));

    // 2. If the node is leaf, convert it to the target format and emit it.
    if (nodes.IsLeaf(node)) {
        return emitter.EmitLeaf(nodes.Convert(node), bitTrail);
    }

    // 3. Reserve space for the interior node.
    auto reservation = emitter.ReserveInterior();

    // 4. Recursively process children.
    auto leftRes = FlattenLightTree(nodes, nodes.Left(node), bitTrail, depth + 1, emitter);
    auto rightRes = FlattenLightTree(nodes, nodes.Right(node), bitTrail | (1u << depth), depth + 1, emitter);

    // 5. Finalize the interior node by merging the bounds of its processed children.  
    return emitter.FinalizeInterior(reservation, leftRes, rightRes);
}

}

#endif //PBRT_UTIL_LIGHTTREE_GENERIC_H