// Copyright(c) 2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include "ltc.h"

#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>

#include <pbrt/util/hash.h>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/lighttreebuilder.h>
#include <pbrt/gpu/util.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>

#include <array>

#endif //PBRT_BUILD_GPU_RENDERER

#include <pbrt/util/shadingpoints.h>
#include <algorithm>
#include <cmath>

namespace pbrt {

///////////////////////////////////////////////////////////////////////////
// Learning To Cluster LightSampler

STAT_MEMORY_COUNTER("Memory/Learning To Cluster Light Tree", LTCLightTreeBytes);
STAT_MEMORY_COUNTER("Memory/Learning To Cluster Scene Partition", LTCScenePartitionBytes);

LTCLightSampler::LTCLightSampler(pstd::span<const Light> lights, Allocator alloc, Float beta, Float omega, Float gamma) :
    m_tree(alloc), m_partitions(alloc), m_infiniteLights(alloc), m_lightToBitTrail(alloc), m_beta(beta), m_omega(omega), m_gamma(gamma) {
    std::vector<LightBVHBuildContainer> treeLights;
    for (size_t i = 0; i < lights.size(); ++i) {
        // Store $i$th light in either _infiniteLights_ or _treeLights_
        Light light = lights[i];
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        if (!lightBounds) {
            m_infiniteLights.push_back(light);
        }
        else if (lightBounds->phi > 0) {
            const auto lightIdx = static_cast<uint32_t>(m_tree.lights.size());
            m_tree.lights.emplace_back(light);
            treeLights.emplace_back(*lightBounds, lightIdx);
            m_tree.allLightBounds = Union(m_tree.allLightBounds, lightBounds->bounds);
        }
    }

    if (!treeLights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        bool buildOnGPU = buildLightTreeGPU(treeLights);
        if (!buildOnGPU)
#endif
        {
            LTCNodeEmitter emitter(m_tree, m_lightToBitTrail);
            BuildLightTree<16, LightBVHBuildContainer, SAOHCostEvaluator, LTCNodeEmitter>(treeLights, 0, treeLights.size(), 0, 0, SAOHCostEvaluator(), emitter);
        }
    }

    LTCLightTreeBytes += (m_tree.lights.size() + m_infiniteLights.size()) * sizeof(Light) + 
                          m_tree.nodes.size() * sizeof(LightcutsTreeNode) +
                          m_lightToBitTrail.capacity() * (sizeof(Light) + sizeof(uint32_t));
}

#ifdef PBRT_BUILD_GPU_RENDERER
class LTCTreeBuilderGPU final : public LightTreeBuilderGPU<LightBounds, uint64_t, SAOHCostEvaluator> {
  public:
    explicit LTCTreeBuilderGPU(const Bounds3f &bounds) : m_allLightBounds(bounds) {}

    bool Build(std::vector<LightBVHBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);

        LightTreeBuildState<LightBounds> buildState(State());
        std::array<uint8_t, 3> ax = DetermineAxisOrder(buildState.allLightBounds);

        LightBVHBuildContainer* dLightsContainer = GPUAllocAsync<LightBVHBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        uint64_t* dMortonCodes = MortonCodes();
        MortonCodes() = SortNodesMorton(State(), MortonCodes(), [buildState, ax, dLightsContainer, dMortonCodes] PBRT_GPU(int idx) {
            LightBVHBuildContainer cont = dLightsContainer[idx];
            LightTreeConstructionNodeGPU<LightBounds> leaf(cont.bounds, kInvalidIndex, idx);
            Point3f centroid = cont.bounds.Centroid();
            Vector3f offset = buildState.allLightBounds.Offset(centroid);

            Point3f position = {offset[ax[0]], offset[ax[1]], offset[ax[2]]};
            Vector3f direction = Normalize(cont.bounds.w);

            dMortonCodes[idx] = EncodeExtendedMorton5(position, direction);
            buildState.dClusterIndices[idx] = idx;
            buildState.dNodes[idx] = leaf;
        });
        
        GPUFreeAsync(dLightsContainer);
        dLightsContainer = nullptr;

        BuildNodes(SAOHCostEvaluator());
        
        return true;
    }

    void FlattenTree(LTCLightTree& tree, std::vector<LightBVHBuildContainer> &lights, HashMap<Light, uint32_t>& bitTrailContainer) {
        const LightTreeBuildState<LightBounds> &state(State());
        if (state.nLights == 0)
            return;

        uint32_t nNodes = 0;
        uint32_t rootIndex = 0;
        GPUCopyToHost(&nNodes, state.nMergedClusters, 1);
        GPUCopyToHost(&rootIndex, state.dClusterIndices, 1);
        std::vector<LightTreeConstructionNodeGPU<LightBounds>> hostNodes(nNodes);
        GPUCopyToHost(hostNodes.data(), state.dNodes, nNodes);

        tree.nodes.reserve(nNodes);

        LTCNodeEmitter emitter(tree, bitTrailContainer);
        GPUToLightBVHLeaf adapter(hostNodes);
        
        FlattenLightTree<GPUToLightBVHLeaf, LTCNodeEmitter>(adapter, rootIndex, 0, 0, emitter);
    }

private:
    Bounds3f m_allLightBounds;
};

bool LTCLightSampler::buildLightTreeGPU(std::vector<LightBVHBuildContainer> &lights) {
    if (lights.size() < 100 || !Options->useGPU)
        return false;

    LTCTreeBuilderGPU builder(m_tree.allLightBounds);
    if (!builder.Build(lights))
        return false;

    builder.FlattenTree(m_tree, lights, m_lightToBitTrail);
    return true;
}
#endif

PartitionTree::PartitionTree(Allocator alloc) :
    alloc(alloc), leaves(alloc), innerNodes(alloc), representantPoints(alloc) {}

PartitionTree::~PartitionTree() {
    for (OnlineLightTreeCut *leaf : leaves)
        alloc.delete_object(leaf);
}

void PartitionTree::EmplaceLeaf() {
    leaves.emplace_back(alloc.new_object<OnlineLightTreeCut>());
}

uint32_t LTCLightSampler::BuildPartitionTree(pstd::span<ShadingPoint>& items, int start, int end) {
    DCHECK_LT(start, end);
    
    int splitAxis = -1;
    int splitIndex = -1;
    Float splitValue = std::numeric_limits<Float>::max();
    {
        const int partitionSize = end - start;
        auto emitLeaf = [&]() -> uint32_t {
            uint32_t leafIndex = static_cast<uint32_t>(m_partitions.leaves.size());
            uint32_t nodeIndex = static_cast<uint32_t>(m_partitions.innerNodes.size());
            m_partitions.innerNodes.emplace_back(PartitionTreeNode::MakeLeaf(leafIndex));
            m_partitions.EmplaceLeaf();

            Point3f avgPosition(0, 0, 0);
            DirectionCone cone;
            for (int i = start; i < end; ++i) {
                avgPosition += items[i].point;

                Vector3f normal(items[i].dir);
                cone = Union(cone, DirectionCone(normal));
            }
            
            avgPosition /= partitionSize;

            m_partitions.representantPoints.emplace_back(avgPosition, Normal3f(cone.w));

            return nodeIndex;
        };

        // 1. Base Case: Leaf node
        if (partitionSize <= 150) {
            return emitLeaf();
        }

        // 2. Find tight bounds for the current node
        Bounds3f spaceBounds;
        Bounds2i directionBounds;
        for (int i = start; i < end; ++i) {
            const ShadingPoint& item(items[i]);
            spaceBounds = Union(spaceBounds, item.point);

            Point2i direction2d(item.dir[0], item.dir[1]);
            directionBounds = Union(directionBounds, direction2d);
        }

        // 3. Find the longest dimension for split for tight bounds
        const Vector3f spaceDiagonal = spaceBounds.Diagonal();
        const Vector2i dirDiagonal = directionBounds.Diagonal();

        const Float normalizedExtents[5] = {
            spaceDiagonal.x / m_partitions.sceneExtent.x,
            spaceDiagonal.y / m_partitions.sceneExtent.y,
            spaceDiagonal.z / m_partitions.sceneExtent.z,
            dirDiagonal.x / 65535.f,
            dirDiagonal.y / 65535.f,
        };

        const Float* maxExtent = std::max_element(normalizedExtents, normalizedExtents + 5);
        if (*maxExtent <= MathEpsilon) {
            // degenerate bounds
            return emitLeaf();
        }

        splitAxis = static_cast<int>(maxExtent - normalizedExtents);

        // 4. Find the median and split value in O(n) time
        auto findPartitionSplit = [&items, start, end](Float& splitValue, auto getVal) -> int {
            int medianIndex = (start + end) / 2;
            auto startIt = items.begin() + start;
            auto endIt = items.begin() + end;
            auto midIt = items.begin() + medianIndex;
            
            std::nth_element(startIt, midIt, endIt, [&getVal](const ShadingPoint& a, const ShadingPoint& b){
                return getVal(a) < getVal(b);
            });

            const Float median = getVal(*midIt);
            splitValue = median;
            
            // 3-way partition
            auto leftEnd = std::partition(startIt, endIt, [&getVal, median](const ShadingPoint& x){
                return getVal(x) < median;
            });
            auto rightBegin = std::partition(leftEnd, endIt, [&getVal, median](const ShadingPoint& x){
                return getVal(x) == median;
            });

            const bool isMedianUnique = std::distance(leftEnd, rightBegin) == 1;
            if (isMedianUnique) {
                return std::distance(items.begin(), rightBegin);
            }

            const size_t strictlyLess = std::distance(startIt, leftEnd);
            const size_t strictlyMore = std::distance(rightBegin, endIt);
            
            auto finalMid = leftEnd;
            if (strictlyLess >= strictlyMore) {
                // median duplicates went right.
                // left side has values < median. Right has >= median.
                // Traversal always uses <= for the left child
                // Must find the highest value on the left for the splitValue
                auto maxLeftIt = std::max_element(startIt, leftEnd, [&getVal](const ShadingPoint& a, const ShadingPoint& b){
                    return getVal(a) < getVal(b);
                });

                const Float maxLeftVal = getVal(*maxLeftIt);
                // try to split the empty space
                const Float gapMidPoint = (maxLeftVal + median) * Float(0.5);
                splitValue = gapMidPoint >= median ? maxLeftVal : gapMidPoint;
            } else {
                finalMid = rightBegin;
                // median duplicates went left
                // finding the split value in the gap space between median and minValueRight
                auto minRightIt = std::min_element(rightBegin, endIt, [&getVal](const ShadingPoint& a, const ShadingPoint& b){
                    return getVal(a) < getVal(b);
                });

                const Float minRightVal = getVal(*minRightIt);
                // try to split the empty space
                const Float gapMidPoint = (median + minRightVal) * Float(0.5);
                splitValue = gapMidPoint >= minRightVal ? median : gapMidPoint;
            }

            return std::distance(items.begin(), finalMid);
        };

        if (splitAxis < 3) {
            int dim = splitAxis;
            splitIndex = findPartitionSplit(splitValue,
                [dim](const ShadingPoint& x) { return x.point[dim]; });
        } else {
            int dim = splitAxis - 3;
            splitIndex = findPartitionSplit(splitValue,
                [dim](const ShadingPoint& x) { return x.dir[dim]; });
        }
    }

    // 5. Recursion
    DCHECK_GT(splitIndex, start);
    DCHECK_LT(splitIndex, end);
    const uint32_t reservationIndex = m_partitions.innerNodes.size();
    m_partitions.innerNodes.emplace_back(PartitionTreeNode::MakeInterior(splitAxis, splitValue, 0));

    const uint32_t leftChildIdx = BuildPartitionTree(items, start, splitIndex);
    DCHECK_EQ(leftChildIdx, reservationIndex + 1);

    const uint32_t rightChildIdx = BuildPartitionTree(items, splitIndex, end);
    
    m_partitions.innerNodes[reservationIndex].rightChildOrLeafIndex = rightChildIdx;
    return reservationIndex;
}

PBRT_CPU_GPU
static void MakeInitialTreeCut(OnlineLightTreeCut& cut, ShadingPoint representant, const LTCTreeNode* nodes, Bounds3f allLightBounds) {
    constexpr uint32_t initialCutSize = 4;

    auto AddCluster = [&cut](uint32_t cutIndex, uint32_t index, Float importance, uint32_t bitTrail, uint32_t depth) {
        cut.q[cutIndex] = importance;
        cut.variance[cutIndex] = 0;
        cut.clusterIndex[cutIndex] = index;
        cut.bitTrail[cutIndex] = bitTrail;
        cut.depth[cutIndex] = depth;
        cut.visitCount[cutIndex] = 0;
    };

    // Initialize the cut with the root node
    const LTCTreeNode* root = &nodes[0];
    AddCluster(0, 0, root->compactLightBounds.PhiOrI(), 0, 0);
    cut.cutSize = 1;
    cut.currentIteration = 1;
    cut.lastUpdateIteration = 0;

    const Point3f p = representant.point;
    const Vector3f wo(representant.dir);
    const Normal3f n(wo);

    const Frame shadingFrame = Frame::FromZ(n);

    // iteratively refine the cut until we hit the target size
    while (cut.cutSize < initialCutSize) {
        int maxIdx = -1;
        Float maxImportance = -1.0f;

        // Find the node in the current cut with the highest importance
        for (uint32_t i = 0; i < cut.cutSize; ++i) {
            uint32_t nodeIndex = cut.clusterIndex[i];
            const LTCTreeNode* node = &nodes[nodeIndex];

            const bool isLeaf = node->isLeaf;
            if (!isLeaf && cut.q[i] > maxImportance) {
                maxImportance = cut.q[i];
                maxIdx = i;
            }
        }

        if (maxIdx == -1) {
            break;
        }

        // expand the selected node
        //OnlineCutData parentData = cut.data[maxIdx];
        const uint32_t clusterIndex = cut.clusterIndex[maxIdx];
        const LTCTreeNode* parentNode = &nodes[clusterIndex];

        const uint32_t leftChildIndex = clusterIndex + 1;
        const uint32_t rightChildIndex = parentNode->childOrLightIndex;

        const LTCTreeNode* leftChild = &nodes[leftChildIndex];
        const LTCTreeNode* rightChild = &nodes[rightChildIndex];

        Float lu0, lu1;
        if (!ComputeErrorBounds(lu0, lu1, p, wo, n, shadingFrame, nullptr, leftChild->compactLightBounds, rightChild->compactLightBounds, allLightBounds)) {
            lu0 = std::max(leftChild->compactLightBounds.PhiOrI() *  MathEpsilon, MathEpsilon);
            lu1 = std::max(rightChild->compactLightBounds.PhiOrI() * MathEpsilon, MathEpsilon);
        }

        if (lu0 == 0) {
            lu0 = std::max(leftChild->compactLightBounds.PhiOrI() * MathEpsilon, MathEpsilon);
        }

        if (lu1 == 1) {
            lu1 = std::max(rightChild->compactLightBounds.PhiOrI() * MathEpsilon, MathEpsilon);
        }

        const uint32_t parentBitTrail = cut.bitTrail[maxIdx];
        const uint32_t parentDepth = cut.depth[maxIdx];
        AddCluster(maxIdx, leftChildIndex, lu0, parentBitTrail, parentDepth + 1);
        AddCluster(cut.cutSize, rightChildIndex, lu1, parentBitTrail | (1 << parentDepth), parentDepth + 1);

        ++cut.cutSize;
    }

    Float sum = 0;
    for (uint32_t i = 0; i < cut.cutSize; ++i) {
        sum += cut.q[i];
        cut.prefixSum[i] = sum;
    }
}

void LTCLightSampler::SetupScenePartitions(pstd::span<ShadingPoint> shadingPoints, const Bounds3f& sceneBounds) {
    if (shadingPoints.empty())
        return;

    m_partitions.sceneExtent = sceneBounds.Diagonal();
    BuildPartitionTree(shadingPoints, 0, shadingPoints.size());
    
    if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
        GPUParallelFor("Initialize LTC tree cuts", ProfilerKernelGroup::WAVEFRONT, m_partitions.leaves.size(),
            [this] PBRT_GPU(int idx) {
            MakeInitialTreeCut(*m_partitions.leaves[idx], m_partitions.representantPoints[idx], m_tree.nodes.data(), m_tree.allLightBounds);
        });
        GPUWait();
#else
        LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
    }
    else {
        ParallelFor(0, m_partitions.leaves.size(), [this](int idx) {
            MakeInitialTreeCut(*m_partitions.leaves[idx], m_partitions.representantPoints[idx], m_tree.nodes.data(), m_tree.allLightBounds);
        });
    }

    LTCScenePartitionBytes += m_partitions.innerNodes.size() * sizeof(PartitionTreeNode) + 
                              m_partitions.leaves.size() * sizeof(OnlineLightTreeCut) +
                              m_partitions.leaves.size() * sizeof(OnlineLightTreeCut*) +
                              m_partitions.representantPoints.size() * sizeof(ShadingPoint);
}

PBRT_CPU_GPU
static void ApplyIterationUpdate(OnlineLightTreeCut& cut, const ShadingPoint& representant, const LTCLightTree* tree, uint32_t cutIndex, const Float gamma, const Float beta, const Float omega) {
    const uint32_t lastCutSize = cut.cutSize;
    
    // max cut reached
    if (lastCutSize == PBRT_LTC_MAX_CUT_SIZE - 1) {
        return;
    }
    
    // stop refining if no updates happen for a while
    if (cut.currentIteration > cut.lastUpdateIteration &&
        static_cast<Float>(cut.currentIteration - cut.lastUpdateIteration) / lastCutSize > gamma) {
        return;
    }

    // initial sampling budget for learning
    constexpr Float n0 = 8.0f;
    constexpr Float initialCutSize = 4.0f;
    const Float nt = std::max(lastCutSize / initialCutSize, Float(2)) * n0;

    //Count actual pixel samples influencing this cut
    uint32_t totalCutSamples = 0;
    for (uint32_t idx = 0; idx < lastCutSize; ++idx) {
        totalCutSamples += cut.visitCountAccumulator[idx];
    }

    if (totalCutSamples < nt) {
        return;
    }

    const uint32_t currentIteration = cut.currentIteration;
    const Float t = currentIteration;
    const Float learningRate = 1 / (beta * std::pow(t, omega));

    ++cut.currentIteration;

    Float sumVariance = 0;
    for (uint32_t idx = 0; idx < lastCutSize; ++idx) {
        const uint32_t numSamples = cut.visitCountAccumulator[idx];
        Float var = cut.variance[idx];

        if (numSamples >= 1) {
            // Batch statistic for the current iteration
            const Float batchMean = cut.sumAccumulator[idx] / numSamples;

            // https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
            // incremental EMA for mean and variance, without the need for second moment
            Float importance = cut.q[idx];
            const Float meanDelta = batchMean - importance;
            importance += learningRate * meanDelta;

            const Float varianceDelta = meanDelta * (batchMean - importance) - var;
            var += learningRate * varianceDelta;
            
            const Float visitedRatio = static_cast<Float>(numSamples) / totalCutSamples;
            const uint32_t scaledVisits = visitedRatio * nt;

            cut.q[idx] = importance;
            cut.variance[idx] = var;
            cut.visitCount[idx] += scaledVisits;
        }

        cut.prefixSum[idx] = 0;
        cut.sumAccumulator[idx] = 0;
        cut.visitCountAccumulator[idx] = 0;
        sumVariance += var;
    }
    sumVariance += MathEpsilon;

    uint32_t splitCount = 0;
    if (sumVariance > MathEpsilon) {
        const Point3f p = representant.point;
        const Vector3f wo(representant.dir);
        const Normal3f n(wo);
        const Frame shadingFrame = Frame::FromZ(n);

        constexpr uint32_t maxToSplit = PBRT_LTC_MAX_CUT_SIZE / 4;
        uint32_t toSplit[maxToSplit];
        uint32_t newCutSize = lastCutSize;

        RNG rng(Hash(sumVariance, cutIndex, currentIteration));
        for (uint32_t i = 0; i < lastCutSize; ++i) {
            const uint32_t clusterIndex = cut.clusterIndex[i];
            const uint32_t visitCount = cut.visitCount[i];
            const LTCTreeNode* node = &tree->nodes[clusterIndex];

            if (node->isLeaf || newCutSize >= PBRT_LTC_MAX_CUT_SIZE - 1 || visitCount <= 1) {
                continue;
            }

            const Float var = cut.variance[i];
            const Float relativeVariance = var / sumVariance;
            const Float visitTerm = (visitCount - 1) / static_cast<Float>(visitCount);

            const Float relativeSize = lastCutSize / initialCutSize;
            const Float splitProb = relativeVariance * visitTerm / (1 + relativeSize * std::exp(-var));

            const Float u = rng.Uniform<Float>();
            if (u <= splitProb) {
                toSplit[splitCount] = i;
                ++splitCount;
                ++newCutSize;
            }
        }

        const Bounds3f allLightBounds = tree->allLightBounds;
        for (uint32_t i = 0; i < splitCount; ++i) {
            const uint32_t cutIdx = toSplit[i];
            const uint32_t clusterIndex = cut.clusterIndex[cutIdx];
            const uint32_t parentBitTrail = cut.bitTrail[cutIdx];
            const uint32_t parentDepth = cut.depth[cutIdx];
            const Float parentImportance = cut.q[cutIdx];

            const LTCTreeNode* node = &tree->nodes[clusterIndex];
            const uint32_t childIndex0 = clusterIndex + 1;
            const uint32_t childIndex1 = node->childOrLightIndex;
            
            const LTCTreeNode* child0 = &tree->nodes[childIndex0];
            const LTCTreeNode* child1 = &tree->nodes[childIndex1];
            
            // Child initialization
            Float lu0, lu1;
            if (!ComputeErrorBounds(lu0, lu1, p, wo, n, shadingFrame, nullptr, child0->compactLightBounds, child1->compactLightBounds, allLightBounds, true)) {
                // the representative light could be a bad pick for the cluster of light
                // fallback to the light intensities
                lu0 = std::max(child0->compactLightBounds.PhiOrI() * MathEpsilon, MathEpsilon);
                lu1 = std::max(child1->compactLightBounds.PhiOrI() * MathEpsilon, MathEpsilon);
            }

            if (lu0 == 0) {
                lu0 = std::max(MathEpsilon, child0->compactLightBounds.PhiOrI() * MathEpsilon);
            }
            
            if (lu1 == 0) {
                lu1 = std::max(MathEpsilon, child1->compactLightBounds.PhiOrI() * MathEpsilon);
            }

            Float sqrt0 = std::sqrt(lu0);
            Float sqrt1 = std::sqrt(lu1);
            Float sumSqrt = sqrt0 + sqrt1;
            const Float probLeft = std::min(sqrt0 / sumSqrt, OneMinusEpsilon);
            const Float probRight = 1 - probLeft;

            // Approximate visit counts based on the bounds
            const uint32_t visitCount = cut.visitCount[cutIdx];
            const Float nc0 = probLeft * visitCount;
            const Float nc1 = probRight * visitCount;

            const Float OneMinusLearningRate = 1 - learningRate;

            // decay factors
            const Float decay0 = std::pow(OneMinusLearningRate, nc0);
            const Float decay1 = std::pow(OneMinusLearningRate, nc1);

            {
                const Float importance = decay0 * lu0 + (1 - decay0) * parentImportance;
                cut.clusterIndex[cutIdx] = childIndex0;
                cut.bitTrail[cutIdx] = parentBitTrail;
                cut.depth[cutIdx] = parentDepth + 1;
                cut.q[cutIdx] = importance;
                cut.variance[cutIdx] = 0;
                cut.visitCount[cutIdx] = 0;
            }
            {
                const Float importance = decay1 * lu1 + (1 - decay1) * parentImportance;
                cut.clusterIndex[cut.cutSize] = childIndex1;
                cut.bitTrail[cut.cutSize] = parentBitTrail | (1 << parentDepth);
                cut.depth[cut.cutSize] = parentDepth + 1;
                cut.q[cut.cutSize] = importance;
                cut.variance[cut.cutSize] = 0;
                cut.visitCount[cut.cutSize] = 0;

                cut.sumAccumulator[cut.cutSize] = 0;
                cut.visitCountAccumulator[cut.cutSize] = 0;
                ++cut.cutSize;
            }
        }
    }

    Float sum = 0;
    for (uint32_t idx = 0; idx < cut.cutSize; ++idx) {
        const Float importance = cut.q[idx];
        sum += importance;
        cut.prefixSum[idx] = sum;
    }

    if (splitCount > 0) {
        cut.lastUpdateIteration = currentIteration;
    }
}

void LTCLightSampler::Update(const uint32_t currentIteration) {
#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        // Must wait for the previous work to be completed before accessing it.
        GPUWait();
    }
#endif

    if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
        GPUParallelFor("Apply LTC update to partition cuts", ProfilerKernelGroup::WAVEFRONT, m_partitions.leaves.size(),
            [this] PBRT_GPU(int idx) {
            OnlineLightTreeCut& cut(*m_partitions.leaves[idx]);
            ApplyIterationUpdate(cut, m_partitions.representantPoints[idx], &m_tree, idx, m_gamma, m_beta, m_omega);
        });

        GPUWait();
#else
        LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
    }
    else {
        ParallelFor(0, m_partitions.leaves.size(), [this](int idx) {
            OnlineLightTreeCut& cut(*m_partitions.leaves[idx]);
            ApplyIterationUpdate(cut, m_partitions.representantPoints[idx], &m_tree, idx, m_gamma, m_beta, m_omega);
        });
    }
}

PBRT_CPU_GPU
uint32_t LTCLightSampler::GetPartitionIndex(const Point3f& p, const UniformDiskVector& vec) const {
    if (m_partitions.innerNodes.empty()) {
        return 0;
    }

    uint32_t nodeIndex = 0;
    const PartitionTreeNode* node = &m_partitions.innerNodes[nodeIndex];

    while (!node->IsLeaf()) {
        // extract the value to test based on the split axis of the node
        Float testValue;
        const uint32_t splitAxis = node->splitAxis;
        if (splitAxis < 3) {
            testValue = p[splitAxis];
        } else {
            testValue = static_cast<Float>(vec[splitAxis - 3]);
        }

        const uint32_t childIndices[2] = { nodeIndex + 1,
                                           node->rightChildOrLeafIndex };

        // left child == 0 when testValue <= splitValue
        const int child = testValue > node->splitValue;
        nodeIndex = childIndices[child];
        node = &m_partitions.innerNodes[nodeIndex];
    }

    return node->rightChildOrLeafIndex;
}

std::string LTCLightSampler::ToString() const {
    return StringPrintf("[ LTCLightSampler nodes: %s ]", m_tree.nodes);
}


}
