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
    std::vector<LightcutsBuildContainer> treeLights;
    for (size_t i = 0; i < lights.size(); ++i) {
        // Store $i$th light in either _infiniteLights_ or _treeLights_
        Light light = lights[i];
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        if (!lightBounds) {
            m_infiniteLights.push_back(light);
        }
        else if (lightBounds->phi > 0) {
            treeLights.emplace_back(*lightBounds, light);
            m_tree.allLightBounds = Union(m_tree.allLightBounds, lightBounds->bounds);
        }
    }

    RNG rng;
    Float u = rng.Uniform<Float>();
    if (!treeLights.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        bool buildOnGPU = buildLightTreeGPU(treeLights, u);
        if (!buildOnGPU)
#endif
        {
            SLCNodeEmitter emitter(m_tree, m_lightToBitTrail);
            BuildLightTree<16, LightcutsBuildContainer, SAOHCostEvaluator, SLCNodeEmitter>(treeLights, 0, treeLights.size(), 0, 0, SAOHCostEvaluator(), emitter, u);
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

    bool Build(std::vector<LightcutsBuildContainer> &lights) {
        if (lights.empty())
            return false;

        Allocate(static_cast<uint32_t>(lights.size()), m_allLightBounds);

        LightTreeBuildState<LightBounds> buildState(State());
        std::array<uint8_t, 3> ax = DetermineAxisOrder(buildState.allLightBounds);

        LightcutsBuildContainer* dLightsContainer = GPUAllocAsync<LightcutsBuildContainer>(buildState.nLights);
        GPUCopyToDevice(dLightsContainer, lights.data(), lights.size());

        uint64_t* dMortonCodes = MortonCodes();
        MortonCodes() = SortNodesMorton(State(), MortonCodes(), [buildState, ax, dLightsContainer, dMortonCodes] PBRT_GPU(int idx) {
            LightcutsBuildContainer cont = dLightsContainer[idx];
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

    void FlattenTree(LightcutsTree& tree, std::vector<LightcutsBuildContainer> &lights, HashMap<Light, uint32_t>& bitTrailContainer, Float& u) {
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

        SLCNodeEmitter emitter(tree, bitTrailContainer);
        GPUToLightcutsLeaf adapter(hostNodes, lights);
        
        FlattenLightTree<GPUToLightcutsLeaf, SLCNodeEmitter>(adapter, rootIndex, 0, 0, emitter, u);
    }

private:
    Bounds3f m_allLightBounds;
};

bool LTCLightSampler::buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights, Float& u) {
    if (lights.size() < 100 || !Options->useGPU)
        return false;

    LTCTreeBuilderGPU builder(m_tree.allLightBounds);
    if (!builder.Build(lights))
        return false;

    builder.FlattenTree(m_tree, lights, m_lightToBitTrail, u);
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
static void MakeInitialTreeCut(OnlineLightTreeCut& cut, const LightcutsTreeNode* nodes) {
    constexpr uint32_t initialCutSize = 4;

    auto AddCluster = [](OnlineCutData& cluster, uint32_t index, Float importance, uint32_t bitTrail, uint32_t depth) {
        cluster.q = importance;
        cluster.variance = 0;
        cluster.clusterIndex = index;
        cluster.bitTrail = bitTrail;
        cluster.depth = depth;
        cluster.visitCount = 0;
    };

    // Initialize the cut with the root node
    const LightcutsTreeNode* root = &nodes[0];
    AddCluster(cut.data[0], 0, root->compactLightBounds.PhiOrI(), 0, 0);
    cut.cutSize = 1;

    // iteratively refine the cut until we hit the target size
    while (cut.cutSize < initialCutSize) {
        int maxIdx = -1;
        Float maxImportance = -1.0f;

        // Find the node in the current cut with the highest importance
        for (uint32_t i = 0; i < cut.cutSize; ++i) {
            uint32_t nodeIndex = cut.data[i].clusterIndex;
            const LightcutsTreeNode* node = &nodes[nodeIndex];

            const bool isLeaf = node->isLeaf;
            if (!isLeaf && cut.data[i].q > maxImportance) {
                maxImportance = cut.data[i].q;
                maxIdx = i;
            }
        }

        if (maxIdx == -1) {
            break;
        }

        // expand the selected node
        OnlineCutData parentData = cut.data[maxIdx];
        const LightcutsTreeNode* parentNode = &nodes[parentData.clusterIndex];

        const uint32_t leftChildIndex = parentData.clusterIndex + 1;
        const uint32_t rightChildIndex = parentNode->childOrLightIndex;

        const LightcutsTreeNode* leftChild = &nodes[leftChildIndex];
        const LightcutsTreeNode* rightChild = &nodes[rightChildIndex];

        AddCluster(cut.data[maxIdx], leftChildIndex, leftChild->compactLightBounds.PhiOrI(), parentData.bitTrail, parentData.depth + 1);
        AddCluster(cut.data[cut.cutSize], rightChildIndex, rightChild->compactLightBounds.PhiOrI(), parentData.bitTrail | (1 << parentData.depth), parentData.depth + 1);

        ++cut.cutSize;
    }
}

void LTCLightSampler::SetupScenePartitions(pstd::span<ShadingPoint> shadingPoints, const Bounds3f& sceneBounds) {
    for (OnlineLightTreeCut *leaf : m_partitions.leaves)
        m_partitions.alloc.delete_object(leaf);
    m_partitions.leaves.clear();
    m_partitions.innerNodes.clear();
    m_partitions.representantPoints.clear();

    if (shadingPoints.empty())
        return;

    m_partitions.sceneExtent = sceneBounds.Diagonal();
    BuildPartitionTree(shadingPoints, 0, shadingPoints.size());
    
    if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
        const LightcutsTreeNode* treeNodes = m_tree.nodes.data();
        OnlineLightTreeCut** leaves = m_partitions.leaves.data();
        GPUParallelFor("Initialize LTC tree cuts", ProfilerKernelGroup::WAVEFRONT, m_partitions.leaves.size(),
            [treeNodes, leaves] PBRT_GPU(int idx) {
            MakeInitialTreeCut(*leaves[idx], treeNodes);
        });
        GPUWait();
#else
        LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
    }
    else {
        ParallelFor(0, m_partitions.leaves.size(), [this](int idx) {
            MakeInitialTreeCut(*m_partitions.leaves[idx], m_tree.nodes.data());
        });
    }

    LTCScenePartitionBytes += m_partitions.innerNodes.size() * sizeof(PartitionTreeNode) + 
                              m_partitions.leaves.size() * sizeof(OnlineLightTreeCut) +
                              m_partitions.leaves.size() * sizeof(OnlineLightTreeCut*) +
                              m_partitions.representantPoints.size() * sizeof(ShadingPoint);
}

PBRT_CPU_GPU
static bool ApplyIterationUpdate(OnlineLightTreeCut& cut, const ShadingPoint& representant, const LightcutsTree* tree, const uint32_t currentIteration, uint32_t cutIndex, const Float learningRate) {
    const uint32_t lastCutSize = cut.cutSize;

    // Count actual pixel samples influencing this cut
    uint32_t totalCutSamples = 0;
    for (uint32_t idx = 0; idx < lastCutSize; ++idx) {
        totalCutSamples += cut.visitCountAccumulator[idx];
    }

    // initial sampling budget for learning
    constexpr Float n0 = 4.0f;
    constexpr Float initialCutSize = 4.0f;
    const Float nt = std::max(lastCutSize / initialCutSize, Float(2)) * n0;

    Float sumVariance = 0;
    for (uint32_t idx = 0; idx < lastCutSize; ++idx) {
        const uint32_t numSamples = cut.visitCountAccumulator[idx];
        OnlineCutData& cluster(cut.data[idx]);
        if (numSamples >= 1) {
            // Batch statistic for the current iteration
            const Float batchMean = cut.sumAccumulator[idx] / numSamples;

            // https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
            // incremental EMA for mean and variance, without the need for second moment
            const Float meanDelta = batchMean - cluster.q;
            cluster.q += learningRate * meanDelta;

            const Float varianceDelta = meanDelta * (batchMean - cluster.q) - cluster.variance;
            cluster.variance += learningRate * varianceDelta;

            const Float visitedRatio = static_cast<Float>(numSamples) / totalCutSamples;
            const Float scaledVisits = visitedRatio * nt;

            cluster.visitCount += scaledVisits;
        }

        cut.sumAccumulator[idx] = 0;
        cut.visitCountAccumulator[idx] = 0;
        sumVariance += cluster.variance;
    }
    sumVariance += MathEpsilon;

    const Point3f p = representant.point;
    const Vector3f wo(representant.dir);
    const Normal3f n(wo);
    const Frame shadingFrame = Frame::FromZ(n);

    constexpr uint32_t maxToSplit = PBRT_LTC_MAX_CUT_SIZE / 2;
    uint32_t toSplit[maxToSplit];
    uint32_t splitCount = 0;
    uint32_t newCutSize = lastCutSize;

    RNG rng(Hash(sumVariance, cutIndex, currentIteration));
    for (uint32_t i = 0; i < lastCutSize; ++i) {
        OnlineCutData cluster(cut.data[i]);
        const LightcutsTreeNode* node = &tree->nodes[cluster.clusterIndex];

        if (node->isLeaf || newCutSize >= PBRT_LTC_MAX_CUT_SIZE - 1 || cluster.visitCount <= 1) {
            continue;
        }

        const Float relativeVariance = cluster.variance / sumVariance;
        const Float visitTerm = (cluster.visitCount - 1) / static_cast<Float>(cluster.visitCount);

        const Float relativeSize = lastCutSize / initialCutSize;
        const Float splitProb = relativeVariance * visitTerm / (1 + relativeSize * std::exp(-cluster.variance));

        const Float u = rng.Uniform<Float>();
        if (u <= splitProb) {
            toSplit[splitCount] = i;
            ++splitCount;
            ++newCutSize;
        }
    }

    const Bounds3f allLightBounds = tree->allLightBounds;
    for (uint32_t i = 0; i < splitCount; ++i) {
        OnlineCutData& cluster(cut.data[toSplit[i]]);
        const LightcutsTreeNode* node = &tree->nodes[cluster.clusterIndex];
        const uint32_t childIndex0 = cluster.clusterIndex + 1;
        const uint32_t childIndex1 = node->childOrLightIndex;
        const uint32_t parentBitTrail = cluster.bitTrail;
        const uint32_t parentDepth = cluster.depth;
        const LightcutsTreeNode* child0 = &tree->nodes[childIndex0];
        const LightcutsTreeNode* child1 = &tree->nodes[childIndex1];
        const Float parentImportance = cluster.q;

        // Child initialization
        Float lu0, lu1;
        if (!ComputeErrorBounds(lu0, lu1, p, wo, n, shadingFrame, nullptr, child0, child1, allLightBounds, true)) {
            // the representative light could be a bad pick for the cluster of light
            // fallback to the light intensities
            lu0 = child0->compactLightBounds.PhiOrI();
            lu1 = child1->compactLightBounds.PhiOrI();
        } else {
            lu0 = std::max(lu0, MathEpsilon);
            lu1 = std::max(lu1, MathEpsilon);
        }
        
        const Float luSum = lu0 + lu1;
        const Float probLeft = std::min(lu0 / luSum, OneMinusEpsilon);
        const Float probRight = 1 - probLeft;

        // Approximate visit counts based on the bounds
        const Float nc0 = probLeft * cluster.visitCount;
        const Float nc1 = probRight * cluster.visitCount;

        const Float OneMinusLearningRate = 1 - learningRate;

        // decay factors
        const Float decay0 = std::pow(OneMinusLearningRate, nc0);
        const Float decay1 = std::pow(OneMinusLearningRate, nc1);

        {
            OnlineCutData c0;
            const Float importance = decay0 * lu0 + (1 - decay0) * parentImportance;
            c0.clusterIndex = childIndex0;
            c0.bitTrail = parentBitTrail;
            c0.depth = parentDepth + 1;
            c0.q = importance;
            c0.variance = 0;
            c0.visitCount = 0;
            cluster = c0;
        }
        {
            OnlineCutData c1;
            const Float importance = decay1 * lu1 + (1 - decay1) * parentImportance;
            c1.clusterIndex = childIndex1;
            c1.bitTrail = parentBitTrail | (1 << parentDepth);
            c1.depth = parentDepth + 1;
            c1.q = importance;
            c1.variance = 0;
            c1.visitCount = 0;
            cut.data[cut.cutSize] = c1;
            cut.sumAccumulator[cut.cutSize] = 0;
            cut.visitCountAccumulator[cut.cutSize] = 0;
            ++cut.cutSize;
        }
    }

    return splitCount > 0;
}

void LTCLightSampler::Update(const uint32_t currentIteration) {
#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        // Must wait for the previous work to be completed before accessing it.
        GPUWait();
    }
#endif

    const Float t = currentIteration;
    const Float learningRate = 1 / (m_beta * std::pow(t, m_omega));

    //for (int idx = 0; idx < m_partitions.leaves.size(); ++idx) {
    //    OnlineLightTreeCut& cut(*m_partitions.leaves[idx]);
    //
    //    // max cut reached
    //    if (cut.cutSize == PBRT_LTC_MAX_CUT_SIZE - 1) {
    //        return;
    //    }
    //    
    //    const uint32_t lastCutSize = cut.cutSize;
    //    
    //    // stop refining if no updates happen for a while
    //    if (currentIteration > cut.lastUpdateIteration &&
    //        static_cast<Float>(currentIteration - cut.lastUpdateIteration) / lastCutSize > m_gamma) {
    //        return;
    //    }
    //
    //    if (ApplyIterationUpdate(cut, m_partitions.representantPoints[idx], &m_tree, currentIteration, idx, learningRate)) {
    //        cut.lastUpdateIteration = currentIteration;
    //    }
    //}
    //
    //return;

    if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
        OnlineLightTreeCut** leaves = m_partitions.leaves.data();
        ShadingPoint* representants = m_partitions.representantPoints.data();
        const LightcutsTree* tree = &m_tree;
        const Float gamma = m_gamma;
        GPUParallelFor("Apply LTC update to partition cuts", ProfilerKernelGroup::WAVEFRONT, m_partitions.leaves.size(),
            [tree, leaves, representants, currentIteration, learningRate, gamma] PBRT_GPU(int idx) {
            OnlineLightTreeCut& cut(*leaves[idx]);

            // max cut reached
            if (cut.cutSize == PBRT_LTC_MAX_CUT_SIZE - 1) {
                return;
            }
            
            const uint32_t lastCutSize = cut.cutSize;
            
            // stop refining if no updates happen for a while
            if (currentIteration > cut.lastUpdateIteration &&
                static_cast<Float>(currentIteration - cut.lastUpdateIteration) / lastCutSize > gamma) {
                return;
            }

            if (ApplyIterationUpdate(cut, representants[idx], tree, currentIteration, idx, learningRate)) {
                cut.lastUpdateIteration = currentIteration;
            }
        });

        GPUWait();
#else
        LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
    }
    else {
        ParallelFor(0, m_partitions.leaves.size(), [this, currentIteration, learningRate](int idx) {
            OnlineLightTreeCut& cut(*m_partitions.leaves[idx]);

            // max cut reached
            if (cut.cutSize == PBRT_LTC_MAX_CUT_SIZE - 1) {
                return;
            }
            
            const uint32_t lastCutSize = cut.cutSize;
            
            // stop refining if no updates happen for a while
            if (currentIteration > cut.lastUpdateIteration &&
                static_cast<Float>(currentIteration - cut.lastUpdateIteration) / lastCutSize > m_gamma) {
                return;
            }

            if (ApplyIterationUpdate(cut, m_partitions.representantPoints[idx], &m_tree, currentIteration, idx, learningRate)) {
                cut.lastUpdateIteration = currentIteration;
            }
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
