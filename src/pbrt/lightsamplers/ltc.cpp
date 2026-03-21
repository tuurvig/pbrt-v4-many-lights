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
    leaves(alloc), innerNodes(alloc) {}

PBRT_CPU_GPU
void LTCLightSampler::MakeInitialTreeCut(OnlineLightTreeCut& cut, const ShadingPoint& representant) const {
    cut.cutSize = 0;

    const LightcutsTreeNode* root = &m_tree.nodes[0];
    const uint32_t childIndex0 = 1;
    const uint32_t childIndex1 = root->childOrLightIndex;

    const LightcutsTreeNode* child0 = &m_tree.nodes[childIndex0];
    const LightcutsTreeNode* child1 = &m_tree.nodes[childIndex1];

    const uint32_t childIndex00 = childIndex0 + 1;
    const uint32_t childIndex01 = child0->childOrLightIndex;
    const uint32_t childIndex10 = childIndex0 + 1;
    const uint32_t childIndex11 = child0->childOrLightIndex;

    const LightcutsTreeNode* child00 = &m_tree.nodes[childIndex00];
    const LightcutsTreeNode* child01 = &m_tree.nodes[childIndex01];
    const LightcutsTreeNode* child10 = &m_tree.nodes[childIndex10];
    const LightcutsTreeNode* child11 = &m_tree.nodes[childIndex11];

    auto AddCluster = [](OnlineCutData& cluster, uint32_t index, Float importance, uint32_t bitTrail, uint32_t depth) {
        cluster.q = importance;
        cluster.variance = 0;
        cluster.secondMoment = 0;

        cluster.clusterIndex = index;
        cluster.bitTrail = bitTrail;
        cluster.depth = depth;
        cluster.visitCount = 0;
    };

    // light clustering is initialized based on the total power of each light cluster
    AddCluster(cut.data[cut.cutSize], childIndex00, child00->compactLightBounds.PhiOrI(), 0, 2);
    ++cut.cutSize;
    AddCluster(cut.data[cut.cutSize], childIndex01, child01->compactLightBounds.PhiOrI(), 1, 2);
    ++cut.cutSize;
    AddCluster(cut.data[cut.cutSize], childIndex10, child10->compactLightBounds.PhiOrI(), 2, 2);
    ++cut.cutSize;
    AddCluster(cut.data[cut.cutSize], childIndex11, child11->compactLightBounds.PhiOrI(), 3, 2);
    ++cut.cutSize;
}

uint32_t LTCLightSampler::BuildPartitionTree(pstd::span<ShadingPoint>& items, int start, int end) {
    DCHECK_LT(start, end);

    // 1. Base Case: Leaf node
    if (end - start <= 250) {
        uint32_t leafIndex = static_cast<uint32_t>(m_partitions.leaves.size());
        uint32_t nodeIndex = static_cast<uint32_t>(m_partitions.innerNodes.size());
        m_partitions.innerNodes.emplace_back(PartitionTreeNode::MakeLeaf(leafIndex));
        m_partitions.leaves.emplace_back();
        
        Point3f avgPosition(0, 0, 0);
        DirectionCone cone;
        for (size_t i = 0; i < items.size(); ++i) {
            avgPosition += items[i].point;

            Vector3f normal(items[i].dir);
            Union(cone, DirectionCone(normal));
        }
        
        avgPosition /= items.size();
        m_partitions.representantPoints.emplace_back(avgPosition, OctahedralVector(cone.w));

        return nodeIndex;
    }

    int splitAxis = -1;
    int mid = -1;
    Float splitValue = std::numeric_limits<Float>::max();
    {
        // 2. Find tight bounds for the current node
        Bounds3f spaceBounds;
        Bounds2i directionBounds;
        for (int i = start; i < end; ++i) {
            const ShadingPoint& item(items[i]);
            spaceBounds = Union(spaceBounds, item.point);

            Point2i direction2d(item.dir.X(), item.dir.Y());
            directionBounds = Union(directionBounds, direction2d);
        }

        // 3. Find the longest dimension for split for tight bounds
        const Vector3f spaceDiagonal = spaceBounds.Diagonal();
        const Vector2i dirDiagonal = directionBounds.Diagonal();

        Float normalizedExtents[5] = {
            spaceDiagonal.x / m_partitions.sceneExtent.x,
            spaceDiagonal.y / m_partitions.sceneExtent.y,
            spaceDiagonal.z / m_partitions.sceneExtent.z,
            dirDiagonal.x / 65535.f,
            dirDiagonal.y / 65535.f,
        };

        splitAxis = std::distance(normalizedExtents, std::max_element(normalizedExtents, normalizedExtents + 5));

        // 4. Find the median and split value in O(n) time
        mid = (start + end) / 2;
        const int dim = splitAxis;

        if (dim < 3) {
            std::nth_element(items.begin() + start, items.begin() + mid, items.end() + end,
                [dim](const ShadingPoint& a, const ShadingPoint& b) {
                    return a.point[dim] < b.point[dim];
                });
            splitValue = items[mid].point[dim];
        } else if (dim == 3){
            std::nth_element(items.begin() + start, items.begin() + mid, items.end() + end,
                [](const ShadingPoint& a, const ShadingPoint& b) {
                    return a.dir.X() < b.dir.X();
                });
            const auto splitInt = items[mid].dir.X();
            splitValue = splitInt;
            if (splitInt == directionBounds.pMax.x || splitInt == directionBounds.pMin.x) {
                splitValue = (directionBounds.pMax.x + directionBounds.pMin.x) * Float(0.5);
                auto midIt = std::partition(items.begin() + start, items.end() + end,
                    [=](const ShadingPoint& item) {
                        return Float(item.dir.X()) <= splitValue;
                    });
                mid = std::distance(items.begin(), midIt);
            }
        } else {
            std::nth_element(items.begin() + start, items.begin() + mid, items.end() + end,
                [](const ShadingPoint& a, const ShadingPoint& b) {
                    return a.dir.Y() < b.dir.Y();
                });
            const auto splitInt = items[mid].dir.Y();
            splitValue = splitInt;
            if (splitInt == directionBounds.pMax.y || splitInt == directionBounds.pMin.y) {
                splitValue = (directionBounds.pMax.y + directionBounds.pMin.y) * Float(0.5);
                auto midIt = std::partition(items.begin() + start, items.end() + end,
                    [=](const ShadingPoint& item) {
                        return Float(item.dir.Y()) <= splitValue;
                    });
                mid = std::distance(items.begin(), midIt);
            }
        }
    }

    // 5. Recursion
    const uint32_t reservationIndex = m_partitions.innerNodes.size();
    m_partitions.innerNodes.emplace_back();

    const uint32_t leftChildIdx = BuildPartitionTree(items, start, mid);
    DCHECK_EQ(leftChildIdx, reservationIndex + 1);

    const uint32_t rightChildIdx = BuildPartitionTree(items, mid, end);
    
    m_partitions.innerNodes[reservationIndex] = PartitionTreeNode::MakeInterior(splitAxis, splitValue, rightChildIdx);
    return reservationIndex;
}

void LTCLightSampler::SetupScenePartitions(pstd::span<ShadingPoint> shadingPoints, const Bounds3f& sceneBounds) {
    m_partitions.sceneExtent = sceneBounds.Diagonal();
    BuildPartitionTree(shadingPoints, 0, shadingPoints.size());
    
    for (size_t i = 0; i < m_partitions.representantPoints.size(); ++i) {
        MakeInitialTreeCut(m_partitions.leaves[i], m_partitions.representantPoints[i]);
    }
//    if (Options->useGPU) {
//#ifdef PBRT_BUILD_GPU_RENDERER
//        GPUParallelFor("Initialize LTC tree cuts", ProfilerKernelGroup::LTC, m_partitions.leaves.size(),
//            PBRT_CPU_GPU [=](int idx) mutable {
//            MakeInitialTreeCut(m_partitions.leaves[idx], representantShadingPoints[idx]);
//        });
//#else
//        LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
//#endif
//    }
//    else {
//        ParallelFor(0, m_partitions.leaves.size(), [this, &representantShadingPoints](int idx) {
//            MakeInitialTreeCut(m_partitions.leaves[idx], representantShadingPoints[idx]);
//        });
//    }

    LTCScenePartitionBytes += m_partitions.innerNodes.size() * sizeof(PartitionTreeNode) + 
                              m_partitions.leaves.size() * sizeof(OnlineLightTreeCut) + 
                              m_partitions.representantPoints.size() * sizeof(ShadingPoint);
}

PBRT_CPU_GPU
void LTCLightSampler::Update(const uint32_t currentIteration) {
    const Float t = currentIteration;
    const Float learningRate = 1 / (m_beta * std::pow(t, m_omega));

    for (size_t i = 0; i < m_partitions.leaves.size(); ++i) {
        ApplyIterationUpdate(m_partitions.leaves[i], m_partitions.representantPoints[i], currentIteration, learningRate);
    }
}

PBRT_CPU_GPU
void LTCLightSampler::ApplyIterationUpdate(OnlineLightTreeCut& cut, const ShadingPoint& representant, const uint32_t currentIteration, const Float learningRate) const {
    // max cut reached
    if (cut.cutSize == PBRT_LTC_MAX_CUT_SIZE - 1) {
        return;
    }

    const uint32_t lastCutSize = cut.cutSize;
    const Float OneMinusLearningRate = 1 - learningRate;

    // stop refining if no updates happen for a while
    if (currentIteration > cut.lastUpdateIteration &&
        static_cast<Float>(currentIteration - cut.lastUpdateIteration) / lastCutSize > m_gamma) {
        return;
    }
   
    Float sumVariance = 0;
    for (uint32_t idx = 0; idx < lastCutSize; ++idx) {
        const uint32_t numSamples = cut.visitCountAccumulator[idx];
        if (numSamples < 1)
            return;

        // Batch statistic for the current iteration
        const Float batchMean = cut.sumAccumulator[idx] / numSamples;
        const Float batchSqrMean = cut.sumSquaredAccumulator[idx] / numSamples;

        // update the global EMA statistics for first and second moments
        OnlineCutData& cluster(cut.data[idx]);
        cluster.q = OneMinusLearningRate * cluster.q + learningRate * batchMean;
        cluster.secondMoment = OneMinusLearningRate * cluster.secondMoment + learningRate * batchSqrMean;
        
        // update variance from accumulated statistics
        // V = E[X^2] - E[X]^2
        cluster.variance = std::max(cluster.secondMoment - Sqr(cluster.q), Float(0));

        cut.sumAccumulator[idx] = 0;
        cut.sumSquaredAccumulator[idx] = 0;
        cut.visitCountAccumulator[idx] = 0;

        sumVariance += cluster.variance;
    }
    sumVariance += MathEpsilon;

    const Point3f p = representant.point;
    const Vector3f wo(representant.dir);
    const Normal3f n(wo);
    const Frame shadingFrame = Frame::FromZ(n);

    constexpr Float initialCutSize = 4;
    constexpr uint32_t maxToSplit = PBRT_LTC_MAX_CUT_SIZE / 2;
    uint32_t toSplit[maxToSplit];
    uint32_t splitCount = 0;
    uint32_t newCutSize = lastCutSize;

    for (uint32_t i = 0; i < lastCutSize; ++i) {
        OnlineCutData cluster(cut.data[i]);
        const LightcutsTreeNode* node = &m_tree.nodes[cluster.clusterIndex];

        if (node->isLeaf || newCutSize >= PBRT_LTC_MAX_CUT_SIZE - 1 || cluster.visitCount <= 1) {
            continue;
        }

        const Float relativeVariance = cluster.variance / sumVariance;
        const Float visitTerm = (cluster.visitCount - 1) / static_cast<Float>(cluster.visitCount);

        const Float relativeSize = lastCutSize / initialCutSize;
        const Float splitProb = relativeVariance * visitTerm / (1 + relativeSize * std::exp(-cluster.variance));

        const Float u = HashFloat(i, cluster.clusterIndex, currentIteration);
        if (u <= splitProb) {
            toSplit[splitCount] = i;
            ++splitCount;
            ++newCutSize;
        }
    }

    for (uint32_t i = 0; i < splitCount; ++i) {
        OnlineCutData& cluster(cut.data[toSplit[i]]);
        const LightcutsTreeNode* node = &m_tree.nodes[cluster.clusterIndex];
        const uint32_t childIndex0 = cluster.clusterIndex + 1;
        const uint32_t childIndex1 = node->childOrLightIndex;
        const LightcutsTreeNode* child0 = &m_tree.nodes[childIndex0];
        const LightcutsTreeNode* child1 = &m_tree.nodes[childIndex1];

        // Child initialization
        Float lu0, lu1;
        if (!ComputeErrorBounds(lu0, lu1, p, wo, n, shadingFrame, nullptr, child0, child1, m_tree.allLightBounds, true)) {
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

        // decay factors
        const Float decay0 = std::pow(OneMinusLearningRate, nc0);
        const Float decay1 = std::pow(OneMinusLearningRate, nc1);

        {
            OnlineCutData c0;
            c0.clusterIndex = childIndex0;
            c0.bitTrail = cluster.bitTrail;
            c0.depth = cluster.depth + 1;
            c0.visitCount = 0;
            c0.variance = 0;
            c0.secondMoment = 0;
            c0.q = decay0 * lu0 + (1 - decay0) * cluster.q;
            cluster = c0;
        }
        {
            OnlineCutData c1;
            c1.clusterIndex = childIndex1;
            c1.bitTrail = cluster.bitTrail | (1 << cluster.depth);
            c1.depth = cluster.depth + 1;
            c1.visitCount = 0;
            c1.variance = 0;
            c1.secondMoment = 0;
            c1.q = decay1 * lu1 + (1 - decay1) * cluster.q;
            cut.data[cut.cutSize] = c1;
            ++cut.cutSize;
        }
    }

    if (splitCount > 0) {
        cut.lastUpdateIteration = currentIteration;
    }
}

PBRT_CPU_GPU
uint32_t LTCLightSampler::GetPartitionIndex(const Point3f& p, const OctahedralVector& oct) const {
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
        } else if (splitAxis == 3) {
            testValue = static_cast<Float>(oct.X());
        } else {
            testValue = static_cast<Float>(oct.Y());
        }

        const uint32_t childIndices[2] = { nodeIndex + 1,
                                           node->rightChildOrLeafIndex };

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
