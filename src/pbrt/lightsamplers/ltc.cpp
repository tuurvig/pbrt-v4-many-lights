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

    // The simplest geometry proxy for outgoing direction (wo) is the normal of the surface.
    // This decision was made to not initialize cuts on demand during kernel execution due to thread divergence.
    // In this case, it is better to initialize all cuts at the same time by an approximate importance.
    // The importance collected inside the kernels will receive diferent rays with different wo values.
    // Approximating it with wo = n should be sufficient.

    const Point3f p = representant.point;
    const Vector3f wo(representant.dir);
    const Normal3f n(wo);
    const Frame shadingFrame = Frame::FromZ(n);

    Float errBound0, errBound1;
    if (!ComputeErrorBounds(errBound0, errBound1, p, wo, n, shadingFrame, nullptr, child0, child1, m_tree.allLightBounds, true)) {
        return;
    }

    auto AddCluster = [&](uint32_t index, Float importance, uint32_t bitTrail, uint32_t depth) {
        OnlineCutData& cluster(cut.data[cut.cutSize]);
        ++cut.cutSize;

        cluster.q = importance;
        cluster.variance = 0;
        cluster.secondMoment = 0;

        cluster.clusterIndex = index;
        cluster.bitTrail = bitTrail;
        cluster.depth = depth;
        cluster.visitCount = 0;
    };

    if (errBound0 != 0) {
        const uint32_t childIndex00 = childIndex0 + 1;
        const uint32_t childIndex01 = child0->childOrLightIndex;

        const LightcutsTreeNode* child00 = &m_tree.nodes[childIndex00];
        const LightcutsTreeNode* child01 = &m_tree.nodes[childIndex01];
        Float errBound00, errBound01;
        if (!ComputeErrorBounds(errBound00, errBound01, p, wo, n, shadingFrame, nullptr, child00, child01, m_tree.allLightBounds, true)) {
            return;
        }

        if (errBound00 != 0) {
            AddCluster(childIndex00, errBound00, 0, 2);
        }

        if (errBound01 != 0) {
            AddCluster(childIndex01, errBound01, 1, 2);
        }
    }

    if (errBound1 != 0) {
        const uint32_t childIndex10 = childIndex0 + 1;
        const uint32_t childIndex11 = child0->childOrLightIndex;

        const LightcutsTreeNode* child10 = &m_tree.nodes[childIndex10];
        const LightcutsTreeNode* child11 = &m_tree.nodes[childIndex11];
        Float errBound10, errBound11;
        if (!ComputeErrorBounds(errBound10, errBound11, p, wo, n, shadingFrame, nullptr, child10, child11, m_tree.allLightBounds, true)) {
            return;
        }

        if (errBound10 != 0) {
            AddCluster(childIndex10, errBound10, 2, 2);
        }

        if (errBound11 != 0) {
            AddCluster(childIndex11, errBound11, 3, 2);
        }
    }
}

uint32_t LTCLightSampler::BuildPartitionTree(pstd::span<ShadingPoint>& items, int start, int end) {
    DCHECK_LT(start, end);

    int splitAxis = -1;
    int mid = -1;
    Float splitValue = std::numeric_limits<Float>::max();
    {
        // 1. Find tight bounds for the current node
        Bounds3f spaceBounds;
        Bounds2i directionBounds;
        for (int i = start; i < end; ++i) {
            const ShadingPoint& item(items[i]);
            spaceBounds = Union(spaceBounds, item.point);

            Point2i direction2d(item.dir.X(), item.dir.Y());
            directionBounds = Union(directionBounds, direction2d);
        }

        // 2. Find the longest dimension for split for tight bounds
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

        // 3. Find the median and split value in O(n) time
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
            splitValue = items[mid].dir.X();
        } else {
            std::nth_element(items.begin() + start, items.begin() + mid, items.end() + end,
                [](const ShadingPoint& a, const ShadingPoint& b) {
                    return a.dir.Y() < b.dir.Y();
                });
            splitValue = items[mid].dir.Y();
        }

        // 4. Base Case: Leaf node
        if (end - start <= 250) {
            uint32_t leafIndex = static_cast<uint32_t>(m_partitions.leaves.size());
            uint32_t nodeIndex = static_cast<uint32_t>(m_partitions.innerNodes.size());
            m_partitions.representantPoints.emplace_back(items[mid]);
            m_partitions.innerNodes.emplace_back(PartitionTreeNode::MakeLeaf(leafIndex));
            m_partitions.leaves.emplace_back();
            return nodeIndex;
        }

        const auto splitInt = static_cast<uint16_t>(splitValue);
        if (dim == 3 && (splitInt == directionBounds.pMax.x || splitInt == directionBounds.pMin.x)) {
            splitValue = (directionBounds.pMax.x + directionBounds.pMin.x) * Float(0.5);
            auto midIt = std::partition(items.begin() + start, items.end() + end,
                [=](const ShadingPoint& item) {
                    return Float(item.dir.X()) <= splitValue;
                });
            mid = std::distance(items.begin(), midIt);
        } else if (dim == 4 && (splitInt == directionBounds.pMax.y || splitInt == directionBounds.pMin.y)) {
            splitValue = (directionBounds.pMax.y + directionBounds.pMin.y) * Float(0.5);
            auto midIt = std::partition(items.begin() + start, items.end() + end,
                [=](const ShadingPoint& item) {
                    return Float(item.dir.Y()) <= splitValue;
                });
            mid = std::distance(items.begin(), midIt);
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
void LTCLightSampler::Update(uint32_t currentIteration) {
    
}

PBRT_CPU_GPU
void LTCLightSampler::ApplyIterationUpdate(OnlineLightTreeCut& cut, const ShadingPoint& representant, uint32_t currentIteration) const {
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

    const Float t = currentIteration;
    const Float learningRate = 1 / (m_beta * std::pow(t, m_omega));

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
        cluster.q = (1 - learningRate) * cluster.q + learningRate * batchMean;
        cluster.secondMoment = (1 - learningRate) * cluster.secondMoment + learningRate * batchSqrMean;
        
        // update variance from accumulated statistics
        // V = E[X^2] - E[X]^2
        cluster.variance = std::max(cluster.secondMoment - Sqr(cluster.q), Float(0));

        cut.sumAccumulator[idx] = 0;
        cut.sumSquaredAccumulator[idx] = 0;
        cut.visitCountAccumulator[idx] = 0;

        sumVariance += cluster.variance;
    }
    sumVariance += MathEpsilon;

    constexpr Float initialCutSize = 4;
    uint32_t newCutSize = 0;
    //RNG rng()
    for (uint32_t idx = 0; idx < lastCutSize; ++idx) {
        OnlineCutData& cluster(cut.data[idx]);
        const LightcutsTreeNode* node = &m_tree.nodes[cluster.clusterIndex];

        if (node->isLeaf || newCutSize >= PBRT_LTC_MAX_CUT_SIZE - 1 || cluster.visitCount <= 1) {
            continue;
        }

        const Float relativeVariance = cluster.variance / sumVariance;
        const Float visitTerm = (cluster.visitCount - 1) / static_cast<Float>(cluster.visitCount);

        const Float relativeSize = lastCutSize / initialCutSize;
        const Float splitProb = relativeVariance * visitTerm / (1 + relativeSize * std::exp(-cluster.variance));
    }
}


std::string LTCLightSampler::ToString() const {
    return StringPrintf("[ LTCLightSampler nodes: %s ]", m_tree.nodes);
}


}
