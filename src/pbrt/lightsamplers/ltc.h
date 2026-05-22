// Copyright(c) 2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_LTC_LIGHTSAMPLER_H
#define  PBRT_LTC_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/bsdf.h>

#include <pbrt/util/manylights.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/math.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/containers.h>

namespace pbrt {

struct ShadingPoint;

/// @brief Node in the 5D partition tree over shading position and normal.
/// Interior nodes store one split axis/value and implicit left child
/// (`nodeIndex + 1`), while right child is encoded in
/// `rightChildOrLeafIndex`. One leaf node holds a light tree cut for a scene partition.
struct alignas(8) PartitionTreeNode {
    PartitionTreeNode() = default;

    PartitionTreeNode(Float splitValue, uint32_t splitAxis, uint32_t rightChildOrLeafIndex)
        : splitValue(splitValue), splitAxis(splitAxis), rightChildOrLeafIndex(rightChildOrLeafIndex) {}

    static PartitionTreeNode MakeLeaf(uint32_t leafIdx) {
        return PartitionTreeNode(std::numeric_limits<Float>::max(), 7, leafIdx);
    }

    static PartitionTreeNode MakeInterior(uint32_t splitAxis, Float splitValue, uint32_t childIdx) {
        return PartitionTreeNode(splitValue, splitAxis, childIdx);
    }

    PBRT_CPU_GPU
    bool IsLeaf() const { return splitAxis == 7; }

    std::string ToString() const;

    /// Split value for the chosen axis.
    Float splitValue;
    struct {
        /// Split axis in [0,4] for interior nodes; 7 for leaves.
        uint32_t splitAxis : 3;
        /// Right-child index (interior) or leaf index (leaf).
        uint32_t rightChildOrLeafIndex : 29;
    };
};

#ifndef PBRT_LTC_MAX_CUT_SIZE
#define PBRT_LTC_MAX_CUT_SIZE 64
#endif

/// @brief Per-partition online cut state used by LTC learning updates.
/// Stores estimated cluster importances and accumulators for contributions
/// observed during the current render wave.
struct OnlineLightTreeCut {
    PBRT_CPU_GPU
    OnlineLightTreeCut() : cutSize(0), lastUpdateIteration(0) {}

    Float q[PBRT_LTC_MAX_CUT_SIZE];               ///< Estimated importance of each selected cluster.
    Float variance[PBRT_LTC_MAX_CUT_SIZE];        ///< Online variance estimate per cluster.
    Float visitCount[PBRT_LTC_MAX_CUT_SIZE];      ///< Running effective sample count per cluster.
    Float prefixSum[PBRT_LTC_MAX_CUT_SIZE];       ///< Prefix sums over `q` used for inverse-CDF sampling.

    uint32_t clusterIndex[PBRT_LTC_MAX_CUT_SIZE]; ///< Light-tree node index of each cut cluster.
    uint32_t bitTrail[PBRT_LTC_MAX_CUT_SIZE];     ///< Encoded bitTrail from tree root to a cluster.
    uint32_t depth[PBRT_LTC_MAX_CUT_SIZE];        ///< Depth of each cluster root in the light tree.

    AtomicFloat sumAccumulator[PBRT_LTC_MAX_CUT_SIZE];                ///< Wave-local accumulated scalar contribution per cluster.
    AtomicInt<uint32_t> visitCountAccumulator[PBRT_LTC_MAX_CUT_SIZE]; ///< Wave-local visit count per cluster.
    uint32_t cutSize;                                                 ///< Number of active clusters in the cut.
    uint32_t lastUpdateIteration;                                     ///< Last learning iteration that updated this cut.
    uint32_t currentIteration;                                        ///< Current wave index being processed.
};

/// @brief Partition-tree container and per-leaf online cuts.
struct PartitionTree {
    explicit PartitionTree(Allocator alloc);
    ~PartitionTree();

    PBRT_CPU_GPU
    OnlineLightTreeCut &Leaf(size_t idx) { return *leaves[idx]; }

    /// @brief Appends one empty cut leaf to `leaves`.
    void EmplaceLeaf();

    Allocator alloc;
    pstd::vector<OnlineLightTreeCut*> leaves;      ///< One online cut per partition-tree leaf.
    pstd::vector<PartitionTreeNode> innerNodes;    ///< Flattened partition-tree in implicit left-child layout.
    pstd::vector<ShadingPoint> representantPoints; ///< Representative shading point for each partition.
    Vector3f sceneExtent;                          ///< Scene extent used to normalize spatial split dimensions.
};

/// @brief Learning To Cluster (LTC) light sampler implementation based on paper from Wang et al. 2021
/// Combines a light hierarchy with an online-updated cut per shading partition.
/// The first render wave collects shading points, builds 5D partitions, and
/// initializes partition cuts. Subsequent waves update cut importances from
/// measured direct-light contributions.
class LTCLightSampler {
  public:
    /// @brief Builds the LTC light hierarchy and initializes runtime state.
    /// @param lights Scene lights visible to this integrator.
    /// @param alloc Pbrt allocator used by hierarchy and partition data.
    /// @param beta Learning rate first parameter.
    /// @param omega Learning rate second parameter.
    /// @param gamma Parameter to determine the max iteration where the learning should stop.
    LTCLightSampler(pstd::span<const Light> lights, Allocator alloc, Float beta = 2, Float omega = Float(6)/7, Float gamma = 128);

    /// @brief Builds the shading partition tree and initializes per-leaf cuts.
    /// Called after the first render wave once shading samples are collected.
    /// @param shadingPoints First-wave shading points from the integrator.
    /// @param sceneBounds Scene bounds used to normalize spatial dimensions.
    void SetupScenePartitions(pstd::span<ShadingPoint> shadingPoints, const Bounds3f& sceneBounds);

    /// @brief Samples a light using partition-conditioned LTC traversal.
    /// The returned hint encodes partition/cluster indices for later
    /// `AccumulateContribution()` calls.
    PBRT_CPU_GPU PBRT_NOINLINE
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Float u) const {
        Float pmf = 1;
        if (!m_infiniteLights.empty()) {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.lights.size(), pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        if (m_tree.nodes.empty())
            return {};

        const Point3f p = ctx.p();
        const Vector3f wo = ctx.wo;
        const Normal3f n = ctx.ns;

        BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
        Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));

        // root start
        uint32_t nodeIndex = 0;

        // Encodes `(partitionIndex << 8) | cutOffset`, consumed later by
        // integrators for `AccumulateContribution()`.
        uint32_t lightSamplerHint = std::numeric_limits<uint32_t>::max();
        constexpr uint32_t clusterIndexPowerTwoCapacity = 8;

        // Probability of selecting the partition cut cluster; multiplied into pmf.
        Float clusterSelectionProb = 1;
        if (!m_partitions.leaves.empty() && n != Normal3f(0, 0, 0)) {
            // Map shading context to a single 5D partition.
            Vector3f nv(n);
            UniformDiskVector diskVector(nv);
            uint32_t partitionIdx = GetPartitionIndex(p, diskVector);
            lightSamplerHint = partitionIdx << clusterIndexPowerTwoCapacity;

            const OnlineLightTreeCut &cut(*m_partitions.leaves[partitionIdx]);

            // Sample one cut entry by inverting the prefix-sum CDF in `cut.prefixSum`.
            uint32_t offset = 0;
            const uint32_t cutSize = cut.cutSize;
            const Float weightSum = cut.prefixSum[cutSize - 1];
            Float up = u * weightSum;
            if (up >= weightSum) {
                up = NextFloatDown(up);
                offset = cutSize - 1;
            } else {
                uint32_t high = cutSize - 1;

                // Binary search first prefix > up.
                while (offset < high) {
                    uint32_t mid = offset + ((high - offset) >> 1);
                    const Float prefixSum = cut.prefixSum[mid];

                    offset = prefixSum <= up ? mid + 1 : offset;
                    high = prefixSum > up ? mid : high;
                }
                DCHECK_LT(offset, cutSize);
            }

            // Extract selected cut entry and remap u to the selected entry interval.
            const Float importance = cut.q[offset];
            const Float prevPrefix = (offset > 0) ? cut.prefixSum[offset - 1] : 0;
            clusterSelectionProb = importance / weightSum;
            u = std::min((up - prevPrefix) / importance, OneMinusEpsilon);

            // Encode selected cut entry into the hint for online learning updates.
            lightSamplerHint |= offset;

            // Continue traversal from the selected cluster root instead of the tree root.
            nodeIndex = cut.clusterIndex[offset];
        }
        
        const LTCTreeNode* node = &m_tree.nodes[nodeIndex];

        Float clusterPdf = 1;
        pmf *= clusterSelectionProb;

        while (!node->isLeaf) {
            uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

            const LTCTreeNode *children[2] = {&m_tree.nodes[childrenIndices[0]],
                                              &m_tree.nodes[childrenIndices[1]]};
            
            const Float nodeIntensities[2] = {children[0]->compactLightBounds.PhiOrI(),
                                              children[1]->compactLightBounds.PhiOrI()};
            Float errBounds[2] = {1, 1};

            // Compute LTC bound for each child; if it fails, treat as zero signal.
            if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0]->compactLightBounds, children[1]->compactLightBounds, m_tree.allLightBounds)) {
                AccumulateContribution(0, lightSamplerHint);
                return {};
            }

            // LTC traversal weights are proportional to sqrt(bound) * uniform-prob term.
            Float weights[2] = {0};
            //const Float sqrt0 = SafeSqrt(errBounds[0]) * children[0]->pUniformSqrt;
            //const Float sqrt1 = SafeSqrt(errBounds[1]) * children[1]->pUniformSqrt;
            const Float sqrt0 = errBounds[0];
            const Float sqrt1 = errBounds[1];
            const Float sumSqrt = sqrt0 + sqrt1;

            if (sumSqrt > 0) {
                weights[0] = std::min(OneMinusEpsilon, sqrt0 / sumSqrt);
            } else {
                weights[0] = 0.5f;
            }
            weights[1] = 1 - weights[0];
            
            // Sample one child and fold its probability into clusterPdf.
            Float nodePMF;
            int child = SampleDiscrete(weights, u, &nodePMF, &u);
            clusterPdf *= nodePMF;

            nodeIndex = childrenIndices[child];
            node = &m_tree.nodes[nodeIndex];

            // Hash scramble `u` between levels to reduce structural correlation.
            const Float scrambleOffset = HashFloat(nodeIndex, seed);
            u += scrambleOffset;
            if (u >= 1) u -= 1;
        }

        pmf *= clusterPdf;

        // `clusterSelectionProb` is stored as `pLearning` for integrator-side
        // contribution normalization during online LTC updates.
        return SampledLight(m_tree.lights[node->childOrLightIndex], pmf, lightSamplerHint, clusterSelectionProb);
    }

    /// @brief Evaluates PMF for a specific light for specific shading point context.
    PBRT_CPU_GPU PBRT_NOINLINE
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t /*seed*/, Light light) const {
        // Infinite lights are not represented in the finite LTC tree.
        if (!m_lightToBitTrail.HasKey(light))
            return InfiniteLightSimplePMF(m_infiniteLights, m_tree.nodes.size());;

        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_tree.nodes.size() == 0 ? 0 : 1));

        uint32_t bitTrail = m_lightToBitTrail[light];

        const Point3f p = ctx.p();
        const Normal3f n = ctx.ns;
        const Vector3f wo = ctx.wo;
        
        Float pmf = 1 - pInfinite;

        BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
        Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));
        
        uint32_t nodeIndex = 0;
        if (!m_partitions.leaves.empty() && n != Normal3f(0, 0, 0)) {
            // Identify the same partition cut that `Sample(...)` would use.
            Vector3f nv(n);
            UniformDiskVector diskVector(nv);
            uint32_t partitionIdx = GetPartitionIndex(p, diskVector);
            const OnlineLightTreeCut &cut(*m_partitions.leaves[partitionIdx]);

            const uint32_t cutSize = cut.cutSize;
            const Float weightSum = cut.prefixSum[cutSize - 1];
            uint32_t foundIndex = cutSize;
            // Find cut entry whose bit prefix matches this light's bit trail.
            for (uint32_t i = 0; i < cutSize; ++i) {
                const uint32_t currentDepth = cut.depth[i];
                const uint32_t currentBitTrail = cut.bitTrail[i];

                const uint32_t bitMask = (1u << currentDepth) - 1;
                const uint32_t masked = bitTrail & bitMask;

                foundIndex = (masked == currentBitTrail) ? i : foundIndex;
            }

            DCHECK_LT(foundIndex, cutSize);

            // Remove matched prefix bits; remaining bits drive subtree replay.
            bitTrail >>= cut.depth[foundIndex];

            // Start replay from the matched cluster root.
            nodeIndex = cut.clusterIndex[foundIndex];

            // Multiply probability of selecting this cut cluster.
            pmf *= cut.q[foundIndex] / weightSum;
        }

        const LTCTreeNode *node = &m_tree.nodes[nodeIndex];

        // Replay the hierarchical child decisions from bitTrail until leaf.
        while (!node->isLeaf) {
            uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

            const LTCTreeNode *children[2] = {&m_tree.nodes[childrenIndices[0]],
                                                    &m_tree.nodes[childrenIndices[1]]};

            Float errBounds[2] = {1, 1};
            
            // PMF replay uses the same local weighting rule as sampling.
            if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0]->compactLightBounds, children[1]->compactLightBounds, m_tree.allLightBounds)) {
                return 0;
            }

            Float weights[2] = {0};
            //const Float sqrt0 = SafeSqrt(errBounds[0]) * children[0]->pUniformSqrt;
            //const Float sqrt1 = SafeSqrt(errBounds[1]) * children[1]->pUniformSqrt;
            const Float sqrt0 = errBounds[0];
            const Float sqrt1 = errBounds[1];
            const Float sumSqrt = sqrt0 + sqrt1;

            if (sumSqrt > 0) {
                weights[0] = std::min(OneMinusEpsilon, sqrt0 / sumSqrt);
            } else {
                weights[0] = 0.5f;
            }
            weights[1] = 1 - weights[0];

            // Next traversal branch comes from current low bit of bitTrail.
            const int child = bitTrail & 1;
            if (weights[child] == 0) {
                DCHECK_GT(weights[child], 0);
                return 0;
            }

            // Multiply per-node branch probability.
            pmf *= weights[child];

            // Advance to selected child and consume one decision bit.
            nodeIndex = childrenIndices[child];
            node = children[child];

            bitTrail >>= 1;
        }

        DCHECK_EQ(light, m_tree.lights[node->childOrLightIndex]);
        return LightPMF(pmf);
    }

    /// @brief Context-free fallback light sampling path.
    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        Float pmf = 1;
        {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.lights.size(), pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        uint32_t index = std::min<uint32_t>(u * static_cast<Float>(m_tree.lights.size()), m_tree.lights.size() - 1);
        pmf /= m_tree.lights.size();
        return SampledLight{m_tree.lights[index], pmf};
    }

    /// @brief PMF for context-free fallback sampling.
    PBRT_CPU_GPU
    LightPMF PMF(Light light) const {
        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return InfiniteLightSimplePMF(m_infiniteLights, m_tree.nodes.size());;

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_tree.nodes.size() == 0 ? 0 : 1));
        
        if (m_tree.lights.empty())
            return 0;

        Float pmf = 1 - pInfinite;
        return LightPMF(pmf / m_tree.lights.size()); 
    }
    
    /// @brief Generates direct-light samples for integrator MIS evaluation.
    /// Also propagates LTC learning metadata (`hint`, `pLearning`) in
    /// `SampledLd`.
    template <int NSamples, typename ScatterEval>
    PBRT_CPU_GPU PBRT_NOINLINE void SampleLd(CountedArray<SampledLd, NSamples>& samples, const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const {
        pstd::optional<SampledLight> sampledLight = Sample(ctx, bsdf, seed, u);
        if (!sampledLight) {
            return;
        }

        Light light = sampledLight->light;
        DCHECK(light && sampledLight->p != 0);
        pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
        if (!ls || !ls->L || ls->pdf == 0) {
            AccumulateContribution(0, sampledLight->hint);
            return;
        }

        Float lightPDF = sampledLight->p * ls->pdf;
        Float scatterPDF = 0;
        SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));
        SampledSpectrum Ld = ClampZero(f_hat * ls->L);

        samples.Add(SampledLd(Ld, light, ls->pLight, lightPDF, scatterPDF, sampledLight->hint, sampledLight->pLearning));
    }

    std::string ToString() const;

    /// @brief Applies one online-learning update for all partition cuts.
    /// @param currentIteration Current render-wave index.
    void Update(uint32_t currentIteration);

    /// @brief Accumulates scalar contribution for the sampled cut cluster.
    /// @param contribution Non-negative scalar proxy of direct illumination.
    /// @param lightSamplerHint Encoded partition and cluster index from `Sample`.
    PBRT_CPU_GPU
    void AccumulateContribution(Float contribution, const uint32_t lightSamplerHint) const {
        if (m_partitions.leaves.empty() || lightSamplerHint == std::numeric_limits<uint32_t>::max()) {
            return;
        }
        if (!IsFinite(contribution)) {
            return;
        }

        constexpr uint32_t clusterIndexPowerTwoCapacity = 8;
        constexpr uint32_t clusterIndexMask = (1 << clusterIndexPowerTwoCapacity) - 1;

        const uint32_t clusterIndex = lightSamplerHint & clusterIndexMask;
        const uint32_t partitionIndex = lightSamplerHint >> clusterIndexPowerTwoCapacity;
        DCHECK_LT(partitionIndex, static_cast<uint32_t>(m_partitions.leaves.size()));

        OnlineLightTreeCut* cut(m_partitions.leaves[partitionIndex]);

        cut->sumAccumulator[clusterIndex].Add(contribution);
        cut->visitCountAccumulator[clusterIndex].Add(1);
    }

  private:
    // Learning To Cluster Light Sampler Private Methods
#ifdef PBRT_BUILD_GPU_RENDERER
    /// @brief Attempts GPU-based construction of the LTC light tree.
    bool buildLightTreeGPU(std::vector<LightBVHBuildContainer> &lights);
#endif

    /// @brief Recursively builds the 5D partition tree.
    /// @param items Mutable shading-point array to split in place.
    /// @param start Inclusive range start.
    /// @param end Exclusive range end.
    /// @return Index of the created partition-tree node.
    uint32_t BuildPartitionTree(pstd::span<ShadingPoint>& items, int start, int end);

    /// @brief Finds partition index for a shading context descriptor.
    /// @param p Shading position.
    /// @param oct Encoded shading normal direction.
    /// @return Index into `m_partitions.leaves`.
    PBRT_CPU_GPU
    uint32_t GetPartitionIndex(const Point3f& p, const UniformDiskVector& oct) const;
     
    // Learning To Cluster Light Sampler Private Members
    LTCLightTree m_tree;                         ///< Finite-light hierarchy and node bounds.
    PartitionTree m_partitions;                  ///< 5D shading partitions with per-leaf online cuts.
    pstd::vector<Light> m_infiniteLights;        ///< Infinite/environment lights sampled separately.
    HashMap<Light, uint32_t> m_lightToBitTrail;  ///< BitTrail path per each light.
    Float m_beta;                                ///< LTC update parameter beta.
    Float m_omega;                               ///< LTC update parameter omega.
    Float m_gamma;                               ///< LTC update parameter gamma.
};

}

#endif // PBRT_LTC_LIGHTSAMPLER_H
