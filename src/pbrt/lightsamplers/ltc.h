// ltc.h - LTCLightSampler class is Copyright(c) 2025-2026 Richard Kvasnica.
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

    // Partition5DNode Public Members
    Float splitValue;
    struct {
        uint32_t splitAxis : 3;
        uint32_t rightChildOrLeafIndex : 29;
    };
};

//struct alignas(32) OnlineCutData {
//    Float q; // estimated importance
//    Float variance; // variance estimate
//    Float visitCount;
//    Float prefixSum;
//
//    uint32_t clusterIndex;
//    uint32_t bitTrail;
//    uint32_t depth;
//    uint32_t _padding1;
//};

#define PBRT_LTC_MAX_CUT_SIZE 64

struct OnlineLightTreeCut {
    PBRT_CPU_GPU
    OnlineLightTreeCut() : cutSize(0), lastUpdateIteration(0) {}

    //OnlineCutData data[PBRT_LTC_MAX_CUT_SIZE];
    Float q[PBRT_LTC_MAX_CUT_SIZE];
    Float variance[PBRT_LTC_MAX_CUT_SIZE];
    Float visitCount[PBRT_LTC_MAX_CUT_SIZE];
    Float prefixSum[PBRT_LTC_MAX_CUT_SIZE];

    uint32_t clusterIndex[PBRT_LTC_MAX_CUT_SIZE];
    uint32_t bitTrail[PBRT_LTC_MAX_CUT_SIZE];
    uint32_t depth[PBRT_LTC_MAX_CUT_SIZE];

    AtomicFloat sumAccumulator[PBRT_LTC_MAX_CUT_SIZE];
    AtomicInt<uint32_t> visitCountAccumulator[PBRT_LTC_MAX_CUT_SIZE];
    uint32_t cutSize;
    uint32_t lastUpdateIteration;
    uint32_t currentIteration;
};

struct PartitionTree {
    explicit PartitionTree(Allocator alloc);
    ~PartitionTree();

    PBRT_CPU_GPU
    OnlineLightTreeCut &Leaf(size_t idx) { return *leaves[idx]; }

    void EmplaceLeaf();

    Allocator alloc;
    pstd::vector<OnlineLightTreeCut*> leaves;
    pstd::vector<PartitionTreeNode> innerNodes;
    pstd::vector<ShadingPoint> representantPoints;
    Vector3f sceneExtent;
};

// Learning To Cluster Lightsampler Definition
class LTCLightSampler {
  public:
    // Learning To Cluster Light Sampler Public Methods
    LTCLightSampler(pstd::span<const Light> lights, Allocator alloc, Float beta = 2, Float omega = Float(6)/7, Float gamma = 128);

    void SetupScenePartitions(pstd::span<ShadingPoint> shadingPoints, const Bounds3f& sceneBounds);

    PBRT_CPU_GPU PBRT_NOINLINE
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Float u) const {
        Float pmf = 1;
        if (!m_infiniteLights.empty()) {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.lights.size(), pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        // Traverse light BVH to sample light
        if (m_tree.nodes.empty())
            return {};

        const Point3f p = ctx.p();
        const Vector3f wo = ctx.wo;
        const Normal3f n = ctx.ns;

        BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
        Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));

        uint32_t nodeIndex = 0;
        uint32_t lightSamplerHint = std::numeric_limits<uint32_t>::max();
        constexpr uint32_t clusterIndexPowerTwoCapacity = 8;
        Float clusterSelectionProb = 1;
        if (!m_partitions.leaves.empty() && n != Normal3f(0, 0, 0)) {
            Vector3f nv(n);
            UniformDiskVector diskVector(nv);
            uint32_t partitionIdx = GetPartitionIndex(p, diskVector);
            lightSamplerHint = partitionIdx << clusterIndexPowerTwoCapacity;

            const OnlineLightTreeCut &cut(*m_partitions.leaves[partitionIdx]);

            uint32_t offset = 0;
            const uint32_t cutSize = cut.cutSize;
            const Float weightSum = cut.prefixSum[cutSize - 1];
            Float up = u * weightSum;
            if (up >= weightSum) {
                up = NextFloatDown(up);
                offset = cutSize - 1;
            } else {
                uint32_t high = cutSize - 1;

                // binary search of the target cluster chosen by a random variable u
                while (offset < high) {
                    uint32_t mid = offset + ((high - offset) >> 1);
                    const Float prefixSum = cut.prefixSum[mid];

                    offset = prefixSum <= up ? mid + 1 : offset;
                    high = prefixSum > up ? mid : high;
                }
                DCHECK_LT(offset, cutSize);
            }

            //const OnlineCutData clusterData = cut.data[offset];
            const Float importance = cut.q[offset];
            const Float prevPrefix = (offset > 0) ? cut.prefixSum[offset - 1] : 0;
            clusterSelectionProb = importance / weightSum;
            u = std::min((up - prevPrefix) / importance, OneMinusEpsilon);
            lightSamplerHint |= offset;
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

            if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0]->compactLightBounds, children[1]->compactLightBounds, m_tree.allLightBounds)) {
                AccumulateContribution(0, lightSamplerHint);
                return {};
            }

            Float weights[2] = {0};
            const Float sqrt0 = errBounds[0];
            const Float sqrt1 = errBounds[1];
            //const Float sqrt0 = SafeSqrt(errBounds[0]) / children[0]->lightCountSqrt;
            //const Float sqrt1 = SafeSqrt(errBounds[1]) / children[1]->lightCountSqrt;
            const Float sumSqrt = sqrt0 + sqrt1;

            if (sumSqrt > 0) {
                weights[0] = std::min(OneMinusEpsilon, sqrt0 / sumSqrt);
            } else {
                weights[0] = 0.5f;
            }
            weights[1] = 1 - weights[0];
            
            // Randomly sample light BVH child node
            Float nodePMF;
            int child = SampleDiscrete(weights, u, &nodePMF, &u);
            clusterPdf *= nodePMF;

            nodeIndex = childrenIndices[child];
            node = &m_tree.nodes[nodeIndex];

            const Float scrambleOffset = HashFloat(nodeIndex, seed);
            u += scrambleOffset;
            if (u >= 1) u -= 1;
        }

        pmf *= clusterPdf;

        return SampledLight(m_tree.lights[node->childOrLightIndex], pmf, lightSamplerHint, clusterSelectionProb);
    }

    PBRT_CPU_GPU PBRT_NOINLINE
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t /*seed*/, Light light) const {
        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return InfiniteLightSimplePMF(m_infiniteLights, m_tree.nodes.size());;

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_tree.nodes.size() == 0 ? 0 : 1));

        // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = m_lightToBitTrail[light];
        const Point3f p = ctx.p();
        const Normal3f n = ctx.ns;
        const Vector3f wo = ctx.wo;
        
        Float pmf = 1 - pInfinite;

        BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
        Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));
        
        uint32_t nodeIndex = 0;
        if (!m_partitions.leaves.empty() && n != Normal3f(0, 0, 0)) {
            Vector3f nv(n);
            UniformDiskVector diskVector(nv);
            uint32_t partitionIdx = GetPartitionIndex(p, diskVector);
            const OnlineLightTreeCut &cut(*m_partitions.leaves[partitionIdx]);

            const uint32_t cutSize = cut.cutSize;
            const Float weightSum = cut.prefixSum[cutSize - 1];
            uint32_t foundIndex = cutSize;
            for (uint32_t i = 0; i < cutSize; ++i) {
                const uint32_t currentDepth = cut.depth[i];
                const uint32_t currentBitTrail = cut.bitTrail[i];

                const uint32_t bitMask = (1u << currentDepth) - 1;
                const uint32_t masked = bitTrail & bitMask;

                foundIndex = (masked == currentBitTrail) ? i : foundIndex;
            }

            DCHECK_LT(foundIndex, cutSize);

            bitTrail >>= cut.depth[foundIndex];
            nodeIndex = cut.clusterIndex[foundIndex];

            pmf *= cut.q[foundIndex] / weightSum;
        }

        const LTCTreeNode *node = &m_tree.nodes[nodeIndex];

        // Compute light's cluster pdf by walking down tree nodes to the light
        while (!node->isLeaf) {
            // Compute child importances and update PMF for current node
            uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

            const LTCTreeNode *children[2] = {&m_tree.nodes[childrenIndices[0]],
                                                    &m_tree.nodes[childrenIndices[1]]};

            Float errBounds[2] = {1, 1};
            
            if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0]->compactLightBounds, children[1]->compactLightBounds, m_tree.allLightBounds)) {
                return 0;
            }

            Float weights[2] = {0};
            const Float sqrt0 = errBounds[0];
            const Float sqrt1 = errBounds[1];
            //const Float sqrt0 = SafeSqrt(errBounds[0]) / children[0]->lightCountSqrt;
            //const Float sqrt1 = SafeSqrt(errBounds[1]) / children[1]->lightCountSqrt;
            const Float sumSqrt = sqrt0 + sqrt1;

            if (sumSqrt > 0) {
                weights[0] = std::min(OneMinusEpsilon, sqrt0 / sumSqrt);
            } else {
                weights[0] = 0.5f;
            }
            weights[1] = 1 - weights[0];

            const int child = bitTrail & 1;
            if (weights[child] == 0) {
                DCHECK_GT(weights[child], 0);
                return 0;
            }

            pmf *= weights[child];

            // Use _bitTrail_ to find next node index and update its value
            nodeIndex = childrenIndices[child];
            node = children[child];

            bitTrail >>= 1;
        }

        DCHECK_EQ(light, m_tree.lights[node->childOrLightIndex]);
        return LightPMF(pmf);
    }

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
        SampledSpectrum Ld = f_hat * ls->L;

        samples.Add(SampledLd(Ld, light, ls->pLight, lightPDF, scatterPDF, sampledLight->hint, sampledLight->pLearning));
    }

    std::string ToString() const;

    void Update(uint32_t currentIteration);

    PBRT_CPU_GPU
    void AccumulateContribution(Float contribution, const uint32_t lightSamplerHint) const {
        if (m_partitions.leaves.empty() || lightSamplerHint == std::numeric_limits<uint32_t>::max()) {
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
    bool buildLightTreeGPU(std::vector<LightBVHBuildContainer> &lights);
#endif

    uint32_t BuildPartitionTree(pstd::span<ShadingPoint>& items, int start, int end);

    PBRT_CPU_GPU
    uint32_t GetPartitionIndex(const Point3f& p, const UniformDiskVector& oct) const;
     
    // Learning To Cluster Light Sampler Private Members
    LTCLightTree m_tree;
    PartitionTree m_partitions;
    pstd::vector<Light> m_infiniteLights;
    HashMap<Light, uint32_t> m_lightToBitTrail;
    Float m_beta;
    Float m_omega;
    Float m_gamma;
};

}

#endif // PBRT_LTC_LIGHTSAMPLER_H
