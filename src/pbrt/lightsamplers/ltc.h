// ltc.h - LTCLightSampler class is Copyright(c) 2025-2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt and lightcuts.h source code is licensed under the Apache License, Version 2.0.
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

    PBRT_CPU_GPU static PartitionTreeNode MakeLeaf(uint32_t leafIdx) {
        return PartitionTreeNode{std::numeric_limits<Float>::max(), {7, leafIdx}};
    }

    PBRT_CPU_GPU static PartitionTreeNode MakeInterior(uint32_t splitAxis, Float splitValue, uint32_t childIdx) {
        return PartitionTreeNode{splitValue, {splitAxis, childIdx}};
    }

    bool IsLeaf() const { return splitAxis == 7; }

    std::string ToString() const;

    // Partition5DNode Public Members
    Float splitValue;
    struct {
        uint32_t splitAxis : 3;
        uint32_t rightChildOrLeafIndex : 29;
    };
};

struct alignas(32) OnlineCutData {
    Float q; // estimated importance
    Float variance; // variance estimate
    Float secondMoment; // for updating variance online
    Float _padding;

    uint32_t clusterIndex;
    uint32_t bitTrail;
    uint32_t depth;
    uint32_t visitCount;
};

#define PBRT_LTC_MAX_CUT_SIZE 64

struct OnlineLightTreeCut {
    OnlineCutData data[PBRT_LTC_MAX_CUT_SIZE];
    AtomicFloat sumAccumulator[PBRT_LTC_MAX_CUT_SIZE];
    AtomicFloat sumSquaredAccumulator[PBRT_LTC_MAX_CUT_SIZE];
    AtomicInt<uint32_t> visitCountAccumulator[PBRT_LTC_MAX_CUT_SIZE];
    uint32_t cutSize;
    uint32_t lastUpdateIteration;
};

struct PartitionTree {
    PartitionTree(Allocator alloc);
    pstd::vector<OnlineLightTreeCut> leaves;
    pstd::vector<PartitionTreeNode> innerNodes;
    pstd::vector<ShadingPoint> representantPoints;
    Vector3f sceneExtent;
};

// Learning To Cluster Lightsampler Definition
class LTCLightSampler {
  public:
    // Learning To Cluster Light Sampler Public Methods
    LTCLightSampler(pstd::span<const Light> lights, Allocator alloc, Float beta = 4, Float omega = Float(6)/7, Float gamma = 128);

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

        Point3f p = ctx.p();
        Vector3f wo = ctx.wo;
        Normal3f n = ctx.ns;

        BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
        Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));

        int nodeIndex = 0;
        const LightcutsTreeNode* node = &m_tree.nodes[nodeIndex];

        while (!node->isLeaf) {
            uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

            const LightcutsTreeNode *children[2] = {&m_tree.nodes[childrenIndices[0]],
                                                    &m_tree.nodes[childrenIndices[1]]};
            
            const Float nodeIntensities[2] = {children[0]->compactLightBounds.PhiOrI(),
                                              children[1]->compactLightBounds.PhiOrI()};
            Float errBounds[2] = {1, 1};

            if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0], children[1], m_tree.allLightBounds)) {
                return {};
            }

            Float weights[2] = {0};
            weights[0] = std::min(OneMinusEpsilon, errBounds[0] / (errBounds[0] + errBounds[1]));
            weights[1] = 1 - weights[0];
            
            // Randomly sample light BVH child node
            Float nodePMF;
            int child = SampleDiscrete(weights, u, &nodePMF, &u);
            pmf *= nodePMF;

            nodeIndex = childrenIndices[child];
            node = &m_tree.nodes[nodeIndex];

            const Float scrambleOffset = HashFloat(nodeIndex, seed);
            u += scrambleOffset;
            if (u >= 1) u -= 1;
        }

        return SampledLight(m_tree.lights[node->childOrLightIndex], pmf);
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
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        Vector3f wo = ctx.wo;
        
        Float pmf = 1 - pInfinite;
        int nodeIndex = 0;

        BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
        Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));

        const LightcutsTreeNode *node = &m_tree.nodes[nodeIndex];

        // Compute light's PMF by walking down tree nodes to the light
        while (!node->isLeaf) {
            // Compute child importances and update PMF for current node
            uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

            const LightcutsTreeNode *children[2] = {&m_tree.nodes[childrenIndices[0]],
                                                    &m_tree.nodes[childrenIndices[1]]};

            Float errBounds[2] = {1, 1};
            
            if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0], children[1], m_tree.allLightBounds)) {
                return 0;
            }

            Float weights[2] = {0};
            weights[0] = std::min(OneMinusEpsilon, errBounds[0] / (errBounds[0] + errBounds[1]));
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
        if (!ls || !ls->L || ls->pdf == 0)
            return;

        Float lightPDF = sampledLight->p * ls->pdf;
        ls->L *= sampledLight->scale;
        
        Float scatterPDF = 0;
        SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));

        samples.Add(SampledLd(f_hat * ls->L, light, ls->pLight, lightPDF, scatterPDF));
    }

    std::string ToString() const;

    PBRT_CPU_GPU
    void Update(uint32_t currentIteration);
  private:
    // Learning To Cluster Light Sampler Private Methods
#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights, Float& u);
#endif

    uint32_t BuildPartitionTree(pstd::span<ShadingPoint>& items, int start, int end);

    PBRT_CPU_GPU
    void MakeInitialTreeCut(OnlineLightTreeCut& cut, const ShadingPoint& representant) const;

    PBRT_CPU_GPU
    void ApplyIterationUpdate(OnlineLightTreeCut& cut, const ShadingPoint& representant, uint32_t currentIteration) const;

    // Learning To Cluster Light Sampler Private Members
    LightcutsTree m_tree;
    PartitionTree m_partitions;
    pstd::vector<Light> m_infiniteLights;
    HashMap<Light, uint32_t> m_lightToBitTrail;
    Float m_beta;
    Float m_omega;
    Float m_gamma;
};

}

#endif // PBRT_LTC_LIGHTSAMPLER_H
