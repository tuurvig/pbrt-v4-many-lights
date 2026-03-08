// slc.h - SLCLightSampler class is Copyright(c) 2025-2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt and lightcuts.h source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_SLC_LIGHTSAMPLER_H
#define  PBRT_SLC_LIGHTSAMPLER_H

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

// Stochastic Lightcuts Lightsampler Definition
class SLCLightSampler {
  public:
    // LightcutsLightSampler Public Methods
    SLCLightSampler(pstd::span<const Light> lights, Allocator alloc, Float threshold = 0.02);

    PBRT_CPU_GPU PBRT_NOINLINE
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Float u) const {
        Float pmf = 1;
        if (!m_infiniteLights.empty()) {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.nodes.size(), pmf, u);
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

        Float errBounds[PBRT_LIGHTCUTS_CUT_SIZE] = {0};
        CutData data[PBRT_LIGHTCUTS_CUT_SIZE];
        
        uint32_t bitTrail = 0; //dummy
        int cutSize = ComputeLightcutsTreeCut<PBRT_LIGHTCUTS_CUT_SIZE>(errBounds, data, bitTrail, ctx, m_tree.nodes, m_tree.allLightBounds, shadingFrame, bsdf, m_threshold);

        if (cutSize == 0) {
            return {};
        }

        WeightedReservoirSampler<CutData> reservoir(Hash(u));
        for (int i = 0, max = PBRT_LIGHTCUTS_CUT_SIZE; i < max; ++i) {
            Float errBound = errBounds[i];
            if (errBound <= 0) continue;

            reservoir.Add(data[i], errBound);
        }

        pmf *= reservoir.SampleProbability();

        constexpr uint32_t indexMask = std::numeric_limits<uint32_t>::max() >> 1;

        CutData nodeData = reservoir.GetSample();
        uint32_t nodeIndex = nodeData.nodeIndex & indexMask;
        const LightcutsTreeNode* node = &m_tree.nodes[nodeIndex];
        
        Float pmfRepresentant = 1;
        while (!node->isLeaf) {
            uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

            const LightcutsTreeNode *children[2] = {&m_tree.nodes[childrenIndices[0]],
                                                    &m_tree.nodes[childrenIndices[1]]};

            Float errBounds[2] = {1, 1};

            if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0], children[1], m_tree.allLightBounds, true)) {
                return {};
            }

            Float weights[2] = {0};
            weights[0] = std::min(OneMinusEpsilon, errBounds[0] / (errBounds[0] + errBounds[1]));
            weights[1] = 1 - weights[0];

            // Randomly sample light BVH child node
            Float nodePMF;
            int child = SampleDiscrete(weights, u, &nodePMF, &u);
            pmfRepresentant *= nodePMF;

            nodeIndex = childrenIndices[child];
            node = &m_tree.nodes[nodeIndex];

            const Float scrambleOffset = HashFloat(nodeIndex, seed);
            u += scrambleOffset;
            if (u >= 1) u -= 1;
        }

        return SampledLight(m_tree.lights[node->childOrLightIndex], pmf * pmfRepresentant);
    }

    PBRT_CPU_GPU PBRT_NOINLINE
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t /*seed*/, Light light) const {
        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return InfiniteLightSimplePMF(m_infiniteLights, m_tree.lights.size());;

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_tree.lights.size() == 0 ? 0 : 1));

        if (m_tree.lights.empty())
            return 0;

        // Compute cut exactly as in Sample().
        const Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));
        Float cutErrBounds[PBRT_LIGHTCUTS_CUT_SIZE] = {0};
        CutData cutData[PBRT_LIGHTCUTS_CUT_SIZE];

        uint32_t bitTrail = m_lightToBitTrail[light];
        int cutSize = ComputeLightcutsTreeCut<PBRT_LIGHTCUTS_CUT_SIZE>(cutErrBounds, cutData, bitTrail, ctx, m_tree.nodes, m_tree.allLightBounds, shadingFrame, bsdf, m_threshold);
        
        if (cutSize <= 0)
            return 0;

        constexpr uint32_t indexMask = std::numeric_limits<uint32_t>::max() >> 1;

        Float cutWeightSum = 0;
        uint32_t foundIndex = std::numeric_limits<uint32_t>::max();
        for (int i = 0; i < PBRT_LIGHTCUTS_CUT_SIZE; ++i) {
            Float errBound = cutErrBounds[i];
            if (errBound <= 0) continue;

            cutWeightSum += errBound;
            CutData& nodeData(cutData[i]);

            const bool onTrail = nodeData.nodeIndex >> 31;
            if (onTrail) {
                nodeData.nodeIndex &= indexMask;
                foundIndex = i;
            }
        }
        
        if (foundIndex == std::numeric_limits<uint32_t>::max())
            return 0;

        Float cutNodeProbability = cutErrBounds[foundIndex] / cutWeightSum;
        Float pmf = (1 - pInfinite) * cutNodeProbability;

        // Continue exactly with the same heuristic split probabilities as in Sample().
        const Point3f p = ctx.p();
        const Vector3f wo = ctx.wo;
        const Normal3f n = ctx.ns;

        uint32_t nodeIndex = cutData[foundIndex].nodeIndex;
        const LightcutsTreeNode* node = &m_tree.nodes[nodeIndex];
        
        while (!node->isLeaf) {
            const uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1),
                                                 node->childOrLightIndex};
            const LightcutsTreeNode* children[2] = {&m_tree.nodes[childrenIndices[0]],
                                                    &m_tree.nodes[childrenIndices[1]]};
            Float errBounds[2] = {1, 1};
            if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0], children[1], m_tree.allLightBounds, true)) {
                return 0;
            }

            Float weights[2] = {0};
            weights[0] = std::min(OneMinusEpsilon, errBounds[0] / (errBounds[0] + errBounds[1]));
            weights[1] = 1 - weights[0];

            const uint32_t child = bitTrail & 1;
            pmf *= weights[child];

            nodeIndex = childrenIndices[child];
            node = &m_tree.nodes[nodeIndex];
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
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = InfiniteLightSimplePMF(m_infiniteLights, m_tree.lights.size());
        
        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return pInfinite;

        if (m_tree.lights.empty())
            return 0;

        Float pmf = 1 - pInfinite;
        return pmf / m_tree.lights.size(); 
    }

    template <int NSamples, typename ScatterEval>
    PBRT_CPU_GPU PBRT_NOINLINE void SampleLd(CountedArray<SampledLd, NSamples>& samples, const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const {
        Float pmf = 1;
        {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.lights.size(), pmf, u);
            if (infiniteLightSample) {
                Light light = infiniteLightSample->light;
                DCHECK(light && infiniteLightSample->p != 0);
                pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;

                Float lightPDF = infiniteLightSample->p * ls->pdf;
                
                Float scatterPDF = 0;
                SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));

                samples.Add(SampledLd(f_hat * ls->L, light, ls->pLight, lightPDF, scatterPDF));
                return;
            }
        }

        const Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));

        Float errBounds[NSamples] = {0};
        CutData data[NSamples];

        uint32_t bitTrail = 0; // dummy
        int cutSize = ComputeLightcutsTreeCut<NSamples>(errBounds, data, bitTrail, ctx, m_tree.nodes, m_tree.allLightBounds, shadingFrame, bsdf, m_threshold, true);
        if (cutSize <= 0) {
            return;
        }

        Point3f p = ctx.p();
        Vector3f wo = ctx.wo;
        Normal3f n = ctx.ns;
        
        constexpr uint32_t indexMask = std::numeric_limits<uint32_t>::max() >> 1;

        Point2f uOffset = GetR2SequenceOffset();
        for (int i = 0; i < NSamples; ++i) {
            Float errBound = errBounds[i];
            if (errBound <= 0) continue;

            uLight += uOffset;
            if (uLight.x >= 1) uLight.x -= 1;
            if (uLight.y >= 1) uLight.y -= 1;

            CutData clusterData = data[i];

            Float pmfLight = pmf;

            uint32_t nodeIndex = clusterData.nodeIndex & indexMask;
            const LightcutsTreeNode* node = &m_tree.nodes[nodeIndex];

            Float pmfRepresentant = 1;
            bool failed = false;
            while (!node->isLeaf) {
                uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

                const LightcutsTreeNode *children[2] = {&m_tree.nodes[childrenIndices[0]],
                                                        &m_tree.nodes[childrenIndices[1]]};

                Float errBounds[2] = {1, 1};

                if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0], children[1], m_tree.allLightBounds, true)) {
                    failed = true;
                    break;
                }

                Float weights[2] = {0};
                weights[0] = std::min(OneMinusEpsilon, errBounds[0] / (errBounds[0] + errBounds[1]));
                weights[1] = 1 - weights[0];

                // Randomly sample light BVH child node
                Float nodePMF;
                int child = SampleDiscrete(weights, u, &nodePMF, &u);
                pmfRepresentant *= nodePMF;

                nodeIndex = childrenIndices[child];
                node = &m_tree.nodes[nodeIndex];

                const Float scrambleOffset = HashFloat(nodeIndex, seed);
                u += scrambleOffset;
                if (u >= 1) u -= 1;
            }

            if (failed) continue;

            pmfLight *= pmfRepresentant;

            Light light = m_tree.lights[node->childOrLightIndex];
            DCHECK(light && pmfLight != 0);
            pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
            if (!ls || !ls->L || ls->pdf == 0)
                continue;

            Float lightPDF = pmfLight * ls->pdf;

            Float scatterPDF = 0;
            SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));
            samples.Add(SampledLd(f_hat * ls->L, light, ls->pLight, lightPDF, scatterPDF));
        }
    }

    std::string ToString() const;

  private:
#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights, Float& u);
#endif

    // LightcutsLightSampler Private Members
    LightcutsTree m_tree;
    pstd::vector<Light> m_infiniteLights;
    HashMap<Light, uint32_t> m_lightToBitTrail;
    Float m_threshold;
};

}

#endif // PBRT_SLC_LIGHTSAMPLER_H
