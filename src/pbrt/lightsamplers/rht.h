// rht.h - RHTLightSampler class is Copyright(c) 2025-2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt and lightcuts.h source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_RHT_LIGHTSAMPLER_H
#define  PBRT_RHT_LIGHTSAMPLER_H

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

// Resampled Hierarchic Tree Light Sampler Definition
class RHTLightSampler {
  public:
    // Resampled Hierarchic Tree Light Sampler Public Methods
    RHTLightSampler(pstd::span<const Light> lights, Allocator alloc, Float gamma = 0.2);

    PBRT_CPU_GPU PBRT_NOINLINE
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Float u) const {
        Float pmf = 1;
        if (!m_infiniteLights.empty()) {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.leaves.size(), pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }
        
        if (m_tree.innerNodes.empty())
            return {};

        // Declare common variables for light BVH traversal
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;

        int nodeIndex = 0;

        return Sample(u);
    }

    PBRT_CPU_GPU
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Light light) const {
        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return InfiniteLightSimplePMF(m_infiniteLights, m_tree.leaves.size());;

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_tree.leaves.size() == 0 ? 0 : 1));
        
        if (m_tree.leaves.empty())
            return 0;

        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        uint32_t bitTrail = m_lightToBitTrail[light];

        int nodeIndex = 0;
        Float PsParent = 1;
        Float T = 1;
        
        const Float uSplit = HashFloat(seed);
        const ResampledTreeNode* node = &m_tree.innerNodes[nodeIndex];
        while (!node->isLeaf) {
            const uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

            const Float PsNode = std::min(node->bounds.SplitProbability(p, gamma), PsParent);
            const Float PsHatNode = std::max(MathEpsilon, 1 - PsNode);

            // Probability of splitting C_parent given that C has not been split
            const Float Sns = (PsParent - PsNode) / PsHatNode;
            const Float Pns = Sns + (1 - Sns) * T;

            const int child = bitTrail & 1;
            
            const ResampledTreeNode *children[2] = {&m_tree.innerNodes[childrenIndices[0]],
                                                    &m_tree.innerNodes[childrenIndices[1]]};
            const Float ci[2] = {children[0]->bounds.Importance(p, n),
                                 children[1]->bounds.Importance(p, n)};

            const Float sumImportance = ci[0] + ci[1];
            if (sumImportance == 0) {
                return 0;
            }

            Float weight[2] = {0};
            weight[0] = ci[0] / sumImportance;
            weight[1] = 1 - weight[0];

            T = Pns * weight[child];

            PsParent = PsNode;
            nodeIndex = childrenIndices[child];
            node = &m_tree.innerNodes[nodeIndex];

            bitTrail >>= 1;
        }

        return (1 - pInfinite) * (PsParent + (1 - PsParent) * T);
    }

    PBRT_CPU_GPU PBRT_NOINLINE
    pstd::optional<SampledLight> Sample(Float u) const {
        Float pmf = 1;
        {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.leaves.size(), pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        uint32_t index = std::min<uint32_t>(u * static_cast<Float>(m_tree.leaves.size()), m_tree.leaves.size() - 1);
        pmf /= m_tree.leaves.size();
        return SampledLight{m_tree.leaves[index].light, pmf};
    }

    PBRT_CPU_GPU PBRT_NOINLINE
    LightPMF PMF(Light light) const {
        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return InfiniteLightSimplePMF(m_infiniteLights, m_tree.leaves.size());;

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_tree.leaves.size() == 0 ? 0 : 1));
        
        if (m_tree.leaves.empty())
            return 0;

        Float pmf = 1 - pInfinite;
        return LightPMF(pmf / m_tree.leaves.size()); 
    }
    
#define PBRT_RHT_RESERVOIR_SET_H_SIZE 16
#define PBRT_RHT_RESAMPLED_CANDIDATES 1

    template <typename ScatterEval>
    PBRT_CPU_GPU PBRT_NOINLINE pstd::optional<SampledLd> SampleLd(const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const {
        Float pmf = 1;
        {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.leaves.size(), pmf, u);
            if (infiniteLightSample) {
                Light light = infiniteLightSample->light;
                DCHECK(light && infiniteLightSample->p != 0);
                pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
                if (!ls || !ls->L || ls->pdf == 0)
                    return {};

                Float lightPDF = infiniteLightSample->p * ls->pdf;
                ls->L *= infiniteLightSample->scale;
                
                Float scatterPDF = 0;
                SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));
                return SampledLd(f_hat * ls->L, ls->pLight, lightPDF, scatterPDF);
            }
        }

        const Point3f p = ctx.p();
        const Normal3f n = ctx.ns;
        
        HeuristicHReservoirSet heuristicHSampler(Hash(u, seed));
        CollectLightCandidates(heuristicHSampler, ctx, seed, u, HashFloat(seed), pmf);

        Point2f uLightOffset = GetR2SequenceOffset();
        WeightedReservoirSampler<SampledLd> heuristicFSampler(Hash(u, MixBits(seed)));
        for (int i = 0; i < heuristicHSampler.Size(); ++i) {
            // advance the sample unconditionally
            const Point2f uLightCurrent = uLight;
            uLight += uLightOffset;
            if (uLight.x >= 1) uLight.x -= 1;
            if (uLight.y >= 1) uLight.y -= 1;
        
            const StatelessWeightedReservoirSampler<LightCandidate>& reservoir(heuristicHSampler.GetReservoir(i));
            if (!reservoir.HasSample()) {
                continue;
            }
        
            const LightCandidate& sample(reservoir.GetSample());
            const Float hProb = reservoir.SampleProbability();
        
            Light light = m_tree.leaves[sample.lightIdx].light;
            DCHECK(light && sample.pmf != 0);
            pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLightCurrent, lambda, true);
            if (!ls || !ls->L || ls->pdf == 0)
                continue;
        
            const Float lightPDF = sample.pmf * ls->pdf;
            
            Float scatterPDF = 0;
            SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));

            f_hat *= ls->L;

            //SampledLd sLd(f_hat / hProb, ls->pLight, lightPDF, scatterPDF);
        
            // F(Si) = bsdf * (Li / pdfLight) * misW * hW(Li)
            const Float fWeight = f_hat.MaxComponentValue() / (lightPDF * hProb + scatterPDF);
            if (fWeight > 0) {
                heuristicFSampler.Add([&]{return SampledLd(f_hat / hProb, ls->pLight, lightPDF, scatterPDF);}, fWeight);
            }
        }
        
        if (!heuristicFSampler.HasSample()) {
            return {};
        }
        
        SampledLd resultLd(heuristicFSampler.GetSample());
        const Float fProb = heuristicFSampler.SampleProbability();
        
        resultLd.Ld /= fProb;
        return resultLd;
    }

    std::string ToString() const;

  private:
#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildLightTreeGPU(std::vector<RHTBuildContainer> &lights);
#endif

    using HeuristicHReservoirSet = WeightedReservoirSetSampler<LightCandidate, PBRT_RHT_RESERVOIR_SET_H_SIZE>;
    //using HeuristicHReservoirSet = RestirSampler<LightCandidate>;

    PBRT_CPU_GPU PBRT_NOINLINE
    void CollectLightCandidates(HeuristicHReservoirSet& reservoirSet, const LightSampleContext& ctx, uint32_t seed, Float u, Float uSplit, Float pmf) const;


    // Resampled Hierarchic Tree Light Sampler Private Members
    ResampledTree m_tree;
    pstd::vector<Light> m_infiniteLights;
    HashMap<Light, uint32_t> m_lightToBitTrail;
    Float gamma;
};

}
#endif // PBRT_RHT_LIGHTSAMPLER_H
