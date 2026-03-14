// lightcuts.h - LightcutsLightSampler class is Copyright(c) 2025-2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt and lightcuts.h source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_LIGHTCUTS_LIGHTSAMPLER_H
#define  PBRT_LIGHTCUTS_LIGHTSAMPLER_H

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

// LightcutsLightSampler Definition
class LightcutsLightSampler {
    
public:
    // LightcutsLightSampler Public Methods
    LightcutsLightSampler(pstd::span<const Light> lights, Allocator alloc, Float threshold = 0.02);

    PBRT_CPU_GPU PBRT_NOINLINE pstd::optional<SampledLight> Sample(const LightSampleContext& ctx, const BSDF* bsdf, uint32_t seed, Float u) const {
        const size_t totalSize = m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size();
        Float pmf = 1;
        if (!m_infiniteLights.empty()) {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, totalSize, pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        Float weights[3] = {
            m_spotTree.lights.empty() ? 0 : m_spotTree.nodes[0].compactLightBounds.PhiOrI(),
            m_pointTree.lights.empty() ? 0 : m_pointTree.nodes[0].compactLightBounds.PhiOrI(),
            m_otherLightIntensities}; 

        Float groupPMF;
        int groupIdx = SampleDiscrete(weights, u, &groupPMF, &u);
        pmf *= groupPMF;

        if (groupIdx == 0) {
            return SampleLightTree(ctx, m_spotTree, false, bsdf, pmf, u);
        } else if (groupIdx == 1) {
            return SampleLightTree(ctx, m_pointTree, true, bsdf, pmf, u);
        }

        int index = std::min<int>(u * m_otherLights.size(), m_otherLights.size() - 1);
        pmf /= m_otherLights.size();
        return SampledLight{m_otherLights[index], pmf};
    }

    PBRT_CPU_GPU PBRT_NOINLINE LightPMF PMF(const LightSampleContext& ctx, const BSDF* bsdf, uint32_t /*seed*/, Light light) const {
        const size_t totalSize = m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size();

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = InfiniteLightSimplePMF(m_infiniteLights, totalSize);
        LightLocation loc = m_lightToLocation[light];

        // Handle infinite _light_ PMF computation
        if (loc.treeIdx == 2) {
            return pInfinite;
        }

        Float pmf = (1 - pInfinite);

        Float weights[3] = {m_spotTree.nodes[0].compactLightBounds.PhiOrI(), m_pointTree.nodes[0].compactLightBounds.PhiOrI(), m_otherLightIntensities}; 
        Float sumWeights = weights[0] + weights[1] + weights[2];
        
        if (loc.treeIdx == 3) {
            pmf *= weights[2] / sumWeights;
            return pmf / m_otherLights.size();
        }

        return 0;
    }

    PBRT_CPU_GPU PBRT_NOINLINE pstd::optional<SampledLight> Sample(Float u) const {
        const size_t totalSize = m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size();
        Float pmf = 1;
        {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, totalSize, pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        Float weights[3] = {m_spotTree.nodes[0].compactLightBounds.PhiOrI(), m_pointTree.nodes[0].compactLightBounds.PhiOrI(), m_otherLightIntensities}; 

        Float groupPMF;
        int groupIdx = SampleDiscrete(weights, u, &groupPMF, &u);
        pmf *= groupPMF;

        if (groupIdx == 0) {
            int index = std::min<int>(u * m_spotTree.lights.size(), m_spotTree.lights.size() - 1);
            pmf /= m_spotTree.lights.size();
            return SampledLight{m_spotTree.lights[index], pmf};
        } else if (groupIdx == 1) {
            int index = std::min<int>(u * m_pointTree.lights.size(), m_pointTree.lights.size() - 1);
            pmf /= m_pointTree.lights.size();
            return SampledLight{m_pointTree.lights[index], pmf};
        }

        int index = std::min<int>(u * m_otherLights.size(), m_otherLights.size() - 1);
        pmf /= m_otherLights.size();
        return SampledLight{m_otherLights[index], pmf};
    }

    PBRT_CPU_GPU PBRT_NOINLINE LightPMF PMF(Light light) const {
        const size_t totalSize = m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size();
        LightLocation loc = m_lightToLocation[light];

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = InfiniteLightSimplePMF(m_infiniteLights, totalSize);

        if (loc.treeIdx == 2) {
            return pInfinite;
        }

        Float weights[3] = {m_spotTree.nodes[0].compactLightBounds.PhiOrI(), m_pointTree.nodes[0].compactLightBounds.PhiOrI(), m_otherLightIntensities}; 
        Float sumWeights = weights[0] + weights[1] + weights[2];
        Float pmf = (1 - pInfinite);
        
        if (loc.treeIdx == 3) {
            pmf *= weights[2] / sumWeights;
            return pmf / m_otherLights.size();
        }

        const LightcutsTree& t(loc.treeIdx == 1 ? m_pointTree : m_spotTree);
        pmf *= weights[loc.treeIdx] / sumWeights;
        return pmf / t.lights.size(); 
    }
    
    template <int NSamples, typename ScatterEval>
    PBRT_CPU_GPU PBRT_NOINLINE void SampleLd(CountedArray<SampledLd, NSamples>& samples, const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const {
        const size_t totalSize = m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size();
        
        Float pmf = 1;
        if (!m_infiniteLights.empty()) {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, totalSize, pmf, u);
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

        Float weights[3] = {
            m_spotTree.lights.empty() ? 0 : m_spotTree.nodes[0].compactLightBounds.PhiOrI(),
            m_pointTree.lights.empty() ? 0 : m_pointTree.nodes[0].compactLightBounds.PhiOrI(),
            m_otherLightIntensities}; 

        Float groupPMF;
        const int groupIdx = SampleDiscrete(weights, u, &groupPMF, &u);
        const LightcutsTree* selectedTree = groupIdx == 0 ? &m_spotTree : groupIdx == 1 ? &m_pointTree : nullptr;

        pmf *= groupPMF;

        if (!selectedTree) {
            int index = std::min<int>(u * m_otherLights.size(), m_otherLights.size() - 1);
            pmf /= m_otherLights.size();

            Light light = m_otherLights[index];
            DCHECK(light && pmf != 0);
            pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
            if (!ls || !ls->L || ls->pdf == 0)
                return;

            Float lightPDF = pmf * ls->pdf;
            
            Float scatterPDF = 0;
            SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));
            samples.Add(SampledLd(f_hat * ls->L, light, ls->pLight, lightPDF, scatterPDF));
            return;
        }

        const Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.ns));

        Float errBounds[NSamples] = {0};
        CutData data[NSamples];

        uint32_t bitTrail = 0;
        int cutSize = ComputeLightcutsTreeCut<NSamples>(errBounds, data, bitTrail, ctx, selectedTree->nodes, selectedTree->allLightBounds, shadingFrame, bsdf, m_threshold, selectedTree == &m_spotTree);

        if (cutSize <= 0) {
            return;
        }
        
        constexpr uint32_t indexMask = std::numeric_limits<uint32_t>::max() >> 1;

        Point2f uOffset = GetR2SequenceOffset();
        for (int i = 0; i < NSamples; ++i) {
            Float errBound = errBounds[i];
            if (errBound <= 0) continue;

            uLight += uOffset;
            if (uLight.x >= 1) uLight.x -= 1;
            if (uLight.y >= 1) uLight.y -= 1;

            CutData clusterData = data[i];

            const uint32_t nodeIndex = clusterData.nodeIndex & indexMask;
            const LightcutsTreeNode& innerNode(selectedTree->nodes[nodeIndex]);
            const LightcutsTreeNode& representant(selectedTree->nodes[innerNode.representantIdx]);

            const Float nodeIntensity = innerNode.compactLightBounds.PhiOrI();
            const Float repIntensity = representant.compactLightBounds.PhiOrI();

            const int representantLightIndex = representant.childOrLightIndex;
            const Float scale = nodeIntensity / repIntensity;

            Light light = selectedTree->lights[representantLightIndex];
            DCHECK(light && pmf != 0);
            pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
            if (!ls || !ls->L || ls->pdf == 0)
                continue;

            Float lightPDF = pmf * ls->pdf;
            ls->L *= scale;

            Float scatterPDF = 0;
            SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));
            samples.Add(SampledLd(f_hat * ls->L, light, ls->pLight, lightPDF, scatterPDF));
        }

        return;
    }

    std::string ToString() const;

private:
    // LightcutsLightSampler Private Methods
#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights, LightcutsTree& tree, bool isPoint, Float& u);
#endif

    PBRT_CPU_GPU
    pstd::optional<SampledLight> SampleLightTree(const LightSampleContext& ctx, const LightcutsTree& tree, bool isPoint, const BSDF* bsdf, Float pmf, Float u) const;

    // LightcutsLightSampler Private Members
    LightcutsTree m_pointTree;
    LightcutsTree m_spotTree;
    pstd::vector<Light> m_otherLights;
    pstd::vector<Light> m_infiniteLights;
    HashMap<Light, LightLocation> m_lightToLocation;
    Float m_otherLightIntensities;
    Float m_threshold;
};

}

#endif  // PBRT_LIGHTCUTS_LIGHTSAMPLER_H
