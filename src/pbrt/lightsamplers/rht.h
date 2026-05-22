// Copyright(c) 2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
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

/// @brief Resampled Hierarchic Tree (RHT) light sampler.
/// Implements the stochastic traversal with splitting from Conty et al. (2024)
/// over a hierarchy of spherical light bounds.
class RHTLightSampler {
  public:
    // RHTLightSampler Public Methods
    /// @brief Builds the RHT hierarchy and caches bitTrails per-light.
    /// @param lights Scene lights provided by the integrator.
    /// @param alloc Pbrt allocator.
    /// @param gamma Controls the split-probability falloff in `SplitProbability()`.
    RHTLightSampler(pstd::span<const Light> lights, Allocator alloc, Float gamma = 0.2);

    /// @brief Context-dependent sampling entry point.
    /// Fallsback to the context-free `Sample(u)` path since the result might return more than one sample for resampling.
    /// Integrators will never use this path, but this must be preserved to keep the TaggedPointer functional.
    PBRT_CPU_GPU PBRT_NOINLINE
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Float u) const {
        return Sample(u);
    }

    /// @brief Evaluates PMF for context-dependent RHT sampling.
    /// Replays the stochastic traversal probabilities using the light's bitTrail.
    PBRT_CPU_GPU
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t seed, Light light) const {
        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return InfiniteLightSimplePMF(m_infiniteLights, m_tree.leaves.size());;
        
        if (m_tree.leaves.empty())
            return 0;
        
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_tree.leaves.size() == 0 ? 0 : 1));

        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        uint32_t bitTrail = m_lightToBitTrail[light];

        uint32_t nodeIndex = 0;
        Float PsParent = 1;
        Float T = 1;
        
        const Float uSplit = HashFloat(seed);
        const ResampledTreeNode* node = &m_tree.innerNodes[nodeIndex];
        while (!node->isLeaf) {
            const uint32_t childrenIndices[2] = {nodeIndex + 1, node->childOrLightIndex};

            const Float splitProb = node->bounds.SplitProbability(p, gamma);
            Float PsNode = PsParent; // Ps(C)
            Float T_node = T;
            if (PsParent - splitProb > MachineEpsilon) {
                PsNode = splitProb;
                const Float PsHatNode = 1 - PsNode; // Ps_hat(C)

                // Probability of splitting parent given that current node has not split.
                const Float Pns = std::min((PsParent - PsNode) / PsHatNode, OneMinusEpsilon); // Pns(C)
                T_node = Pns + (1 - Pns) * T;
            }

            const uint32_t child = bitTrail & 1;
            
            const ResampledTreeNode *children[2] = {&m_tree.innerNodes[childrenIndices[0]],
                                                    &m_tree.innerNodes[childrenIndices[1]]};
            const Float ci[2] = {children[0]->bounds.Importance(p, n),
                                 children[1]->bounds.Importance(p, n)};

            const Float sumImportance = ci[0] + ci[1];
            if (sumImportance == 0) {
                return 0;
            }

            Float weight[2] = {0};
            weight[0] = std::clamp(ci[0] / sumImportance, MathEpsilon, OneMinusEpsilon);
            weight[1] = 1 - weight[0];

            T = T_node * weight[child];

            PsParent = PsNode;
            nodeIndex = childrenIndices[child];
            node = &m_tree.innerNodes[nodeIndex];

            bitTrail >>= 1;
        }

        return (1 - pInfinite) * (PsParent + (1 - PsParent) * T);
    }

    /// @brief Context-free fallback: uniform sampling over lights in leaves.
    PBRT_CPU_GPU PBRT_NOINLINE
    pstd::optional<SampledLight> Sample(Float u) const {
        Float pmf = 1;
        {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.leaves.size(), pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        if (m_tree.leaves.empty()) return {};

        uint32_t index = std::min<uint32_t>(u * static_cast<Float>(m_tree.leaves.size()), m_tree.leaves.size() - 1);
        pmf /= m_tree.leaves.size();
        return SampledLight{m_tree.leaves[index].light, pmf};
    }

    /// @brief PMF for context-free fallback path.
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
    
    /// Size of the first-stage candidate reservoir set (heuristic H).
#ifndef PBRT_RHT_RESERVOIR_SET_H_SIZE
#define PBRT_RHT_RESERVOIR_SET_H_SIZE 16
#endif

    /// @brief Produces direct-light samples using two-stage reservoir resampling.
    /// Stage H collects candidates from tree traversal; stage F resamples by contribution.
    /// @tparam NSamples Maximum number of output samples.
    /// @tparam ScatterEval Callable evaluating BSDF contribution and scatter PDF.
    template <int NSamples, typename ScatterEval>
    PBRT_CPU_GPU PBRT_NOINLINE void SampleLd(CountedArray<SampledLd, NSamples>& samples, const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const {
        Float pmf = 1;
        {
            pstd::optional<SampledLight> infiniteLightSample = InfiniteLightSimpleSample(m_infiniteLights, m_tree.leaves.size(), pmf, u);
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

        const Point3f p = ctx.p();
        const Normal3f n = ctx.ns;

        // Heuristic F reservoir: final selection over evaluated direct-light candidates.
        InPlaceWeightedReservoirSetSampler<SampledLd, NSamples> heuristicFSampler(samples.elements, Hash(u, MixBits(seed)));
        {
            // Heuristic H reservoir set: tree-traversal candidates with proposal PDFs.
            HeuristicHReservoirSet heuristicHSampler(Hash(u, seed));
            CollectLightCandidates(heuristicHSampler, ctx, seed, u, HashFloat(seed), pmf);
            {
                Point2f uLightOffset = GetR2SequenceOffset();

                for (int i = 0; i < heuristicHSampler.Size(); ++i) {
                    const Point2f uLightCurrent = uLight;
                    uLight += uLightOffset;

                    const auto overOneX = static_cast<int>(uLight.x);
                    const auto overOneY = static_cast<int>(uLight.y);
                    uLight.x -= static_cast<Float>(overOneX);
                    uLight.y -= static_cast<Float>(overOneY);
                 
                    const StatelessWeightedReservoirSampler<LightCandidate>& reservoir(heuristicHSampler.GetReservoir(i));
                    if (!reservoir.HasSample()) {
                        continue;
                    }
                
                    const LightCandidate& sample(reservoir.GetSample());
                    const Float hProb = reservoir.SampleProbability();
                
                    Light light = m_tree.leaves[sample.lightIdx].light;
                    DCHECK(light && sample.pmf != 0);
                    pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLightCurrent, lambda, true);
                    if (!ls || !ls->L || ls->pdf == 0 || hProb <= 0)
                        continue;
                
                    const Float lightPDF = sample.pmf * ls->pdf;
                    
                    Float scatterPDF = 0;
                    SampledSpectrum contribution = ls->L;
                    contribution *= scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));

                    // F(Si) = bsdf * (Li / pdfLight) * misW * hW(Li)
                    const Float denom = std::max(hProb * (lightPDF + scatterPDF), MathEpsilon);
                    const Float fWeight = contribution.MaxComponentValue() / denom;
                    if (fWeight > 0) {
                        heuristicFSampler.Add([&]{
                            return SampledLd(ClampZero(contribution), light, ls->pLight, lightPDF, scatterPDF, std::numeric_limits<uint32_t>::max(), hProb);
                        }, fWeight);
                    }
                }
            }
        }
        
        // Compact valid F-reservoir entries to the front and always apply
        // the reservoir selection probability for unbiased normalization.
        int out = 0;
        for (int i = 0; i < NSamples; ++i) {
            if (!heuristicFSampler.HasSample(i))
                continue;

            if (out != i)
                samples.elements[out] = samples.elements[i];

            const Float fProb = heuristicFSampler.SampleProbability(i);
            const Float hProb = samples.elements[i].pdfCancellationFactor;
            samples.elements[out].Ld /= std::max(fProb * hProb, MathEpsilon);
            ++out;
        }
        samples.count = out;
    }

    std::string ToString() const;

  private:
#ifdef PBRT_BUILD_GPU_RENDERER
    /// @brief Attempts GPU construction of the RHT hierarchy.
    bool buildLightTreeGPU(std::vector<RHTBuildContainer> &lights);
#endif

    /// Reservoir set type for heuristic H candidate generation.
    using HeuristicHReservoirSet = WeightedReservoirSetSampler<LightCandidate, PBRT_RHT_RESERVOIR_SET_H_SIZE>;

    PBRT_CPU_GPU PBRT_NOINLINE
    /// @brief Traverses the RHT and fills heuristic-H candidate reservoirs.
    /// @param reservoirSet Output reservoir set receiving candidate leaves.
    /// @param ctx Shading context.
    /// @param seed Randomization seed.
    /// @param u Primary random sample.
    /// @param uSplit Random sample that decides split vs no-split events.
    /// @param pmf Prefix probability accumulated before entering RHT sampling.
    void CollectLightCandidates(HeuristicHReservoirSet& reservoirSet, const LightSampleContext& ctx, uint32_t seed, Float u, Float uSplit, Float pmf) const;


    // RHTLightSampler Private Members
    ResampledTree m_tree;                       ///< Light hierarchy (spherical inner nodes + compact leaves with tighter bounds).
    pstd::vector<Light> m_infiniteLights;       ///< Infinite/environment lights.
    HashMap<Light, uint32_t> m_lightToBitTrail; ///< Encoded bitTrail paths for PMF reconstruction.
    Float gamma;                                ///< Split-probability shape parameter.
};

}
#endif // PBRT_RHT_LIGHTSAMPLER_H
