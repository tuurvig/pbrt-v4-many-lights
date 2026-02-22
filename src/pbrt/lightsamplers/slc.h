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
        Normal3f n = ctx.n;

        BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
        Frame shadingFrame(bsdf ? bsdf->shadingFrame : Frame::FromZ(ctx.n));

        Float estL = 0;
        Float estParent = 0;
        Float clusterEst[2] = {};

        constexpr Float floatUintMax = 0x1p32f;
        uint32_t currentU = static_cast<uint32_t>(u * floatUintMax);

        int nodeIndex = 0;
        const LightcutsTreeNode* node = &m_tree.nodes[nodeIndex];

        Float clusterIntensity = 1;
        //Float repIntensity = 1;

        while (!node->isLeaf) {
            uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

            const LightcutsTreeNode *children[2] = {&m_tree.nodes[childrenIndices[0]],
                                                    &m_tree.nodes[childrenIndices[1]]};

            Float errBounds[2] = {1, 1};

            if (!ComputeErrorBounds(errBounds[0], errBounds[1], p, wo, n, shadingFrame, bsdf, children[0], children[1], m_tree.allLightBounds, true)) {
                return {};
            }

            const Float nodeIntensities[2] = {children[0]->compactLightBounds.PhiOrI(),
                                              children[1]->compactLightBounds.PhiOrI()};

            const LightcutsTreeNode *representants[2] = {&m_tree.nodes[children[0]->representantIdx],
                                                         &m_tree.nodes[children[1]->representantIdx]};

            const Float clusterEst[2] = {
                ComputeClusterEstimate(bsdf, bsdfFlags, representants[0]->compactLightBounds.Bound(m_tree.allLightBounds, false), p, n, wo, nodeIntensities[0]),
                ComputeClusterEstimate(bsdf, bsdfFlags, representants[1]->compactLightBounds.Bound(m_tree.allLightBounds, false), p, n, wo, nodeIntensities[1])
            };

            Float weights[2] = {0};
            weights[0] = std::min(OneMinusEpsilon, errBounds[0] / (errBounds[0] + errBounds[1]));
            weights[1] = 1 - weights[0];
            
            uint32_t threshold = static_cast<uint32_t>(weights[0] * floatUintMax);

            // Randomly sample a children node
            int child = 0;
            if (currentU < threshold) {
                currentU = static_cast<uint32_t>((static_cast<float>(currentU) / weights[0]));
            } else {
                child = 1;

                currentU -= threshold;
                currentU = static_cast<uint32_t>((static_cast<float>(currentU) / weights[1]));
            }
            
            currentU ^= FastIntegerHash(nodeIndex);
            pmf *= weights[child];
            nodeIndex = childrenIndices[child];
            node = &m_tree.nodes[nodeIndex];

            estL = estL - estParent + clusterEst[0] + clusterEst[1];
            estParent = clusterEst[child];

            if (errBounds[child] < m_threshold * estL) {
                clusterIntensity = nodeIntensities[child];
                //repIntensity = clusterIntensity;
                break;
            }
        }

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

            uint32_t threshold = static_cast<uint32_t>(weights[0] * floatUintMax);

            // Randomly sample a children node
            int child = 0;
            if (currentU < threshold) {
                currentU = static_cast<uint32_t>((static_cast<float>(currentU) / weights[0]));
            } else {
                child = 1;

                currentU -= threshold;
                currentU = static_cast<uint32_t>((static_cast<float>(currentU) / weights[1]));
            }

            currentU ^= FastIntegerHash(nodeIndex);
            pmfRepresentant *= weights[child];
            nodeIndex = childrenIndices[child];
            node = &m_tree.nodes[nodeIndex];

            //repIntensity = node->compactLightBounds.PhiOrI();
        }

        return SampledLight(m_tree.lights[node->childOrLightIndex], pmf, 1 / pmfRepresentant);
    }

    PBRT_CPU_GPU PBRT_NOINLINE
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, uint32_t /*seed*/, Light light) const {
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = InfiniteLightSimplePMF(m_infiniteLights, m_tree.nodes.size());

        // Handle infinite _light_ PMF
        if (!m_lightToBitTrail.HasKey(light))
            return pInfinite;

        // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = m_lightToBitTrail[light];
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        
        Float pmf = 1 - pInfinite;
        
        // Compute light's PMF by walking down tree nodes to the light
        return pmf / m_tree.lights.size();
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

    template <typename ScatterEval>
    PBRT_CPU_GPU PBRT_NOINLINE pstd::optional<SampledLd> SampleLd(const LightSampleContext& ctx, const SampledWavelengths& lambda, const BSDF* bsdf, uint32_t seed, Float u, Point2f uLight, ScatterEval scatterEval) const {
        pstd::optional<SampledLight> sampledLight = Sample(ctx, bsdf, seed, u);
        if (!sampledLight) {
            return {};
        }

        Light light = sampledLight->light;
        DCHECK(light && sampledLight->p != 0);
        pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
        if (!ls || !ls->L || ls->pdf == 0)
            return {};

        Float lightPDF = sampledLight->p * ls->pdf;
        ls->L *= sampledLight->scale;
        
        Float scatterPDF = 0;
        SampledSpectrum f_hat = scatterEval(scatterPDF, ctx.wo, ls->wi, IsDeltaLight(light.Type()));
        return SampledLd(f_hat * ls->L, ls->pLight, lightPDF, scatterPDF);
    }

    std::string ToString() const;

  private:
#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights, Float& u);
#endif

    PBRT_CPU_GPU
    bool ComputeErrorBounds(Float &err0, Float &err1, Point3f p, Vector3f wo, Normal3f n, const Frame& frame, const BSDF* bsdf, const LightcutsTreeNode * child0, const LightcutsTreeNode * child1, const Bounds3f& allLightBounds, bool isOriented) const {
        const Float nodeI0 = child0->compactLightBounds.PhiOrI();
        const Float nodeI1 = child1->compactLightBounds.PhiOrI();

        BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
        
        if (nodeI0 != 0 && nodeI1 != 0) {
            const Bounds3f nodeBound0 = child0->compactLightBounds.Bounds(allLightBounds);
            const Bounds3f nodeBound1 = child1->compactLightBounds.Bounds(allLightBounds);

            Float geomBound0 = ComputeGeometricBound(child0, nodeBound0, frame, isOriented, p, wo, bsdf && IsTransmissive(bsdfFlags));
            Float geomBound1 = ComputeGeometricBound(child1, nodeBound1, frame, isOriented, p, wo, bsdf && IsTransmissive(bsdfFlags));

            if (geomBound0 > MachineEpsilon && geomBound1 > MachineEpsilon) {
                Float ub0 = geomBound0 * nodeI0;
                Float ub1 = geomBound1 * nodeI1;

                Float matBound0 = 1;
                Float matBound1 = 1;

                if (bsdf) {
                    matBound0 = bsdf->Max_f(wo, nodeBound0, p);
                    matBound1 = bsdf->Max_f(wo, nodeBound1, p);
                }
                
                if ((matBound0 > MachineEpsilon && matBound1 > MachineEpsilon)) {
                    ub0 *= matBound0;
                    ub1 *= matBound1;

                    const Float diagonalLengthSqr0 = std::max(LengthSquared(nodeBound0.Diagonal()), MathEpsilon);
                    const Float diagonalLengthSqr1 = std::max(LengthSquared(nodeBound1.Diagonal()), MathEpsilon);

                    Float dist2Min0 = DistanceSquared(p, ClosestPoint(p, nodeBound0));
                    Float dist2Min1 = DistanceSquared(p, ClosestPoint(p, nodeBound1));

                    if (dist2Min0 > diagonalLengthSqr0 && dist2Min1 > diagonalLengthSqr1) {
                        Float dBoundMin0 = 1 / dist2Min0;
                        Float dBoundMin1 = 1 / dist2Min1;
                    
                        err0 = dBoundMin0 * ub0;
                        err1 = dBoundMin1 * ub1;
                    }
                    else {
                        err0 = ub0;
                        err1 = ub1;
                    }
                } else {
                    if (matBound0 < MachineEpsilon && matBound1 < MachineEpsilon) {
                        return false;
                    }

                    // weight of the first child will be 1 or 0 based on whether the other child is 0.
                    err0 = static_cast<Float>(matBound1 < MachineEpsilon);
                    err1 = 1 - err0;
                }
            } else {
                if (geomBound0 < MachineEpsilon && geomBound1 < MachineEpsilon) {
                    return false;
                }
                // weight of the first child will be 1 or 0 based on whether the other child is 0.
                err0 = static_cast<Float>(geomBound1 < MachineEpsilon);
                err1 = 1 - err0;
            }
        }    
        else {
            if (nodeI0 == 0 && nodeI1 == 0) {
                return false;
            }
            // weight of the first child will be 1 or 0 based on whether the other child is 0.
            err0 = static_cast<Float>(nodeI0 == 0);
            err1 = 1 - err0;
        }
        
        return true;
    }

    // LightcutsLightSampler Private Members
    LightcutsTree m_tree;
    pstd::vector<Light> m_infiniteLights;
    HashMap<Light, uint32_t> m_lightToBitTrail;
    Float m_threshold;
};

}

#endif // PBRT_SLC_LIGHTSAMPLER_H
