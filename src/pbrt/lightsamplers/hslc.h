// slc.h - SLCLightSampler class is Copyright(c) 2025-2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt and lightcuts.h source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_HSLC_LIGHTSAMPLER_H
#define  PBRT_HSLC_LIGHTSAMPLER_H

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

// Hierarchic Stochastic Lightcuts Lightsampler Definition
class HSLCLightSampler {
  public:
    // HierarchicLightcutsLightSampler Public Methods
    HSLCLightSampler(pstd::span<const Light> lights, Allocator alloc);

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
            
            constexpr Float minLengthSqr = 1e-6f;

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
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, Light light) const {
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
    // HierarchicLightcutsLightSampler Private Methods
#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildLightTreeGPU(std::vector<LightcutsBuildContainer> &lights, Float& u);
#endif

    PBRT_CPU_GPU
    inline bool ComputeErrorBounds(Float &err0, Float &err1, Point3f p, Vector3f wo, Normal3f n, const Frame& frame, const BSDF* bsdf, const LightcutsTreeNode * child0, const LightcutsTreeNode * child1, const Bounds3f& allLightBounds) const {
        const Float nodeI0 = child0->compactLightBounds.PhiOrI();
        const Float nodeI1 = child1->compactLightBounds.PhiOrI();

        BxDFFlags bsdfFlags = bsdf ? bsdf->Flags() : BxDFFlags::All;
        
        const Bounds3f nodeBound0 = child0->compactLightBounds.Bounds(allLightBounds);
        const Bounds3f nodeBound1 = child1->compactLightBounds.Bounds(allLightBounds);

        Float geomBound0 = ComputeGeometricBound(child0, nodeBound0, frame, true, p, wo, bsdf && IsTransmissive(bsdfFlags));
        Float geomBound1 = ComputeGeometricBound(child1, nodeBound1, frame, true, p, wo, bsdf && IsTransmissive(bsdfFlags));
        
        constexpr Float minLengthSqr = 1e-6f;
        
        Float ub0 = geomBound0 * nodeI0;
        Float ub1 = geomBound1 * nodeI1; 

        if (ub0 > MachineEpsilon && ub1 > MachineEpsilon) {   
            Float dist2Min0 = DistanceSquared(p, ClosestPoint(p, nodeBound0));
            Float dist2Min1 = DistanceSquared(p, ClosestPoint(p, nodeBound1));

            const Float diagonalLengthSqr0 = LengthSquared(nodeBound0.Diagonal());
            const Float diagonalLengthSqr1 = LengthSquared(nodeBound1.Diagonal());

            if (dist2Min0 >= diagonalLengthSqr0 && dist2Min1 >= diagonalLengthSqr1) {
                if (bsdf) {
                    ub0 *= bsdf->Max_f(wo, nodeBound0, p);
                    ub1 *= bsdf->Max_f(wo, nodeBound1, p);
                }

                err0 = ub0 / std::max(dist2Min0, minLengthSqr);
                err1 = ub1 / std::max(dist2Min1, minLengthSqr);
            }
            else {
                err0 = ub0;
                err1 = ub1;
            }
        } else {
            if (ub0 <= MachineEpsilon && ub1 <= MachineEpsilon) {
                return false;
            }
            // weight of the first child will be 1 or 0 based on whether the other child is 0.
            err0 = static_cast<Float>(ub1 <= MachineEpsilon);
            err1 = 1 - err0;
        }
        
        return true;
    }

    // HierarchicLightcutsLightSampler Private Members
    LightcutsTree m_tree;
    pstd::vector<Light> m_infiniteLights;
    HashMap<Light, uint32_t> m_lightToBitTrail;
};

}

#endif // PBRT_HSLC_LIGHTSAMPLER_H
