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

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, const BSDF* bsdf, Float u) const {
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

        while (!node->isLeaf) {
            uint32_t childrenIndices[2] = {static_cast<uint32_t>(nodeIndex + 1), node->childOrLightIndex};

            const LightcutsTreeNode *children[2] = {&m_tree.nodes[childrenIndices[0]],
                                                    &m_tree.nodes[childrenIndices[1]]};
            
            const Float nodeIntensities[2] = {children[0]->compactLightBounds.PhiOrI(),
                                              children[1]->compactLightBounds.PhiOrI()};

            //const LightcutsTreeNode *representants[2] = {&tree.nodes[children[0]->representantIdx],
            //                                             &tree.nodes[children[1]->representantIdx]};

            //const Float clusterEst[2] = {
            //    ComputeClusterEstimate(bsdf, bsdfFlags, representants[0]->compactLightBounds.Bound(tree.allLightBounds, false), p, n, wo, nodeIntensities[0]),
            //    ComputeClusterEstimate(bsdf, bsdfFlags, representants[1]->compactLightBounds.Bound(tree.allLightBounds, false), p, n, wo, nodeIntensities[1])
            //};

            Float errBounds[2] = {1, 1};
            
            constexpr Float minLengthSqr = 1e-6f;

            if (nodeIntensities[0] != 0 && nodeIntensities[1] != 0) {
                const Bounds3f nodeBound0 = children[0]->compactLightBounds.Bounds(m_tree.allLightBounds);
                const Bounds3f nodeBound1 = children[1]->compactLightBounds.Bounds(m_tree.allLightBounds);

                const Float diagonalLengthSqr0 = std::max(LengthSquared(nodeBound0.Diagonal()), minLengthSqr);
                const Float diagonalLengthSqr1 = std::max(LengthSquared(nodeBound1.Diagonal()), minLengthSqr);

                Float dist2Min0 = DistanceSquared(p, ClosestPoint(p, nodeBound0));
                Float dist2Min1 = DistanceSquared(p, ClosestPoint(p, nodeBound1));

                //Float dist2Min0 = std::max(DistanceSquared(p, ClosestPoint(p, nodeBound0)), minLengthSqr);
                //Float dist2Min1 = std::max(DistanceSquared(p, ClosestPoint(p, nodeBound1)), minLengthSqr);

                //Float dist2Max0 = std::max(DistanceSquared(p, FurthestPoint(p, nodeBound0)), 1e-6f);
                //Float dist2Max1 = std::max(DistanceSquared(p, FurthestPoint(p, nodeBound1)), 1e-6f);

                Float geomBound0 = ComputeGeometricBound(children[0], nodeBound0, shadingFrame, true, p, wo);
                Float geomBound1 = ComputeGeometricBound(children[1], nodeBound1, shadingFrame, true, p, wo);

                //Float ub0 = nodeIntensities[0];
                //Float ub1 = nodeIntensities[1];
                Float ub0 = geomBound0 * nodeIntensities[0];
                Float ub1 = geomBound1 * nodeIntensities[1];

                if (bsdf) {
                    ub0 *= bsdf->Max_f(wo, nodeBound0, p);
                    ub1 *= bsdf->Max_f(wo, nodeBound1, p);
                }

                //if (dist2Min0 > diagonalLengthSqr0 && dist2Min1 >= diagonalLengthSqr1) {
                if (false) {
                //if (dist2Min0 > 0 && dist2Min1 > 0) {
                    
                    Float dBoundMin0 = 1 / dist2Min0;
                    Float dBoundMin1 = 1 / dist2Min1;
                    //Float dBoundMax0 = 1 / dist2Max0;
                    //Float dBoundMax1 = 1 / dist2Max1;

                    errBounds[0] = dBoundMin0 * ub0;
                    errBounds[1] = dBoundMin1 * ub1;

                    //Float ebMin0 = std::max(dBoundMin0 * ub0, MachineEpsilon);
                    //Float ebMin1 = std::max(dBoundMin1 * ub1, MachineEpsilon);
                    //Float ebMax0 = std::max(dBoundMin0 * ub0, MachineEpsilon);
                    //Float ebMax1 = std::max(dBoundMin1 * ub1, MachineEpsilon);

                    //Float nwMin = std::min(1.f, ebMin0 / (ebMin0 + ebMin1));
                    //Float nwMax = std::min(1.f, ebMax0 / (ebMax0 + ebMax1));

                    //errBounds[0] = (nwMin + nwMax) * 0.5f;
                    //errBounds[1] = 1 - errBounds[0];

                } else {
                    errBounds[0] = ub0;
                    errBounds[1] = ub1;
                    //canEnd = false;
                }
            } else {
                if (nodeIntensities[0] == 0 && nodeIntensities[1] == 0) {
                    return {};
                }
                // weight of the first child will be 1 or 0 based on whether the other child is 0.
                errBounds[0] = static_cast<Float>(nodeIntensities[1] == 0);
                errBounds[1] = 1 - errBounds[0];
                //canEnd = false;
            }

            if (errBounds[0] < MachineEpsilon) {
                
                if (errBounds[1] < MachineEpsilon) {
                    return {};
                }
                errBounds[0] = MachineEpsilon;
            } else if (errBounds[1] < MachineEpsilon){
                errBounds[1] = MachineEpsilon;
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
            pmf *= weights[child];
            nodeIndex = childrenIndices[child];
            node = &m_tree.nodes[nodeIndex];

            estL = estL - estParent + clusterEst[0] + clusterEst[1];
            estParent = clusterEst[child];

            //if (errBounds[child] < m_threshold * estL) {
            //    int representantLightIndex = representants[child]->childOrLightIndex;
            //    Float repIntensity = representants[child]->compactLightBounds.PhiOrI();
            //    return SampledLight(tree.lights[representantLightIndex], pmf, nodeIntensities[child] / repIntensity);
            //}
        }

        return SampledLight(m_tree.lights[node->childOrLightIndex], pmf);
    }

    PBRT_CPU_GPU
    LightPMF PMF(const LightSampleContext &ctx, const BSDF* bsdf, Light light) const {
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
