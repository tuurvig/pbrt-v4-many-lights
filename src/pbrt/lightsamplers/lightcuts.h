// lightcuts.h - LightcutsLightSampler class is Copyright(c) 2025-2026 Richard Kvasnica.
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt and lightcuts.h source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_LIGHTCUTS_LIGHTSAMPLER_H
#define  PBRT_LIGHTCUTS_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>
#include <pbrt/bsdf.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/math.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/containers.h>

namespace pbrt {

struct alignas(32) LightcutsTreeNode {
    LightcutsTreeNode() = default;

    PBRT_CPU_GPU static LightcutsTreeNode MakeLeaf(uint32_t lightIdx, uint32_t representantIdx, const CompactLightBounds& bounds) {
        return LightcutsTreeNode{bounds, representantIdx, {lightIdx, true}};
    }

    PBRT_CPU_GPU static LightcutsTreeNode MakeInterior(uint32_t childIdx, uint32_t representantIdx, const CompactLightBounds& bounds) {
        return LightcutsTreeNode{bounds, representantIdx, {childIdx, false}};
    }

    std::string ToString() const;

    // LightcutsTreeNode Public Members
    CompactLightBounds compactLightBounds; // 24 bytes
    uint32_t representantIdx; // 4 bytes
    struct { // 4 bytes
        uint32_t childOrLightIndex : 31;
        uint32_t isLeaf : 1;
    };
};

struct LightLocation {
    uint32_t treeIdx;
    uint32_t identifier;
};

struct LightcutsTree {
    LightcutsTree(bool isPoint, Allocator alloc);
    pstd::vector<Light> lights;
    pstd::vector<LightcutsTreeNode> nodes;
    Bounds3f allLightBounds;
    bool isPoint;
};

struct LightBuildContainer{
    LightBuildContainer(const LightBounds& bounds, const Light& light) 
        : bounds(bounds), light(light) {}
    LightBounds bounds;
    Light light;
    uint32_t index;
};

struct TreeNodeBuildSuccess {
    LightBounds bounds;
    int representantIdx;
    int nodeIdx;
};

// LightcutsLightSampler Definition
class LightcutsLightSampler {
    
public:
    // LightcutsLightSampler Public Methods
    LightcutsLightSampler(pstd::span<const Light> lights, Allocator alloc, Float threshold = 0.02);

    PBRT_CPU_GPU pstd::optional<SampledLight> Sample(const LightSampleContext& ctx, const BSDF* bsdf, Float u) const {
        const size_t totalSize = m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size();
        Float pmf = 1;
        if (!m_infiniteLights.empty()) {
            pstd::optional<SampledLight> infiniteLightSample = SampleInfiniteLight(totalSize, pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        Float weights[3] = {
            m_spotTree.lights.empty() ? 0 : m_spotTree.nodes[0].compactLightBounds.Phi(),
            m_pointTree.lights.empty() ? 0 : m_pointTree.nodes[0].compactLightBounds.Phi(),
            m_otherLightsPower}; 

        Float groupPMF;
        int groupIdx = SampleDiscrete(weights, u, &groupPMF, &u);
        pmf *= groupPMF;

        if (groupIdx == 0) {
            return SampleLightTree(ctx, m_spotTree, bsdf, pmf, u);
        } else if (groupIdx == 1) {
            return SampleLightTree(ctx, m_pointTree, bsdf, pmf, u);
        }

        int index = std::min<int>(u * m_otherLights.size(), m_otherLights.size() - 1);
        pmf /= m_otherLights.size();
        return SampledLight{m_otherLights[index], pmf};
    }

    PBRT_CPU_GPU LightPMF PMF(const LightSampleContext& ctx, const BSDF* bsdf, Light light) const {
        const size_t totalSize = m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size();
        LightLocation loc = m_lightToLocation[light];

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (totalSize == 0 ? 0 : 1));

        if (loc.treeIdx == 2) {
            return pInfinite / m_infiniteLights.size();
        }

        Float weights[3] = {m_spotTree.nodes[0].compactLightBounds.Phi(), m_pointTree.nodes[0].compactLightBounds.Phi(), m_otherLightsPower}; 
        Float sumWeights = weights[0] + weights[1] + weights[2];
        Float pmf = (1 - pInfinite);
        
        if (loc.treeIdx == 3) {
            pmf *= weights[2] / sumWeights;
            return pmf / m_otherLights.size();
        }

        const LightcutsTree& t(loc.treeIdx == 1 ? m_pointTree : m_spotTree);
        pmf *= weights[loc.treeIdx] / sumWeights;
        
        int nodeIndex = 0;
        Point3f p = ctx.p();
        Vector3f wo = ctx.wo;

        BxDFFlags bsdfFlags = BxDFFlags::All;
        if (bsdf) {
            bsdfFlags = bsdf->Flags();
        }

        Float estL = 0;
        Float estParentL = 0;

        const LightcutsTreeNode* node = &t.nodes[nodeIndex];
        uint32_t bitTrail = loc.identifier;
        while (!node->isLeaf) {
            // Compute child error bounds and update PMF for current node
            const LightcutsTreeNode* children[2] = {&t.nodes[nodeIndex + 1], &t.nodes[node->childOrLightIndex]};

            const LightcutsTreeNode *representants[2] = {&t.nodes[children[0]->representantIdx],
                                                           &t.nodes[children[1]->representantIdx]};

            const Float nodeIntensities[2] = {children[0]->compactLightBounds.Phi(),
                                              children[1]->compactLightBounds.Phi()};
            const Float clusterEst[2] = {
                ComputeClusterEstimate(bsdf, bsdfFlags, representants[0]->compactLightBounds.Bound(t.allLightBounds, false), p, wo, nodeIntensities[0]),
                ComputeClusterEstimate(bsdf, bsdfFlags, representants[1]->compactLightBounds.Bound(t.allLightBounds, false), p, wo, nodeIntensities[1])
            };

            Float errorBounds[2] = {ComputeErrorBounds(children[0], !t.isPoint, t.allLightBounds, bsdf, bsdfFlags, p, wo),
                                    ComputeErrorBounds(children[1], !t.isPoint, t.allLightBounds, bsdf, bsdfFlags, p, wo)};

            int child = bitTrail & 1;
            DCHECK_GT(errorBounds[child], 0);
            pmf *= errorBounds[child] / (errorBounds[0] + errorBounds[1]);

            estL = estL - estParentL + clusterEst[0] + clusterEst[1];
            estParentL = clusterEst[child];

            nodeIndex = child ? node->childOrLightIndex : (nodeIndex + 1);
            node = &t.nodes[nodeIndex];
            
            if (errorBounds[child] < m_threshold * estL) {
                if(light != t.lights[representants[child]->childOrLightIndex]) {
                    return 0;
                }
                Float repIntensity = representants[child]->compactLightBounds.Phi();
                return LightPMF(pmf, nodeIntensities[child] / repIntensity);
            }

            bitTrail >>=1;
        }

        DCHECK_EQ(light, t.lights[node->childOrLightIndex]);

        return pmf;
    }

    PBRT_CPU_GPU pstd::optional<SampledLight> Sample(Float u) const {
        const size_t totalSize = m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size();
        Float pmf = 1;
        {
            pstd::optional<SampledLight> infiniteLightSample = SampleInfiniteLight(totalSize, pmf, u);
            if (infiniteLightSample) {
                return infiniteLightSample;
            }
        }

        Float weights[3] = {m_spotTree.nodes[0].compactLightBounds.Phi(), m_pointTree.nodes[0].compactLightBounds.Phi(), m_otherLightsPower}; 

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

    PBRT_CPU_GPU LightPMF PMF(Light light) const {
        const size_t totalSize = m_pointTree.lights.size() + m_spotTree.lights.size() + m_otherLights.size();
        LightLocation loc = m_lightToLocation[light];

        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (totalSize == 0 ? 0 : 1));

        if (loc.treeIdx == 2) {
            return pInfinite / m_infiniteLights.size();
        }

        Float weights[3] = {m_spotTree.nodes[0].compactLightBounds.Phi(), m_pointTree.nodes[0].compactLightBounds.Phi(), m_otherLightsPower}; 
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

    std::string ToString() const;

    // Similarity Metric
    PBRT_CPU_GPU
    static Float EvaluateCost(const LightBounds& bounds, Float sceneDiagonalSqr, bool isPointLight) {
        const Float diagonalLengthSqr = LengthSquared(bounds.bounds.Diagonal());

        Float similarity = diagonalLengthSqr;
        if (!isPointLight) {
            const Float c_2 = sceneDiagonalSqr;
            const Float boundingConeHalfAngle = bounds.cosTheta_o;
            const Float oneMinusHalfAngle = 1.f - boundingConeHalfAngle;
            similarity += c_2 * oneMinusHalfAngle * oneMinusHalfAngle;
        }

        return bounds.phi * similarity;
    }
private:
    // LightcutsLightSampler Private Methods
    TreeNodeBuildSuccess buildLightTree(std::vector<LightBuildContainer>& lightcutsLights, LightcutsTree& tree, int start, int end, uint32_t bitTrail, int depth, float& u);

#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildLightTreeGPU(std::vector<LightBuildContainer> &lights, LightcutsTree& tree, HashMap<Light, LightLocation>& lightToLocation, float& u);
#endif
    PBRT_CPU_GPU
    pstd::optional<SampledLight> SampleLightTree(const LightSampleContext& ctx, const LightcutsTree& tree, const BSDF* bsdf, Float pmf, Float u) const;

    PBRT_CPU_GPU
    pstd::optional<SampledLight> SampleInfiniteLight(size_t nLights, Float &pmf, Float &u) const;

    PBRT_CPU_GPU
    static Float ComputeClusterEstimate(const BSDF* bsdf, BxDFFlags flags, Point3f lightPos, Point3f point, Vector3f wo, Float phi) {
        Float I = phi;

        Float minDistSqr = DistanceSquared(point, lightPos);
        Float clampedDistSqr = std::max(minDistSqr, 1e-4f);
        Float G = 1.0f / clampedDistSqr;

        if (!bsdf) {
            return I * G;
        }

        Vector3f wi = lightPos - point;
        wi /= std::sqrt(clampedDistSqr);

        SampledSpectrum sp = bsdf->f(wo, wi);
        Float M = sp.Average();

        if (M > 0) {
            Vector3f n = bsdf->shadingFrame.z;
            Float cosTheta = Dot(n, wi);
            if (!IsTransmissive(flags) && cosTheta < 0) {
                cosTheta = 0;
            }
            if (!IsReflective(flags) && cosTheta >= 0) {
                cosTheta = 0;
            }
            M *= cosTheta;
        }

        return I * G * M;
    }

    PBRT_CPU_GPU
    static Float ComputeErrorBounds(const LightcutsTreeNode* node, bool isOriented, const Bounds3f& sceneBounds, const BSDF* bsdf, BxDFFlags flags, Point3f point, Vector3f wo) {
        Bounds3f bounds = node->compactLightBounds.Bounds(sceneBounds);
        
        // 1. Light intensity (I)
        Float I = node->compactLightBounds.Phi();

        // 2. Visibility term (V) has trivial upper bound equal to 1

        // Calculate minimum squared distance to the bounding box
        Float minDistSqr = DistanceSquared(point, bounds);
        
        // 3. Geometric term (G)
        // Prevent division by zero if point is inside or on the bounds
        Float G = 1.0f / std::max(minDistSqr, 1e-4f);

        if (isOriented) {
            Float cosTheta_o = node->compactLightBounds.CosTheta_o();
            Vector3f w = node->compactLightBounds.W();
            Float maxCosEmission = BoundEmissionCosine(bounds, w, std::acos(cosTheta_o), point);
            G *= maxCosEmission;
        }

        if (!bsdf) {
            return I * G;
        }

        // 4. Material term (M)
        // do not compute for invalid bsdfs
        Float M = 1.f;

        // Bounding the max cosine is separate from founding the bsdf
        Float cosBound = BoundMaxCosine(point, IsReflective(flags), IsTransmissive(flags), bsdf->shadingFrame, bounds);
        if (cosBound != 0) {
            SampledSpectrum sp = bsdf->Max_f(wo, bounds, point);

            Float M = sp.MaxComponentValue() * cosBound;
        }
        
        return I * G * M;
    }

    // LightcutsLightSampler Private Members
    LightcutsTree m_pointTree;
    LightcutsTree m_spotTree;
    pstd::vector<Light> m_otherLights;
    pstd::vector<Light> m_infiniteLights;
    HashMap<Light, LightLocation> m_lightToLocation;
    Float m_otherLightsPower;
    Float m_threshold;
};

}

#endif  // PBRT_LIGHTCUTS_LIGHTSAMPLER_H
