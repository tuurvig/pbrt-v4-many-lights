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

#include <pbrt/util/pstd.h>
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

// LightcutsLightSampler Definition
class LightcutsLightSampler {
public:
    // LightcutsLightSampler Public Methods
    LightcutsLightSampler(pstd::span<const Light> lights, Allocator alloc);

    PBRT_CPU_GPU pstd::optional<SampledLight> Sample(const LightSampleContext& ctx, Float u) const {
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_nodes.empty() ? 0 : 1));

        if (u < pInfinite) {
            // Sample infinite lights with uniform probability
            u /= pInfinite;
            int index =
                std::min<int>(u * m_infiniteLights.size(), m_infiniteLights.size() - 1);
            Float pmf = pInfinite / m_infiniteLights.size();
            return SampledLight{m_infiniteLights[index], pmf};

        } else {
            if (m_nodes.empty()) {
                return {};
            }

            Point3f p = ctx.p();
            Normal3f n = ctx.ns;
            u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);
            int nodeIndex = 0;
            Float pmf = 1 - pInfinite;

            while (true) {
                LightcutsTreeNode node = m_nodes[nodeIndex];
                if (!node.isLeaf) {
                    const LightcutsTreeNode *children[2] = {&m_nodes[nodeIndex + 1],
                                                            &m_nodes[node.childOrLightIndex]};


                }
            }
        }
        return {};
    }

    PBRT_CPU_GPU Float PMF(const LightSampleContext& ctx, Light light) const { return PMF(light); }

    PBRT_CPU_GPU pstd::optional<SampledLight> Sample(Float u) const {
        if (m_lights.empty()) {
            return {};
        }
        int lightIndex = std::min<int>(u * m_lights.size(), m_lights.size() - 1);
        return SampledLight{m_lights[lightIndex], 1.f / m_lights.size()};
    }

    PBRT_CPU_GPU Float PMF(Light light) const {
        if (m_lights.empty()) {
            return 0;
        }
        return 1.f / m_lights.size();
    }

    std::string ToString() const;

private:
    struct LightBuildContainer{
        LightBuildContainer(const LightBounds& bounds, const Light& light) 
            : bounds(bounds), light(light) {}
        LightBounds bounds;
        Light light;
    };

    struct TreeNodeBuildSuccess {
        LightBounds bounds;
        int representantIdx;
        int nodeIdx;
    };
    
    // LightcutsLightSampler Private Methods
    TreeNodeBuildSuccess buildLightTree(std::vector<LightBuildContainer>& lightcutsLights, int start, int end, int depth, float& u);

    // Similarity Metric
    Float EvaluateCost(const LightBounds& bounds, bool isOmni) const {
        const Float diagonalLengthSqr = LengthSquared(bounds.bounds.Diagonal());

        Float similarity = diagonalLengthSqr;
        if (!isOmni) {
            const Float c_2 = LengthSquared(m_allLightBounds.Diagonal());
            const Float boundingConeHalfAngle = bounds.cosTheta_o;
            const Float oneMinusHalfAngle = 1.f - boundingConeHalfAngle;
            similarity += c_2 * oneMinusHalfAngle * oneMinusHalfAngle;
        }

        return bounds.phi * similarity;
    }

    // LightcutsLightSampler Private Members
    pstd::vector<Light> m_lights;
    pstd::vector<Light> m_infiniteLights;
    Bounds3f m_allLightBounds;
    pstd::vector<LightcutsTreeNode> m_nodes;
};

}

#endif  // PBRT_LIGHTCUTS_LIGHTSAMPLER_H
