// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef  PBRT_BVH_LIGHTSAMPLER_H
#define  PBRT_BVH_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/containers.h>

namespace pbrt {

// LightBVHNode Definition
struct alignas(32) LightBVHNode {
    // LightBVHNode Public Methods
    LightBVHNode() = default;

    PBRT_CPU_GPU
    static LightBVHNode MakeLeaf(unsigned int lightIndex, const CompactLightBounds &cb) {
        return LightBVHNode{cb, {lightIndex, 1}};
    }

    PBRT_CPU_GPU
    static LightBVHNode MakeInterior(unsigned int child1Index,
                                     const CompactLightBounds &cb) {
        return LightBVHNode{cb, {child1Index, 0}};
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const;

    std::string ToString() const;

    // LightBVHNode Public Members
    CompactLightBounds lightBounds;
    struct {
        unsigned int childOrLightIndex : 31;
        unsigned int isLeaf : 1;
    };
};

struct LightBVHBuildContainer{
    LightBVHBuildContainer(const LightBounds& bounds, int index) 
        : bounds(bounds), index(index) {}
    LightBounds bounds;
    int index;
};

// BVHLightSampler Definition
class BVHLightSampler {
  public:
    // BVHLightSampler Public Methods
    BVHLightSampler(pstd::span<const Light> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
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
            // Traverse light BVH to sample light
            if (m_nodes.empty())
                return {};
            // Declare common variables for light BVH traversal
            Point3f p = ctx.p();
            Normal3f n = ctx.ns;
            u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);
            int nodeIndex = 0;
            Float pmf = 1 - pInfinite;

            while (true) {
                // Process light BVH node for light sampling
                LightBVHNode node = m_nodes[nodeIndex];
                if (!node.isLeaf) {
                    // Compute light BVH child node importances
                    const LightBVHNode *children[2] = {&m_nodes[nodeIndex + 1],
                                                       &m_nodes[node.childOrLightIndex]};
                    Float ci[2] = {
                        children[0]->lightBounds.Importance(p, n, m_allLightBounds),
                        children[1]->lightBounds.Importance(p, n, m_allLightBounds)};
                    if (ci[0] == 0 && ci[1] == 0)
                        return {};

                    // Randomly sample light BVH child node
                    Float nodePMF;
                    int child = SampleDiscrete(ci, u, &nodePMF, &u);
                    pmf *= nodePMF;
                    nodeIndex = (child == 0) ? (nodeIndex + 1) : node.childOrLightIndex;

                } else {
                    // Confirm light has nonzero importance before returning light sample
                    if (nodeIndex > 0)
                        DCHECK_GT(node.lightBounds.Importance(p, n, m_allLightBounds), 0);
                    if (nodeIndex > 0 ||
                        node.lightBounds.Importance(p, n, m_allLightBounds) > 0)
                        return SampledLight{m_lights[node.childOrLightIndex], pmf};
                    return {};
                }
            }
        }
    }

    PBRT_CPU_GPU
    Float PMF(const LightSampleContext &ctx, Light light) const {
        // Handle infinite _light_ PMF computation
        if (!m_lightToBitTrail.HasKey(light))
            return 1.f / (m_infiniteLights.size() + (m_nodes.empty() ? 0 : 1));

        // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = m_lightToBitTrail[light];
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(m_infiniteLights.size()) /
                          Float(m_infiniteLights.size() + (m_nodes.empty() ? 0 : 1));

        Float pmf = 1 - pInfinite;
        int nodeIndex = 0;

        // Compute light's PMF by walking down tree nodes to the light
        while (true) {
            const LightBVHNode *node = &m_nodes[nodeIndex];
            if (node->isLeaf) {
                DCHECK_EQ(light, m_lights[node->childOrLightIndex]);
                return pmf;
            }
            // Compute child importances and update PMF for current node
            const LightBVHNode *child0 = &m_nodes[nodeIndex + 1];
            const LightBVHNode *child1 = &m_nodes[node->childOrLightIndex];
            Float ci[2] = {child0->lightBounds.Importance(p, n, m_allLightBounds),
                           child1->lightBounds.Importance(p, n, m_allLightBounds)};
            DCHECK_GT(ci[bitTrail & 1], 0);
            pmf *= ci[bitTrail & 1] / (ci[0] + ci[1]);

            // Use _bitTrail_ to find next node index and update its value
            nodeIndex = (bitTrail & 1) ? node->childOrLightIndex : (nodeIndex + 1);
            bitTrail >>= 1;
        }
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (m_lights.empty())
            return {};
        int lightIndex = std::min<int>(u * m_lights.size(), m_lights.size() - 1);
        return SampledLight{m_lights[lightIndex], 1.f / m_lights.size()};
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        if (m_lights.empty())
            return 0;
        return 1.f / m_lights.size();
    }

    std::string ToString() const;

    static PBRT_CPU_GPU Float EvaluateCost(const LightBounds& b) {
        // Evaluate direction bounds measure for _LightBounds_
        Float theta_o = std::acos(b.cosTheta_o);
        Float theta_e = std::acos(b.cosTheta_e);
        Float theta_w = std::min(theta_o + theta_e, Pi);
        Float sinTheta_o = SafeSqrt(1 - Sqr(b.cosTheta_o));
        Float M_omega = 2 * Pi * (1 - b.cosTheta_o) +
                        Pi / 2 *
                            (2 * theta_w * sinTheta_o - std::cos(theta_o - 2 * theta_w) -
                             2 * theta_o * sinTheta_o + b.cosTheta_o);

        return b.phi * M_omega * b.bounds.SurfaceArea();
    }

  private:
    // BVHLightSampler Private Methods
    LightBVHBuildContainer buildBVH(
        std::vector<LightBVHBuildContainer> &bvhLights, int start, int end,
        uint32_t bitTrail, int depth);

#ifdef PBRT_BUILD_GPU_RENDERER
    bool buildBVHGPU(std::vector<LightBVHBuildContainer> &bvhLights);
#endif

    PBRT_CPU_GPU Float EvaluateCost(const LightBounds &b, const Bounds3f &bounds, int dim) const {
        // Return complete cost estimate for _LightBounds_
        Float Kr = MaxComponentValue(bounds.Diagonal()) / bounds.Diagonal()[dim];
        return EvaluateCost(b) * Kr;
    }

    // BVHLightSampler Private Members
    pstd::vector<Light> m_lights;
    pstd::vector<Light> m_infiniteLights;
    Bounds3f m_allLightBounds;
    pstd::vector<LightBVHNode> m_nodes;
    HashMap<Light, uint32_t> m_lightToBitTrail;
};

}

#endif // PBRT_BVH_LIGHTSAMPLER_H
