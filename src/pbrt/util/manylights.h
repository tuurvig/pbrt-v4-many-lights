// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// ManyLights util author Copytight(c) 2026 Richard Kvasnica 
#ifndef PBRT_UTIL_MANYLIGHTS_H
#define PBRT_UTIL_MANYLIGHTS_H

#include <pbrt/pbrt.h>

#include <pbrt/lights.h>
#include <pbrt/bsdf.h>

#include <pbrt/base/lightsampler.h>

#include <pbrt/util/vecmath.h>
#include <pbrt/util/lighttree_generic.h>

namespace pbrt {

/// CompactLightBounds Definition
//////////////////////////////////////////////////////////
class CompactLightBounds {
  public:
    // CompactLightBounds Public Methods
    CompactLightBounds() = default;

    PBRT_CPU_GPU
    CompactLightBounds(const LightBounds &lb, Float phiOrI, const Bounds3f &allb)
        : w(Normalize(lb.w)),
          phiOrI(phiOrI),
          qCosTheta_o(QuantizeCos(lb.cosTheta_o)),
          qCosTheta_e(QuantizeCos(lb.cosTheta_e)),
          twoSided(lb.twoSided) {
        // Quantize bounding box into _qb_
        for (int c = 0; c < 3; ++c) {
            qb[0][c] =
                pstd::floor(QuantizeBounds(lb.bounds[0][c], allb.pMin[c], allb.pMax[c]));
            qb[1][c] =
                pstd::ceil(QuantizeBounds(lb.bounds[1][c], allb.pMin[c], allb.pMax[c]));
        }
    }

    std::string ToString() const;
    std::string ToString(const Bounds3f &allBounds) const;

    PBRT_CPU_GPU
    Float PhiOrI() const { return phiOrI; }
    PBRT_CPU_GPU
    Vector3f W() const { return Vector3f(w); }
    PBRT_CPU_GPU
    bool TwoSided() const { return twoSided; }
    PBRT_CPU_GPU
    Float CosTheta_o() const {
        constexpr Float OneOverRange = static_cast<Float>(1.0 / 32767.0);
        return 2 * (qCosTheta_o * OneOverRange) - 1;
    }
    PBRT_CPU_GPU
    Float CosTheta_e() const {
        constexpr Float OneOverRange = static_cast<Float>(1.0 / 32767.0);
        return 2 * (qCosTheta_e * OneOverRange) - 1;
    }

    PBRT_CPU_GPU
    Point3f Bound(const Bounds3f& allb, bool max) const {
        constexpr Float OneOverRange = static_cast<Float>(1.0 / 65535.0);
        return Point3f(Lerp(qb[max][0] * OneOverRange, allb.pMin.x, allb.pMax.x),
                       Lerp(qb[max][1] * OneOverRange, allb.pMin.y, allb.pMax.y),
                       Lerp(qb[max][2] * OneOverRange, allb.pMin.z, allb.pMax.z));
    }

    PBRT_CPU_GPU
    Bounds3f Bounds(const Bounds3f &allb) const {
        return {Bound(allb, false), Bound(allb, true)};
    }

    PBRT_CPU_GPU
    Float Importance(Point3f p, Normal3f n, const Bounds3f &allb) const {
        Bounds3f bounds = Bounds(allb);
        Float cosTheta_o = CosTheta_o(), cosTheta_e = CosTheta_e();
        // Return importance for light bounds at reference point
        // Compute clamped squared distance to reference point
        Point3f pc = (bounds.pMin + bounds.pMax) / 2;
        Float d2 = DistanceSquared(p, pc);
        d2 = std::max(d2, Length(bounds.Diagonal()) / 2);

        // Define cosine and sine clamped subtraction lambdas
        auto cosSubClamped = [](Float sinTheta_a, Float cosTheta_a, Float sinTheta_b,
                                Float cosTheta_b) -> Float {
            if (cosTheta_a > cosTheta_b)
                return 1;
            return cosTheta_a * cosTheta_b + sinTheta_a * sinTheta_b;
        };

        auto sinSubClamped = [](Float sinTheta_a, Float cosTheta_a, Float sinTheta_b,
                                Float cosTheta_b) -> Float {
            if (cosTheta_a > cosTheta_b)
                return 0;
            return sinTheta_a * cosTheta_b - cosTheta_a * sinTheta_b;
        };

        // Compute sine and cosine of angle to vector _w_, $\theta_\roman{w}$
        Vector3f wi = Normalize(p - pc);
        Float cosTheta_w = Dot(Vector3f(w), wi);
        if (twoSided)
            cosTheta_w = std::abs(cosTheta_w);
        Float sinTheta_w = SafeSqrt(1 - Sqr(cosTheta_w));

        // Compute $\cos\,\theta_\roman{\+b}$ for reference point
        Float cosTheta_b = BoundSubtendedDirections(bounds, p).cosTheta;
        Float sinTheta_b = SafeSqrt(1 - Sqr(cosTheta_b));

        // Compute $\cos\,\theta'$ and test against $\cos\,\theta_\roman{e}$
        Float sinTheta_o = SafeSqrt(1 - Sqr(cosTheta_o));
        Float cosTheta_x = cosSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        Float sinTheta_x = sinSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        Float cosThetap = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
        if (cosThetap <= cosTheta_e)
            return 0;

        // Return final importance at reference point
        Float importance = phiOrI * cosThetap / d2;
        DCHECK_GE(importance, -1e-3);
        // Account for $\cos\theta_\roman{i}$ in importance at surfaces
        if (n != Normal3f(0, 0, 0)) {
            Float cosTheta_i = AbsDot(wi, n);
            Float sinTheta_i = SafeSqrt(1 - Sqr(cosTheta_i));
            Float cosThetap_i =
                cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
            importance *= cosThetap_i;
        }

        importance = std::max<Float>(importance, 0);
        return importance;
    }

  private:
    // CompactLightBounds Private Methods
    PBRT_CPU_GPU
    static unsigned int QuantizeCos(Float c) {
        CHECK(c >= -1 && c <= 1);
        return pstd::floor(32767.f * ((c + 1) / 2));
    }

    PBRT_CPU_GPU
    static Float QuantizeBounds(Float c, Float min, Float max) {
        CHECK(c >= min && c <= max);
        if (min == max)
            return 0;
        return 65535.f * Clamp((c - min) / (max - min), 0, 1);
    }

    // CompactLightBounds Private Members
    OctahedralVector w;
    Float phiOrI = 0;
    struct {
        unsigned int qCosTheta_o : 15;
        unsigned int qCosTheta_e : 15;
        unsigned int twoSided : 1;
    };
    uint16_t qb[2][3];
};

/// Light Hierarchy Nodes Definitions
//////////////////////////////////////////////////////////

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

    //PBRT_CPU_GPU
    //pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const;

    std::string ToString() const;

    // LightBVHNode Public Members
    CompactLightBounds lightBounds;
    struct {
        unsigned int childOrLightIndex : 31;
        unsigned int isLeaf : 1;
    };
};

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

/// Cost functions and evaluators
//////////////////////////////////////////////////////////

// Lightcuts original paper (2005) Similarity Metric
PBRT_CPU_GPU
inline Float SimilarityMetric(const LightBounds& bounds, Float sceneDiagonalSqr, bool isPointLight) {
    const Float diagonalLengthSqr = LengthSquared(bounds.bounds.Diagonal());

    Float similarity = diagonalLengthSqr;
    if (!isPointLight) {
        const Float c_2 = sceneDiagonalSqr;
        const Float boundingConeHalfAngle = bounds.cosTheta_o;
        const Float oneMinusHalfAngle = 1.f - boundingConeHalfAngle;
        similarity += c_2 * oneMinusHalfAngle * oneMinusHalfAngle;
    }

    return bounds.I * similarity;
}

struct LightcutsCostEvaluator {
    PBRT_CPU_GPU
    LightcutsCostEvaluator(Bounds3f bounds, bool isPoint) :
        sceneBoundsDiagonalSqr(LengthSquared(bounds.Diagonal())),
        isPoint(isPoint) {}

    PBRT_CPU_GPU Float operator()(const LightBounds &bounds) const {
        return SimilarityMetric(bounds, sceneBoundsDiagonalSqr, isPoint);
    }

    Float sceneBoundsDiagonalSqr;
    bool isPoint;
};

// SAOH heuristic cost from conty and kulla bvh lights paper 2018
PBRT_CPU_GPU
inline Float CostSAOH(const LightBounds& b) {
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

struct SAOHCostEvaluator {
    PBRT_CPU_GPU Float operator()(const LightBounds &bounds) const {
        return CostSAOH(bounds);
    }
};

/// Light Hierarchy Build results
//////////////////////////////////////////////////////////

struct LightcutsBuildResult {
    LightBounds bounds;
    int representantIdx;
    int nodeIdx;
};

struct LightHierarchyNodeBuildResult {
    LightBounds bounds;
    int nodeIdx;
};

/// Light Hierarchy Build containers
//////////////////////////////////////////////////////////

struct LightBVHBuildContainer : public BuildContainerInterface {
    LightBVHBuildContainer(const LightBounds& bounds, int index) 
        : BuildContainerInterface(bounds), index(index) {}
    int index;
};

struct LightcutsBuildContainer : public BuildContainerInterface {
    LightcutsBuildContainer(const LightBounds& bounds, const Light& light) 
        : BuildContainerInterface(bounds), light(light) {}
    LightBounds bounds;
    Light light;
    uint32_t index;
};

struct LightcutsTree {
    LightcutsTree(Allocator alloc);
    pstd::vector<Light> lights;
    pstd::vector<LightcutsTreeNode> nodes;
    Bounds3f allLightBounds;
};

/// Light Hierarchy Node Emitters
//////////////////////////////////////////////////////////

struct LightHierarchyNodeEmitter : public NodeEmitterInterface<LightBVHBuildContainer, LightHierarchyNodeBuildResult> {
    LightHierarchyNodeEmitter(pstd::vector<LightBVHNode>& nodes, HashMap<Light, uint32_t>& lightToBitTrail, const pstd::span<const Light>& lights, const Bounds3f& allLightBounds) : 
        nodes(&nodes), lightToBitTrail(&lightToBitTrail), lights(lights), allLightBounds(allLightBounds) {}

    pstd::vector<LightBVHNode>* nodes;
    HashMap<Light, uint32_t>* lightToBitTrail;
    pstd::span<const Light> lights;
    Bounds3f allLightBounds;

    virtual int ReserveInterior() override;
    virtual LightHierarchyNodeBuildResult EmitLeaf(const LightBVHBuildContainer& item, uint32_t bitTrail) override;
    virtual LightHierarchyNodeBuildResult FinalizeInterior(int reservationIndex, const LightHierarchyNodeBuildResult& left, const LightHierarchyNodeBuildResult& right, Float& u) override;
};

//struct LightcutsNodeEmitter : public NodeEmitterInterface<LightcutsBuildContainer, LightcutsBuildResult> {
//    LightcutsNodeEmitter(pstd::vector<>)
//};

/// Infinite Light Sample functions
//////////////////////////////////////////////////////////

PBRT_CPU_GPU
inline pstd::optional<SampledLight> InfiniteLightSimpleSample(const pstd::vector<Light>& infiniteLights, size_t nOtherLights, Float &pmf, Float &u) {
    // Compute infinite light sampling probability _pInfinite_
    Float pInfinite = Float(infiniteLights.size()) /
                      Float(infiniteLights.size() + (nOtherLights == 0 ? 0 : 1));

    if (u < pInfinite) {
        // Sample infinite lights with uniform probability
        u /= pInfinite;
        int index =
            std::min<int>(u * infiniteLights.size(), infiniteLights.size() - 1);
        Float pmf = pInfinite / infiniteLights.size();
        return SampledLight{infiniteLights[index], pmf};
    }

    u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);
    pmf = 1 - pInfinite;

    return {};
}

PBRT_CPU_GPU
inline Float InfiniteLightSimplePMF(const pstd::vector<Light>& infiniteLights, size_t nOtherLights) {
    Float pmf = 1 / Float(infiniteLights.size() + (nOtherLights == 0 ? 0 : 1));
    return pmf;
}

/// Cluster Estimate function
//////////////////////////////////////////////////////////

PBRT_CPU_GPU
inline Float ComputeClusterEstimate(const BSDF* bsdf, BxDFFlags flags, Point3f lightPos, Point3f point, Normal3f n, Vector3f wo, Float phi) {
    Float I = phi;

    Float minDistSqr = DistanceSquared(point, lightPos);
    Float clampedDistSqr = std::max(minDistSqr, MachineEpsilon);
    Float G = 1.0f / clampedDistSqr;

    n = bsdf ? Normal3f(bsdf->shadingFrame.z) : n;

    Vector3f wi = lightPos - point;
    wi /= std::sqrt(clampedDistSqr);
    Float cosTheta = Dot(n, wi);

    Float M = 1.f;
    if (bsdf) {
        SampledSpectrum sp = bsdf->f(wo, wi);
        Float M = sp.Average();
        if ((!IsTransmissive(flags) && cosTheta < 0) ||
            (!IsReflective(flags) && cosTheta >= 0)) {
            cosTheta = 0;
        }
    }

    return I * G * M * std::abs(cosTheta);
}

/// Geometric bounds
//////////////////////////////////////////////////////////

PBRT_CPU_GPU
inline Float GeomTermBoundInFrame(Point3f point, const Frame& frame, const Bounds3f& bounds) {
    // furthest point on the box along normal N
    Float zMax = MaxDistAlong(point, frame.z, bounds);
    if (zMax <= 0) return 0.0f;

    const Float xMin = AbsMinDistAlong(point, frame.x, bounds);
    const Float yMin = AbsMinDistAlong(point, frame.y, bounds);
    const Float hyp = SafeSqrt(Sqr(xMin) + Sqr(yMin) + Sqr(zMax));
    return zMax / hyp;
}

PBRT_CPU_GPU
inline Float ComputeGeometricBound(const LightcutsTreeNode* node, const Bounds3f& nodeBounds, const Frame& frame, bool isOriented, Point3f point, Vector3f wo) {
    Float G = GeomTermBoundInFrame(point, frame, nodeBounds);

    if (isOriented) {
        const Point3f refMin = point + (point - nodeBounds.pMax);
        const Point3f refMax = point + (point - nodeBounds.pMin);
        const Bounds3f refBounds(refMin, refMax);

        Frame coneFrame = Frame::FromZ(node->compactLightBounds.W());
        const Float boundCos = GeomTermBoundInFrame(point, coneFrame, refBounds);
        
        const Float halfAngle = std::acos(node->compactLightBounds.CosTheta_o());
        const Float maxCos = std::max((Float)0, std::cos(std::max((Float)0, std::acos(boundCos) - halfAngle)));

        G *= maxCos;
    }

    return G;
}

}

#endif //PBRT_UTIL_MANYLIGHTS_H

