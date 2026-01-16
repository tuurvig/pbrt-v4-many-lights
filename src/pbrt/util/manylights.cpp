// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// ManyLights util author Copytight(c) 2026 Richard Kvasnica

#include "manylights.h"

namespace pbrt {

std::string LightBVHNode::ToString() const {
    return StringPrintf(
        "[ LightBVHNode lightBounds: %s childOrLightIndex: %d isLeaf: %d ]", lightBounds, childOrLightIndex, isLeaf);
}

std::string LightcutsTreeNode::ToString() const {
    return StringPrintf(
        "[ LightcutsTreeNode lightBounds: %s representantIndex: %d childOrLightIndex: %d isLeaf: %d ]", compactLightBounds, representantIdx, childOrLightIndex, isLeaf);
}

std::string CompactLightBounds::ToString() const {
    return StringPrintf(
        "[ CompactLightBounds qb: [ [ %u %u %u ] [ %u %u %u ] ] w: %s (%s) phiOrI: %f "
        "qCosTheta_o: %u (%f) qCosTheta_e: %u (%f) twoSided: %u ]",
        qb[0][0], qb[0][1], qb[0][2], qb[1][0], qb[1][1], qb[1][2], w, Vector3f(w), phiOrI,
        qCosTheta_o, CosTheta_o(), qCosTheta_e, CosTheta_e(), twoSided);
}

std::string CompactLightBounds::ToString(const Bounds3f &allBounds) const {
    return StringPrintf(
        "[ CompactLightBounds b: %s qb: [ [ %u %u %u ] [ %u %u %u ] ] w: %s (%s) phiOrI: %f "
        "qCosTheta_o: %u (%f) qCosTheta_e: %u (%f) twoSided: %u ]",
        Bounds(allBounds), qb[0][0], qb[0][1], qb[0][2], qb[1][0], qb[1][1], qb[1][2], w,
        Vector3f(w), phiOrI, qCosTheta_o, CosTheta_o(), qCosTheta_e, CosTheta_e(), twoSided);
}

int LightHierarchyNodeEmitter::ReserveInterior() {
    int index = static_cast<int>(nodes->size());
    nodes->push_back(LightBVHNode());
    return index;
}

LightHierarchyNodeBuildResult LightHierarchyNodeEmitter::EmitLeaf(const LightBVHBuildContainer& item, uint32_t bitTrail) {
    int nodeIndex = static_cast<int>(nodes->size());
    const LightBVHBuildContainer& container(item);
    CompactLightBounds cb(container.bounds, container.bounds.phi, allLightBounds);
    nodes->push_back(LightBVHNode::MakeLeaf(item.index, cb));
    lightToBitTrail->Insert(lights[item.index], bitTrail);
    return {item.bounds, nodeIndex};
}

LightHierarchyNodeBuildResult LightHierarchyNodeEmitter::FinalizeInterior(int reservationIndex, const LightHierarchyNodeBuildResult& left, const LightHierarchyNodeBuildResult& right, Float& u) {
    LightBounds lb = Union(left.bounds, right.bounds);
    CompactLightBounds cb(lb, lb.phi, allLightBounds);
    (*nodes)[reservationIndex] = LightBVHNode::MakeInterior(right.nodeIdx, cb);
    return {lb, reservationIndex};
}

int LightcutsNodeEmitter::ReserveInterior() {
    int index = static_cast<int>(tree->nodes.size());
    tree->nodes.emplace_back();
    return index;
}

LightcutsBuildResult LightcutsNodeEmitter::EmitLeaf(const LightcutsBuildContainer& item, uint32_t bitTrail) {
    const LightcutsBuildContainer& leaf(item);
    CompactLightBounds cb(leaf.bounds, leaf.bounds.I, tree->allLightBounds);

    int nodeIndex = static_cast<int>(tree->nodes.size());
    int lightIndex = static_cast<int>(tree->lights.size());

    tree->lights.emplace_back(leaf.light);
    tree->nodes.emplace_back(LightcutsTreeNode::MakeLeaf(lightIndex, nodeIndex, cb));
    lightToLocation->Insert(leaf.light, {static_cast<uint32_t>(isPoint), bitTrail});
    return {leaf.bounds, nodeIndex, nodeIndex};
}

LightcutsBuildResult LightcutsNodeEmitter::FinalizeInterior(int reservationIndex, const LightcutsBuildResult& left, const LightcutsBuildResult& right, Float& u) {    
    Float intensities[2] = {left.bounds.I, right.bounds.I};
    Float nodePMF;
    int child = SampleDiscrete(intensities, u, &nodePMF, &u);
    int successorIdx = (child == 0) ? left.representantIdx : right.representantIdx;

    LightBounds lb = Union(left.bounds, right.bounds);
    CompactLightBounds cb(lb, lb.I, tree->allLightBounds);
    
    tree->nodes[reservationIndex] = LightcutsTreeNode::MakeInterior(right.nodeIdx, successorIdx, cb);
    return {lb, successorIdx, reservationIndex};
}

int SLCNodeEmitter::ReserveInterior() {
    int index = static_cast<int>(tree->nodes.size());
    tree->nodes.emplace_back();
    return index;
}

LightcutsBuildResult SLCNodeEmitter::EmitLeaf(const LightcutsBuildContainer& item, uint32_t bitTrail) {
    const LightcutsBuildContainer& leaf(item);
    CompactLightBounds cb(leaf.bounds, leaf.bounds.I, tree->allLightBounds);

    int nodeIndex = static_cast<int>(tree->nodes.size());
    int lightIndex = static_cast<int>(tree->lights.size());

    tree->lights.emplace_back(leaf.light);
    tree->nodes.emplace_back(LightcutsTreeNode::MakeLeaf(lightIndex, nodeIndex, cb));
    lightToBitTrail->Insert(leaf.light, bitTrail);
    return {leaf.bounds, nodeIndex, nodeIndex};
}

LightcutsBuildResult SLCNodeEmitter::FinalizeInterior(int reservationIndex, const LightcutsBuildResult& left, const LightcutsBuildResult& right, Float& u) {    
    Float intensities[2] = {left.bounds.I, right.bounds.I};
    Float nodePMF;
    int child = SampleDiscrete(intensities, u, &nodePMF, &u);
    int successorIdx = (child == 0) ? left.representantIdx : right.representantIdx;

    LightBounds lb = Union(left.bounds, right.bounds);
    CompactLightBounds cb(lb, lb.I, tree->allLightBounds);
    
    tree->nodes[reservationIndex] = LightcutsTreeNode::MakeInterior(right.nodeIdx, successorIdx, cb);
    return {lb, successorIdx, reservationIndex};
}

}
