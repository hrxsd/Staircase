//
// Created by qiayuan on 23-1-1.
//

#include "legged_perceptive_interface/constraint/FootPlacementConstraint.h"
#include "legged_perceptive_interface/PerceptiveLeggedPrecomputation.h"
#include "legged_perceptive_interface/PerceptiveLeggedReferenceManager.h"

namespace legged {
FootPlacementConstraint::FootPlacementConstraint(const SwitchedModelReferenceManager& referenceManager,
                                                 const EndEffectorKinematics<scalar_t>& endEffectorKinematics, size_t contactPointIndex,
                                                 size_t numVertices)
    : StateConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
      contactPointIndex_(contactPointIndex),
      numVertices_(numVertices) {}

FootPlacementConstraint::FootPlacementConstraint(const FootPlacementConstraint& rhs)
    : StateConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      endEffectorKinematicsPtr_(rhs.endEffectorKinematicsPtr_->clone()),
      contactPointIndex_(rhs.contactPointIndex_),
      numVertices_(rhs.numVertices_) {}

bool FootPlacementConstraint::isActive(scalar_t time) const {
  return dynamic_cast<const PerceptiveLeggedReferenceManager&>(*referenceManagerPtr_).getFootPlacementFlags(time)[contactPointIndex_];
}

vector_t FootPlacementConstraint::getValue(scalar_t /*time*/, const vector_t& state, const PreComputation& preComp) const {
  const auto param = cast<PerceptiveLeggedPrecomputation>(preComp).getFootPlacementConParameters()[contactPointIndex_];
  return param.a * endEffectorKinematicsPtr_->getPosition(state).front() + param.b;
}

VectorFunctionLinearApproximation FootPlacementConstraint::getLinearApproximation(scalar_t /*time*/, const vector_t& state,
                                                                                  const PreComputation& preComp) const {
  VectorFunctionLinearApproximation approx = VectorFunctionLinearApproximation::Zero(numVertices_, state.size(), 0);
  const auto param = cast<PerceptiveLeggedPrecomputation>(preComp).getFootPlacementConParameters()[contactPointIndex_];

  const auto positionApprox = endEffectorKinematicsPtr_->getPositionLinearApproximation(state).front();
  approx.f = param.a * positionApprox.f + param.b;
  approx.dfdx = param.a * positionApprox.dfdx;
  return approx;
}

}  // namespace legged
