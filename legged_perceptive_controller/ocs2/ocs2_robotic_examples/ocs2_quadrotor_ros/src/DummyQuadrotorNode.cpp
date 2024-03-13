/******************************************************************************
Copyright (c) 2017, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#include <ros/init.h>
#include <ros/package.h>

#include <ocs2_ros_interfaces/mrt/MRT_ROS_Dummy_Loop.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>

#include <ocs2_quadrotor/QuadrotorInterface.h>
#include <ocs2_quadrotor/definitions.h>

#include "ocs2_quadrotor_ros/QuadrotorDummyVisualization.h"

int main(int argc, char** argv) {
  const std::string robotName = "quadrotor";

  // task file
  std::vector<std::string> programArgs{};
  ::ros::removeROSArgs(argc, argv, programArgs);
  if (programArgs.size() <= 1) {
    throw std::runtime_error("No task file specified. Aborting.");
  }
  std::string taskFileFolderName = std::string(programArgs[1]);

  // Initialize ros node
  ros::init(argc, argv, robotName + "_mrt");
  ros::NodeHandle nodeHandle;

  // Robot interface
  const std::string taskFile = ros::package::getPath("ocs2_quadrotor") + "/config/" + taskFileFolderName + "/task.info";
  const std::string libFolder = ros::package::getPath("ocs2_quadrotor") + "/auto_generated";
  ocs2::quadrotor::QuadrotorInterface quadrotorInterface(taskFile, libFolder);

  // MRT
  ocs2::MRT_ROS_Interface mrt(robotName);
  mrt.initRollout(&quadrotorInterface.getRollout());
  mrt.launchNodes(nodeHandle);

  // Visualization
  auto quadrotorDummyVisualization = std::make_shared<ocs2::quadrotor::QuadrotorDummyVisualization>();

  // Dummy loop
  ocs2::MRT_ROS_Dummy_Loop dummyQuadrotor(mrt, quadrotorInterface.mpcSettings().mrtDesiredFrequency_,
                                          quadrotorInterface.mpcSettings().mpcDesiredFrequency_);
  dummyQuadrotor.subscribeObservers({quadrotorDummyVisualization});

  // initial state
  ocs2::SystemObservation initObservation;
  initObservation.state = quadrotorInterface.getInitialState();
  initObservation.input.setZero(ocs2::quadrotor::INPUT_DIM);
  initObservation.time = 0.0;

  // initial command
  const ocs2::TargetTrajectories initTargetTrajectories({initObservation.time}, {initObservation.state}, {initObservation.input});

  // Run dummy (loops while ros is ok)
  dummyQuadrotor.run(initObservation, initTargetTrajectories);

  return 0;
}
