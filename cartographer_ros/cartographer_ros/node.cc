/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer_ros/node.h"

#include <chrono>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "cartographer/common/configuration_file_resolver.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/port.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/pose_graph_interface.h"
#include "cartographer/mapping/proto/submap_visualization.pb.h"
#include "cartographer/metrics/register.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/transform/transform.h"
#include "cartographer_ros/metrics/family_factory.h"
#include "cartographer_ros/msg_conversion.h"
#include "cartographer_ros/sensor_bridge.h"
#include "cartographer_ros/tf_bridge.h"
#include "cartographer_ros/slam_exception.h"
#include "cartographer_ros/time_conversion.h"
#include "cartographer_ros_msgs/StatusCode.h"
#include "cartographer_ros_msgs/StatusResponse.h"
#include "glog/logging.h"
#include "nav_msgs/Odometry.h"
#include "ros/serialization.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf2_eigen/tf2_eigen.h"
#include "visualization_msgs/MarkerArray.h"

#include <generic_lib/StatusResponse.h>
#include <generic_lib/StatusCode.h>

#include <temi_logger.h>
#include <generic_lib/tools/drawing_tools.h>

namespace cartographer_ros {

namespace carto = ::cartographer;

using carto::transform::Rigid3d;
using TrajectoryState =
    ::cartographer::mapping::PoseGraphInterface::TrajectoryState;

namespace {
std::string TrajectoryStateToString(const TrajectoryState trajectory_state) {
  switch (trajectory_state) {
    case TrajectoryState::ACTIVE:
      return "ACTIVE";
    case TrajectoryState::FINISHED:
      return "FINISHED";
    case TrajectoryState::FROZEN:
      return "FROZEN";
    case TrajectoryState::DELETED:
      return "DELETED";
  }
  return "";
}

}  // namespace

using generic_lib::tools::GenericServiceServer;


Node::Node(
    const NodeOptions& node_options,
    const TrajectoryOptions& trajectory_options,
    std::unique_ptr<cartographer::mapping::MapBuilderInterface> map_builder,
    tf2_ros::Buffer* const tf_buffer, const bool collect_metrics)
    : node_options_(node_options),
      default_trajectory_options_(trajectory_options),
      tf_wrapper_(
#ifdef ZMQ_MSG_SYS
        [this](const std::string& dst,
        const std::string& src,
        ros::Time time,
        ros::Duration timeout) {
          tf::StampedTransform tf;
          zmq_transform::get_stamped_transform(
          dst.c_str(), src.c_str(),
          time, timeout, 0, 0, LOOKUP, tf);
          geometry_msgs::TransformStamped tf_msg;
          tf::transformStampedTFToMsg(tf, tf_msg);
          return tf_msg;}
#elif ROS_MSG_SYS
        [tf_buffer](const std::string& dst,
        const std::string& src,
        ros::Time time,
        ros::Duration timeout) {
          return tf_buffer->lookupTransform(
                dst, src,
                time, timeout);}
#endif
      ),
      map_builder_bridge_(node_options_, std::move(map_builder), &tf_wrapper_),
      low_power_mode_(false),
      calling_navigation_node_to_load_(false),
      scan_stable_(false) {
  absl::MutexLock lock(&mutex_);
  if (collect_metrics) {
    metrics_registry_ = absl::make_unique<metrics::FamilyFactory>();
    carto::metrics::RegisterAllMetrics(metrics_registry_.get());
  }
}

Node::~Node() { FinishAllTrajectories(); }

#ifdef ROS_MSG_SYS
::ros::NodeHandle* Node::node_handle() { return GenericBase::nh_; }
#endif

bool Node::HandleSubmapQuery(
    ::cartographer_ros_msgs::SubmapQuery::Request& request,
    ::cartographer_ros_msgs::SubmapQuery::Response& response) {
  absl::MutexLock lock(&mutex_);
  map_builder_bridge_.HandleSubmapQuery(request, response);
  return true;
}

bool Node::HandleTrajectoryQuery(
    ::cartographer_ros_msgs::TrajectoryQuery::Request& request,
    ::cartographer_ros_msgs::TrajectoryQuery::Response& response) {
  absl::MutexLock lock(&mutex_);
  response.status = TrajectoryStateToStatus(
      request.trajectory_id,
      {TrajectoryState::ACTIVE, TrajectoryState::FINISHED,
       TrajectoryState::FROZEN} /* valid states */);
  if (response.status.code != cartographer_ros_msgs::StatusCode::OK) {
    LOG(ERROR) << "Can't query trajectory from pose graph: "
               << response.status.message;
    return true;
  }
  map_builder_bridge_.HandleTrajectoryQuery(request, response);
  return true;
}

void Node::PublishSubmapList(const ::ros::WallTimerEvent& unused_timer_event) {
  absl::MutexLock lock(&mutex_);
  submap_list_publisher_.publish(map_builder_bridge_.GetSubmapList());
}

void Node::AddExtrapolator(const int trajectory_id,
                           const TrajectoryOptions& options) {
  constexpr double kExtrapolationEstimationTimeSec = 0.001;  // 1 ms
  CHECK(extrapolators_.count(trajectory_id) == 0);
  const double gravity_time_constant =
      node_options_.map_builder_options.use_trajectory_builder_3d()
          ? options.trajectory_builder_options.trajectory_builder_3d_options()
                .imu_gravity_time_constant()
          : options.trajectory_builder_options.trajectory_builder_2d_options()
                .imu_gravity_time_constant();
  extrapolators_.emplace(
      std::piecewise_construct, std::forward_as_tuple(trajectory_id),
      std::forward_as_tuple(
          ::cartographer::common::FromSeconds(kExtrapolationEstimationTimeSec),
          gravity_time_constant));
}

void Node::AddSensorSamplers(const int trajectory_id,
                             const TrajectoryOptions& options) {
  CHECK(sensor_samplers_.count(trajectory_id) == 0);
  sensor_samplers_.emplace(
      std::piecewise_construct, std::forward_as_tuple(trajectory_id),
      std::forward_as_tuple(
          options.rangefinder_sampling_ratio, options.odometry_sampling_ratio,
          options.fixed_frame_pose_sampling_ratio, options.imu_sampling_ratio,
          options.landmarks_sampling_ratio));
}

void Node::PublishLocalTrajectoryData(const ::ros::WallTimerEvent& timer_event) {
  absl::MutexLock lock(&mutex_);
  for (const auto& entry : map_builder_bridge_.GetLocalTrajectoryData()) {
    const auto& trajectory_data = entry.second;

    auto& extrapolator = extrapolators_.at(entry.first);
    // We only publish a point cloud if it has changed. It is not needed at high
    // frequency, and republishing it would be computationally wasteful.
    if (trajectory_data.local_slam_data->time !=
        extrapolator.GetLastPoseTime() || low_power_mode_) {
      if (node_options_.publish_scan_matched_point_cloud && !low_power_mode_) {
        // TODO(gaschler): Consider using other message without time
        // information.
        carto::sensor::TimedPointCloud point_cloud;
        point_cloud.reserve(trajectory_data.local_slam_data->range_data_in_local
                                .returns.size());
        for (const cartographer::sensor::RangefinderPoint point :
             trajectory_data.local_slam_data->range_data_in_local.returns) {
          point_cloud.push_back(cartographer::sensor::ToTimedRangefinderPoint(
              point, 0.f /* time */));
        }
        scan_matched_point_cloud_publisher_.publish(ToPointCloud2Message(
            carto::common::ToUniversal(trajectory_data.local_slam_data->time),
            node_options_.map_frame,
            carto::sensor::TransformTimedPointCloud(
                point_cloud, trajectory_data.local_to_map.cast<float>())));
      }
      const ::cartographer::common::Time now = std::max(
          FromRos(ros::Time::now()), extrapolator.GetLastExtrapolatedTime());
      if (trajectory_data.local_slam_data->time > extrapolator.GetLastPoseTime()) {
        extrapolator.AddPose(
              trajectory_data.local_slam_data->time,
                           trajectory_data.local_slam_data->local_pose);
        last_added_pose_[entry.first] = trajectory_data.local_slam_data->local_pose;
      } else if (now - cartographer::common::FromSeconds(node_options_.pose_publish_period_sec) >
                 extrapolator.GetLastPoseTime() && low_power_mode_) {
        extrapolator.AddPose(
              now - cartographer::common::FromSeconds(node_options_.pose_publish_period_sec),
              last_added_pose_[entry.first]);
        last_added_pose_[entry.first] = extrapolator.ExtrapolatePose(now);
      } else {
        TEMI_LOG(trace) << "AddPose skipped: " << ToRos(trajectory_data.local_slam_data->time) << "; " << ToRos(extrapolator.GetLastPoseTime());
      }
    }
    geometry_msgs::TransformStamped stamped_transform;
    // If we do not publish a new point cloud, we still allow time of the
    // published poses to advance. If we already know a newer pose, we use its
    // time instead. Since tf knows how to interpolate, providing newer
    // information is better.
    // Get latest TF instead of using Now, so we would not be mistakenly using
    //  now pose and laser_time TF to generate publish_to_map
    geometry_msgs::TransformStamped latest_published_to_tracking_transform =
        tf_wrapper_.lookupTransform(
            trajectory_data.trajectory_options.tracking_frame,
            trajectory_data.trajectory_options.published_frame,
            ::ros::Time(0.),
            ::ros::Duration(node_options_.lookup_transform_timeout_sec));
    const ::cartographer::common::Time now = std::max(
        FromRos(latest_published_to_tracking_transform.header.stamp), extrapolator.GetLastExtrapolatedTime());
    stamped_transform.header.stamp =
        node_options_.use_pose_extrapolator
            ? ToRos(now)
            : ToRos(trajectory_data.local_slam_data->time);

    // Suppress publishing if we already published a transform at this time.
    // Due to 2020-07 changes to geometry2, tf buffer will issue warnings for
    // repeated transforms with the same timestamp.
    if (last_published_tf_stamps_.count(entry.first) &&
        last_published_tf_stamps_[entry.first] == stamped_transform.header.stamp)
      continue;
    last_published_tf_stamps_[entry.first] = stamped_transform.header.stamp;

    const Rigid3d tracking_to_local_3d =
        node_options_.use_pose_extrapolator
            ? extrapolator.ExtrapolatePose(now)
            : trajectory_data.local_slam_data->local_pose;
    const Rigid3d tracking_to_local = [&] {
      if (trajectory_data.trajectory_options.publish_frame_projected_to_2d) {
        return carto::transform::Embed3D(
            carto::transform::Project2D(tracking_to_local_3d));
      }
      return tracking_to_local_3d;
    }();

    const Rigid3d tracking_to_map =
        trajectory_data.local_to_map * tracking_to_local;

    if (trajectory_data.published_to_tracking != nullptr) {
      if (node_options_.publish_to_tf) {
        if (trajectory_data.trajectory_options.provide_odom_frame) {
          // TODO(Anders): in zmq we don't support sending multi transforms
          // std::vector<geometry_msgs::TransformStamped> stamped_transforms;

          stamped_transform.header.frame_id = node_options_.map_frame;
          stamped_transform.child_frame_id =
              trajectory_data.trajectory_options.odom_frame;
          stamped_transform.transform =
              ToGeometryMsgTransform(trajectory_data.local_to_map);
          // stamped_transforms.push_back(stamped_transform);
          tf_broadcaster_.sendTransform(stamped_transform);

          stamped_transform.header.frame_id =
              trajectory_data.trajectory_options.odom_frame;
          stamped_transform.child_frame_id =
              trajectory_data.trajectory_options.published_frame;
          stamped_transform.transform = ToGeometryMsgTransform(
              tracking_to_local * (*trajectory_data.published_to_tracking));
          // stamped_transforms.push_back(stamped_transform);

          // tf_broadcaster_.sendTransform(stamped_transforms);
          tf_broadcaster_odom_.sendTransform(stamped_transform);
        } else {
          stamped_transform.header.frame_id = node_options_.map_frame;
          stamped_transform.child_frame_id =
              trajectory_data.trajectory_options.published_frame;
          stamped_transform.transform = ToGeometryMsgTransform(
              tracking_to_map * ToRigid3d(latest_published_to_tracking_transform));  // 'published to map'
          tf_broadcaster_.sendTransform(stamped_transform);
        }
      }
      if (node_options_.publish_tracked_pose) {
        ::geometry_msgs::PoseStamped pose_msg;
        pose_msg.header.frame_id = node_options_.map_frame;
        pose_msg.header.stamp = stamped_transform.header.stamp;
        pose_msg.pose = ToGeometryMsgPose(tracking_to_map);
        tracked_pose_publisher_.publish(pose_msg);
      }
    }
  }
}

void Node::PublishTrajectoryNodeList(
    const ::ros::WallTimerEvent& unused_timer_event) {
  if (trajectory_node_list_publisher_.getNumSubscribers() > 0) {
    absl::MutexLock lock(&mutex_);
    trajectory_node_list_publisher_.publish(
        map_builder_bridge_.GetTrajectoryNodeList());
    // added by Temi
    if (node_options_.publish_trajectory_node_data_viz) {
      TEMI_LOG(debug) << "PublishGlobalOptimizationTrajectoryData";
      if(calling_navigation_node_to_load_) {
        TEMI_LOG(debug) << "PublishGlobalOptimizationTrajectoryData blocked due to calling navigation map node to load";
        return;
      }
      const auto &node_poses = map_builder_bridge_.GetTrajectoryNodesData();
      generic_lib::TrajectoryNodeList trajectory_nodes_data;
      for (const int trajectory_id : node_poses.trajectory_ids()) {
        TEMI_LOG(trace) << "GetTrajectoryNodesData for trajectory_id: " << trajectory_id;
        generic_lib::TrajectoryNodeEntry new_trajectory_node_entry;
        new_trajectory_node_entry.trajectory_id = trajectory_id;
        // const auto &last_optimized_node = trajectory_id_to_last_optimized_node_id.find(trajectory_id)->second;
        for (const auto &node_id_data : node_poses.trajectory(trajectory_id)) {
          TEMI_LOG(trace) << "Inserting node id: " << node_id_data.id.node_index;
          new_trajectory_node_entry.global_pose = ToGeometryMsgPose(node_id_data.data.global_pose);
          new_trajectory_node_entry.stamp = ToRos(node_id_data.data.time());
          new_trajectory_node_entry.trajectory_node_index = node_id_data.id.node_index;
          trajectory_nodes_data.trajectory_nodes.push_back(new_trajectory_node_entry);
          // if (last_optimized_node == node_id_data.id) {
          //   break;
          // }
        }
      }
      TEMI_LOG(debug) << "Current trajectory nodes size:" << trajectory_nodes_data.trajectory_nodes.size();
      PublishTrajectoryNodesViz(trajectory_nodes_data);
    }
  }
}

void Node::PublishTrajectoryNodesViz(const generic_lib::TrajectoryNodeList &trajectory_nodes_data) {
    visualization_msgs::MarkerArray trajecotry_nodes_viz;
    geometry_msgs::Vector3 scale =
            generic_lib::tools::DrawingTools::scale(0.15, 0.15, 0.15);
    std_msgs::ColorRGBA color =
            generic_lib::tools::DrawingTools::color(50, 178, 255);
    double duration = 5.0;
    for (const generic_lib::TrajectoryNodeEntry &new_trajectory_entry : trajectory_nodes_data.trajectory_nodes) {
        std_msgs::Header trajectory_header;
        trajectory_header.stamp = new_trajectory_entry.stamp;
        trajectory_header.frame_id = MAP_FRAME;
        std::string trajectory_text = std::to_string(new_trajectory_entry.trajectory_node_index)
                + " , " + std::to_string(new_trajectory_entry.global_pose.position.x) + " , " + std::to_string(new_trajectory_entry.global_pose.position.y);
        trajecotry_nodes_viz.markers.push_back(generic_lib::tools::DrawingTools::addTextMarker(trajectory_text,
                                                                                               new_trajectory_entry.trajectory_node_index,
                                                                                               new_trajectory_entry.global_pose,
                                                                                               scale,
                                                                                               color,
                                                                                               trajectory_header,
                                                                                               std::to_string(
                                                                                                   new_trajectory_entry.trajectory_node_index),
                                                                                               duration));
    }
    trajectory_node_data_viz_publisher_.publish(trajecotry_nodes_viz);
}

void Node::PublishLandmarkPosesList(
    const ::ros::WallTimerEvent& unused_timer_event) {
  if (landmark_poses_list_publisher_.getNumSubscribers() > 0) {
    absl::MutexLock lock(&mutex_);
    landmark_poses_list_publisher_.publish(
        map_builder_bridge_.GetLandmarkPosesList());
  }
}

void Node::PublishConstraintList(
    const ::ros::WallTimerEvent& unused_timer_event) {
  if (constraint_list_publisher_.getNumSubscribers() > 0) {
    absl::MutexLock lock(&mutex_);
    constraint_list_publisher_.publish(map_builder_bridge_.GetConstraintList());
  }
}

std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>
Node::ComputeExpectedSensorIds(const TrajectoryOptions& options) const {
  using SensorId = cartographer::mapping::TrajectoryBuilderInterface::SensorId;
  using SensorType = SensorId::SensorType;
  std::set<SensorId> expected_topics;
  // Subscribe to all laser scan, multi echo laser scan, and point cloud topics.
  for (const std::string& topic :
       ComputeRepeatedTopicNames(kLaserScanTopic, options.num_laser_scans)) {
    expected_topics.insert(SensorId{SensorType::RANGE, topic});
  }
  for (const std::string& topic : ComputeRepeatedTopicNames(
           kMultiEchoLaserScanTopic, options.num_multi_echo_laser_scans)) {
    expected_topics.insert(SensorId{SensorType::RANGE, topic});
  }
  for (const std::string& topic :
       ComputeRepeatedTopicNames(kPointCloud2Topic, options.num_point_clouds)) {
    expected_topics.insert(SensorId{SensorType::RANGE, topic});
  }
  // For 2D SLAM, subscribe to the IMU if we expect it. For 3D SLAM, the IMU is
  // required.
  if (node_options_.map_builder_options.use_trajectory_builder_3d() ||
      (node_options_.map_builder_options.use_trajectory_builder_2d() &&
       options.trajectory_builder_options.trajectory_builder_2d_options()
           .use_imu_data())) {
    expected_topics.insert(SensorId{SensorType::IMU, kImuTopic});
  }
  // Odometry is optional.
  if (options.use_odometry) {
    expected_topics.insert(SensorId{SensorType::ODOMETRY, kOdometryTopic});
  }
  // NavSatFix is optional.
  if (options.use_nav_sat) {
    expected_topics.insert(
        SensorId{SensorType::FIXED_FRAME_POSE, kNavSatFixTopic});
  }
  // Landmark is optional.
  if (options.use_landmarks) {
    expected_topics.insert(SensorId{SensorType::LANDMARK, kLandmarkTopic});
  }
  return expected_topics;
}

int Node::AddTrajectory(const TrajectoryOptions& options) {
  const std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>
      expected_sensor_ids = ComputeExpectedSensorIds(options);
  const int trajectory_id =
      map_builder_bridge_.AddTrajectory(expected_sensor_ids, options);
  AddExtrapolator(trajectory_id, options);
  AddSensorSamplers(trajectory_id, options);
  LaunchSubscribers(options, trajectory_id);
#ifdef ROS_MSG_SYS
  wall_timers_.push_back(GenericWallTimer(
      kTopicMismatchCheckDelaySec,
      &Node::MaybeWarnAboutTopicMismatch, this, /*oneshot=*/true));
#endif
  for (const auto& sensor_id : expected_sensor_ids) {
    subscribed_topics_.insert(sensor_id.id);
  }
  trajectory_options_[trajectory_id] = options;
  current_trajectory_id_ = trajectory_id;
  TEMI_LOG(info) << "AddTrajectory: " << current_trajectory_id_;
  return trajectory_id;
}

void Node::LaunchSubscribers(const TrajectoryOptions& options,
                             const int trajectory_id) {
  for (const std::string& topic :
       ComputeRepeatedTopicNames(kLaserScanTopic, options.num_laser_scans)) {
    subscribers_[trajectory_id].push_back(
        SubscribeWithHandler<sensor_msgs::LaserScan>(
            &Node::HandleLaserScanMessage, trajectory_id, topic));
  }
  for (const std::string& topic : ComputeRepeatedTopicNames(
           kMultiEchoLaserScanTopic, options.num_multi_echo_laser_scans)) {
    subscribers_[trajectory_id].push_back(
        SubscribeWithHandler<sensor_msgs::MultiEchoLaserScan>(
            &Node::HandleMultiEchoLaserScanMessage, trajectory_id, topic));
  }
  for (const std::string& topic :
       ComputeRepeatedTopicNames(kPointCloud2Topic, options.num_point_clouds)) {
    subscribers_[trajectory_id].push_back(
        SubscribeWithHandler<sensor_msgs::PointCloud2>(
            &Node::HandlePointCloud2Message, trajectory_id, topic));
  }

  // For 2D SLAM, subscribe to the IMU if we expect it. For 3D SLAM, the IMU is
  // required.
  if (node_options_.map_builder_options.use_trajectory_builder_3d() ||
      (node_options_.map_builder_options.use_trajectory_builder_2d() &&
       options.trajectory_builder_options.trajectory_builder_2d_options()
           .use_imu_data())) {
    subscribers_[trajectory_id].push_back(
        SubscribeWithHandler<sensor_msgs::Imu>(&Node::HandleImuMessage,
                                               trajectory_id, kImuTopic));
  }

  if (options.use_odometry) {
    subscribers_[trajectory_id].push_back(
        SubscribeWithHandler<nav_msgs::Odometry>(&Node::HandleOdometryMessage,
                                                  trajectory_id, kOdometryTopic));
  }
  if (options.use_nav_sat) {
    subscribers_[trajectory_id].push_back(
        SubscribeWithHandler<sensor_msgs::NavSatFix>(
            &Node::HandleNavSatFixMessage, trajectory_id, kNavSatFixTopic));
  }
  if (options.use_landmarks) {
    subscribers_[trajectory_id].push_back(
        SubscribeWithHandler<cartographer_ros_msgs::LandmarkList>(
            &Node::HandleLandmarkMessage, trajectory_id, kLandmarkTopic));
  }
}

bool Node::ValidateTrajectoryOptions(const TrajectoryOptions& options) {
  if (node_options_.map_builder_options.use_trajectory_builder_2d()) {
    return options.trajectory_builder_options
        .has_trajectory_builder_2d_options();
  }
  if (node_options_.map_builder_options.use_trajectory_builder_3d()) {
    return options.trajectory_builder_options
        .has_trajectory_builder_3d_options();
  }
  return false;
}

bool Node::ValidateTopicNames(const TrajectoryOptions& options) {
  for (const auto& sensor_id : ComputeExpectedSensorIds(options)) {
    const std::string& topic = sensor_id.id;
    if (subscribed_topics_.count(topic) > 0) {
      LOG(ERROR) << "Topic name [" << topic << "] is already used.";
      return false;
    }
  }
  return true;
}

cartographer_ros_msgs::StatusResponse Node::TrajectoryStateToStatus(
    const int trajectory_id, const std::set<TrajectoryState>& valid_states) {
  const auto trajectory_states = map_builder_bridge_.GetTrajectoryStates();
  cartographer_ros_msgs::StatusResponse status_response;

  const auto it = trajectory_states.find(trajectory_id);
  if (it == trajectory_states.end()) {
    status_response.message =
        absl::StrCat("Trajectory ", trajectory_id, " doesn't exist.");
    status_response.code = cartographer_ros_msgs::StatusCode::NOT_FOUND;
    return status_response;
  }

  status_response.message =
      absl::StrCat("Trajectory ", trajectory_id, " is in '",
                   TrajectoryStateToString(it->second), "' state.");
  status_response.code =
      valid_states.count(it->second)
          ? cartographer_ros_msgs::StatusCode::OK
          : cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
  return status_response;
}

cartographer_ros_msgs::StatusResponse Node::FinishTrajectoryUnderLock(
    const int trajectory_id) {
  cartographer_ros_msgs::StatusResponse status_response;
  if (trajectories_scheduled_for_finish_.count(trajectory_id)) {
    status_response.message = absl::StrCat("Trajectory ", trajectory_id,
                                           " already pending to finish.");
    status_response.code = cartographer_ros_msgs::StatusCode::OK;
    LOG(INFO) << status_response.message;
    return status_response;
  }

  // First, check if we can actually finish the trajectory.
  status_response = TrajectoryStateToStatus(
      trajectory_id, {TrajectoryState::ACTIVE} /* valid states */);
  if (status_response.code != cartographer_ros_msgs::StatusCode::OK) {
    LOG(ERROR) << "Can't finish trajectory: " << status_response.message;
    return status_response;
  }

  // Shutdown the subscribers of this trajectory.
  // A valid case with no subscribers is e.g. if we just visualize states.
  if (subscribers_.count(trajectory_id)) {
    for (auto& entry : subscribers_[trajectory_id]) {
      entry->shutdown();
      subscribed_topics_.erase(entry->topic);
      LOG(INFO) << "Shutdown the subscriber of [" << entry->topic << "]";
    }
    CHECK_EQ(subscribers_.erase(trajectory_id), 1);
  }
  absl::MutexLock lock(&mutex_);
  map_builder_bridge_.FinishTrajectory(trajectory_id);
  trajectories_scheduled_for_finish_.emplace(trajectory_id);
  status_response.message =
      absl::StrCat("Finished trajectory ", trajectory_id, ".");
  status_response.code = cartographer_ros_msgs::StatusCode::OK;
  return status_response;
}

bool Node::HandleStartTrajectory(
    ::cartographer_ros_msgs::StartTrajectory::Request& request,
    ::cartographer_ros_msgs::StartTrajectory::Response& response) {
  TrajectoryOptions trajectory_options;
  std::tie(std::ignore, trajectory_options) = LoadOptions(
      request.configuration_directory, request.configuration_basename);

  if (request.use_initial_pose) {
    const auto pose = ToRigid3d(request.initial_pose);
    if (!pose.IsValid()) {
      response.status.message =
          "Invalid pose argument. Orientation quaternion must be normalized.";
      LOG(ERROR) << response.status.message;
      response.status.code =
          cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
      return true;
    }

    // Check if the requested trajectory for the relative initial pose exists.
    response.status = TrajectoryStateToStatus(
        request.relative_to_trajectory_id,
        {TrajectoryState::ACTIVE, TrajectoryState::FROZEN,
         TrajectoryState::FINISHED} /* valid states */);
    if (response.status.code != cartographer_ros_msgs::StatusCode::OK) {
      LOG(ERROR) << "Can't start a trajectory with initial pose: "
                 << response.status.message;
      return true;
    }

    ::cartographer::mapping::proto::InitialTrajectoryPose
        initial_trajectory_pose;
    initial_trajectory_pose.set_to_trajectory_id(
        request.relative_to_trajectory_id);
    *initial_trajectory_pose.mutable_relative_pose() =
        cartographer::transform::ToProto(pose);
    initial_trajectory_pose.set_timestamp(cartographer::common::ToUniversal(
        ::cartographer_ros::FromRos(ros::Time(0))));
    *trajectory_options.trajectory_builder_options
         .mutable_initial_trajectory_pose() = initial_trajectory_pose;
  }

  if (!ValidateTrajectoryOptions(trajectory_options)) {
    response.status.message = "Invalid trajectory options.";
    LOG(ERROR) << response.status.message;
    response.status.code = cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
  } else if (!ValidateTopicNames(trajectory_options)) {
    response.status.message = "Topics are already used by another trajectory.";
    LOG(ERROR) << response.status.message;
    response.status.code = cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
  } else {
    response.status.message = "Success.";
    response.trajectory_id = AddTrajectory(trajectory_options);
    response.status.code = cartographer_ros_msgs::StatusCode::OK;
  }
  return true;
}

void Node::StartTrajectoryWithDefaultTopics(const TrajectoryOptions& options) {
  absl::MutexLock lock(&mutex_);
  CHECK(ValidateTrajectoryOptions(options));
  AddTrajectory(options);
}

std::vector<
    std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>>
Node::ComputeDefaultSensorIdsForMultipleBags(
    const std::vector<TrajectoryOptions>& bags_options) const {
  using SensorId = cartographer::mapping::TrajectoryBuilderInterface::SensorId;
  std::vector<std::set<SensorId>> bags_sensor_ids;
  for (size_t i = 0; i < bags_options.size(); ++i) {
    std::string prefix;
    if (bags_options.size() > 1) {
      prefix = "bag_" + std::to_string(i + 1) + "_";
    }
    std::set<SensorId> unique_sensor_ids;
    for (const auto& sensor_id : ComputeExpectedSensorIds(bags_options.at(i))) {
      unique_sensor_ids.insert(SensorId{sensor_id.type, prefix + sensor_id.id});
    }
    bags_sensor_ids.push_back(unique_sensor_ids);
  }
  return bags_sensor_ids;
}

int Node::AddOfflineTrajectory(
    const std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>&
        expected_sensor_ids,
    const TrajectoryOptions& options) {
  absl::MutexLock lock(&mutex_);
  const int trajectory_id =
      map_builder_bridge_.AddTrajectory(expected_sensor_ids, options);
  AddExtrapolator(trajectory_id, options);
  AddSensorSamplers(trajectory_id, options);
  current_trajectory_id_ = trajectory_id;
  TEMI_LOG(info) << "AddOfflineTrajectory: " << current_trajectory_id_;
  return trajectory_id;
}

bool Node::HandleGetTrajectoryStates(
    ::cartographer_ros_msgs::GetTrajectoryStates::Request& request,
    ::cartographer_ros_msgs::GetTrajectoryStates::Response& response) {
  using TrajectoryState =
      ::cartographer::mapping::PoseGraphInterface::TrajectoryState;
  absl::MutexLock lock(&mutex_);
  response.status.code = ::cartographer_ros_msgs::StatusCode::OK;
  response.trajectory_states.header.stamp = ros::Time::now();
  for (const auto& entry : map_builder_bridge_.GetTrajectoryStates()) {
    response.trajectory_states.trajectory_id.push_back(entry.first);
    switch (entry.second) {
      case TrajectoryState::ACTIVE:
        response.trajectory_states.trajectory_state.push_back(
            ::cartographer_ros_msgs::TrajectoryStates::ACTIVE);
        break;
      case TrajectoryState::FINISHED:
        response.trajectory_states.trajectory_state.push_back(
            ::cartographer_ros_msgs::TrajectoryStates::FINISHED);
        break;
      case TrajectoryState::FROZEN:
        response.trajectory_states.trajectory_state.push_back(
            ::cartographer_ros_msgs::TrajectoryStates::FROZEN);
        break;
      case TrajectoryState::DELETED:
        response.trajectory_states.trajectory_state.push_back(
            ::cartographer_ros_msgs::TrajectoryStates::DELETED);
        break;
    }
  }
  return true;
}

bool Node::HandleFinishTrajectory(
    ::cartographer_ros_msgs::FinishTrajectory::Request& request,
    ::cartographer_ros_msgs::FinishTrajectory::Response& response) {
  // absl::MutexLock lock(&mutex_);
  response.status = FinishTrajectoryUnderLock(request.trajectory_id);
  return true;
}

bool Node::HandleWriteState(
    ::cartographer_ros_msgs::WriteState::Request& request,
    ::cartographer_ros_msgs::WriteState::Response& response) {
  absl::MutexLock lock(&mutex_);
  if (map_builder_bridge_.SerializeState(request.filename,
                                         request.include_unfinished_submaps)) {
    response.status.code = cartographer_ros_msgs::StatusCode::OK;
    response.status.message =
        absl::StrCat("State written to '", request.filename, "'.");
  } else {
    response.status.code = cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
    response.status.message =
        absl::StrCat("Failed to write '", request.filename, "'.");
  }
  return true;
}

bool Node::HandleReadMetrics(
    ::cartographer_ros_msgs::ReadMetrics::Request& request,
    ::cartographer_ros_msgs::ReadMetrics::Response& response) {
  absl::MutexLock lock(&mutex_);
  response.timestamp = ros::Time::now();
  if (!metrics_registry_) {
    response.status.code = cartographer_ros_msgs::StatusCode::UNAVAILABLE;
    response.status.message = "Collection of runtime metrics is not activated.";
    return true;
  }
  metrics_registry_->ReadMetrics(&response);
  response.status.code = cartographer_ros_msgs::StatusCode::OK;
  response.status.message = "Successfully read metrics.";
  return true;
}

void Node::FinishAllTrajectories() {
  // absl::MutexLock lock(&mutex_);
  for (const auto& entry : map_builder_bridge_.GetTrajectoryStates()) {
    if (entry.second == TrajectoryState::ACTIVE) {
      const int trajectory_id = entry.first;
      CHECK_EQ(FinishTrajectoryUnderLock(trajectory_id).code,
               cartographer_ros_msgs::StatusCode::OK);
    }
  }
}

bool Node::FinishTrajectory(const int trajectory_id) {
  // absl::MutexLock lock(&mutex_);
  return FinishTrajectoryUnderLock(trajectory_id).code ==
         cartographer_ros_msgs::StatusCode::OK;
}

void Node::RunFinalOptimization() {
  {
    for (const auto& entry : map_builder_bridge_.GetTrajectoryStates()) {
      const int trajectory_id = entry.first;
      if (entry.second == TrajectoryState::ACTIVE) {
        LOG(WARNING)
            << "Can't run final optimization if there are one or more active "
               "trajectories. Trying to finish trajectory with ID "
            << std::to_string(trajectory_id) << " now.";
        CHECK(FinishTrajectory(trajectory_id))
            << "Failed to finish trajectory with ID "
            << std::to_string(trajectory_id) << ".";
      }
    }
  }
  // Assuming we are not adding new data anymore, the final optimization
  // can be performed without holding the mutex.
  map_builder_bridge_.RunFinalOptimization();
}

void Node::HandleOdometryMessage(const int trajectory_id,
                                 const std::string& sensor_id,
                                 const nav_msgs::Odometry::ConstPtr& msg) {
  absl::MutexLock lock(&mutex_);
  if (!sensor_samplers_.at(trajectory_id).odometry_sampler.Pulse()) {
    return;
  }
  auto sensor_bridge_ptr = map_builder_bridge_.sensor_bridge(trajectory_id);
  auto odometry_data_ptr = sensor_bridge_ptr->ToOdometryData(msg);
  if (odometry_data_ptr != nullptr) {
    extrapolators_.at(trajectory_id).AddOdometryData(*odometry_data_ptr);
  }
  sensor_bridge_ptr->HandleOdometryMessage(sensor_id, msg);
}

void Node::HandleNavSatFixMessage(const int trajectory_id,
                                  const std::string& sensor_id,
                                  const sensor_msgs::NavSatFix::ConstPtr& msg) {
  absl::MutexLock lock(&mutex_);
  if (!sensor_samplers_.at(trajectory_id).fixed_frame_pose_sampler.Pulse()) {
    return;
  }
  map_builder_bridge_.sensor_bridge(trajectory_id)
      ->HandleNavSatFixMessage(sensor_id, msg);
}

void Node::HandleLandmarkMessage(
    const int trajectory_id, const std::string& sensor_id,
    const cartographer_ros_msgs::LandmarkList::ConstPtr& msg) {
  absl::MutexLock lock(&mutex_);
  if (!sensor_samplers_.at(trajectory_id).landmark_sampler.Pulse()) {
    return;
  }
  map_builder_bridge_.sensor_bridge(trajectory_id)
      ->HandleLandmarkMessage(sensor_id, msg);
}

void Node::HandleImuMessage(const int trajectory_id,
                            const std::string& sensor_id,
                            const sensor_msgs::Imu::ConstPtr& msg) {
  absl::MutexLock lock(&mutex_);
  try {
  if (!sensor_samplers_.at(trajectory_id).imu_sampler.Pulse()) {
    return;
  }
  auto sensor_bridge_ptr = map_builder_bridge_.sensor_bridge(trajectory_id);
  auto imu_data_ptr = sensor_bridge_ptr->ToImuData(msg);
  if (imu_data_ptr != nullptr) {
    extrapolators_.at(trajectory_id).AddImuData(*imu_data_ptr);
  }
  sensor_bridge_ptr->HandleImuMessage(sensor_id, msg);
  } catch (SlamException& err) {
    err.HandleException(health_monitor_manager_);
  }
}

void Node::HandleLaserScanMessage(const int trajectory_id,
                                  const std::string& sensor_id,
                                  const sensor_msgs::LaserScan::ConstPtr& msg) {                                 
  absl::MutexLock lock(&mutex_);//线程安全
  scan_stable_ = msg->ranges.size() > kMinScanStableRanges && msg->ranges.size() < kMaxScanStableRanges;
  //不希望每一帧数据都去处理，所以需要一个采样器
  try {
  if (!sensor_samplers_.at(trajectory_id).rangefinder_sampler.Pulse()) {
    return;
  }
  map_builder_bridge_.sensor_bridge(trajectory_id)
      ->HandleLaserScanMessage(sensor_id, msg);
  } catch (SlamException& err) {
    err.HandleException(health_monitor_manager_);
  }
}

void Node::HandleMultiEchoLaserScanMessage(
    const int trajectory_id, const std::string& sensor_id,
    const sensor_msgs::MultiEchoLaserScan::ConstPtr& msg) {
  absl::MutexLock lock(&mutex_);
  if (!sensor_samplers_.at(trajectory_id).rangefinder_sampler.Pulse()) {
    return;
  }
  map_builder_bridge_.sensor_bridge(trajectory_id)
      ->HandleMultiEchoLaserScanMessage(sensor_id, msg);
}

void Node::HandlePointCloud2Message(
    const int trajectory_id, const std::string& sensor_id,
    const sensor_msgs::PointCloud2::ConstPtr& msg) {
  absl::MutexLock lock(&mutex_);
  if (!sensor_samplers_.at(trajectory_id).rangefinder_sampler.Pulse()) {
    return;
  }
  map_builder_bridge_.sensor_bridge(trajectory_id)
      ->HandlePointCloud2Message(sensor_id, msg);
}

bool Node::HandleCommand(generic_lib::SlamInterface::Request& request,
                         generic_lib::SlamInterface::Response& response) {
  if (low_power_mode_ &&
      request.action != generic_lib::SlamInterface::Request::SAVE_STATE) {
    TEMI_LOG(info) << "In low power mode, try wait here.";
    absl::MutexLock lock(&mutex_);
    const auto predicate = [this]() { return (!low_power_mode_); };
    if (!mutex_.AwaitWithTimeout(
            absl::Condition(&predicate),
            absl::FromChrono(cartographer::common::FromSeconds(3.)))) {
      TEMI_LOG(info) << "Still in low power mode after timeout";

      response.status =
          generic_lib::StatusCode::CURRENTLY_LPM_TRY_AGAIN;  // RM WILL ASK
                                                             // AGAIN TO LOAD
      TEMI_LOG(error) << "Cannot change SLAM state while on low power mode, "
                         "rejecting request.";
      return true;
    }
  }
    generic_lib::SlamInterface navigation_interface_srv;
    navigation_interface_srv.request = request;
    switch (request.action) {
      case generic_lib::SlamInterface::Request::RESET_STATE: {
        TEMI_LOG(info) << "RESET_STATE request";
        // Delete pbstream file
          std::string user_home_path = generic_lib::tools::get_user_home_folder();
        std::string pbstream_full_file_path =
            user_home_path + SAVED_SLAM_FILE_PATH;
        generic_lib::tools::delete_file(pbstream_full_file_path);
        FinishAllTrajectories();
        TEMI_LOG(info) << "RESET_STATE request FinishAllTrajectories";
        Reset(false, node_options_);
        TEMI_LOG(info) << "RESET_STATE request Reset";
        StartTrajectoryWithDefaultTopics(default_trajectory_options_);
        TEMI_LOG(info) << "StartTrajectoryWithDefaultTopics.";
        if (!navigation_client_.call(navigation_interface_srv.request, navigation_interface_srv.response) ||
            navigation_interface_srv.response.status !=
                generic_lib::StatusCode::OK) {
          TEMI_LOG(error) << "Error reseting navigation node stat. Response "
                             "from navigation map node: "
                          << static_cast<int>(navigation_interface_srv.response.status);
          response.status = generic_lib::StatusCode::CANCELLED;
          return false;
        } else {
          TEMI_LOG(info)
              << "Navigation node state reset successfuly. Response from navigation map node: "
              << static_cast<int>(navigation_interface_srv.response.status);
        }
        map_builder_bridge_.ValidateReadyAfterReset();
        health_monitor_manager_.clear_event(EVNT_SLAM_LOST_MAP);
        health_monitor_manager_.clear_event(EVNT_SLAM_CORRUPTED_PBSTREAM);
        health_monitor_manager_.clear_event(EVNT_SLAM_PURE_LOCALIZATION);
        response.status = generic_lib::StatusCode::OK;
        syscommand_manager_.publish_msg(SLAM_POSITION_CHANGED_UPDATE, RESET_COMMAND);
        break;
      }
      case generic_lib::SlamInterface::Request::LOAD_STATE_NORMAL: {
        TEMI_LOG(info) << "LOAD_STATE_NORMAL request";
        HandleLoadState(request, response, false);
        break;
      }
      case generic_lib::SlamInterface::Request::LOAD_STATE_PURE: {
        TEMI_LOG(info) << "LOAD_STATE_PURE request";
        HandleLoadState(request, response, true);
        break;
      }
      case generic_lib::SlamInterface::Request::SAVE_STATE: {
        TEMI_LOG(info) << "SAVE_STATE request";
        absl::MutexLock lock(&mutex_);
        if (trajectory_options_.at(current_trajectory_id_).trajectory_builder_options.has_pure_localization_trimmer()) {
          TEMI_LOG(error) << "Calling save in pure localization state!";
          response.status = generic_lib::StatusCode::INVALID_ARGUMENT;
          break;
        }
        if (map_builder_bridge_.SerializeState(request.filename, true /* include_unfinished_submaps */)) {
          std::string user_home_path = generic_lib::tools::get_user_home_folder();
          navigation_interface_srv.request.filename = user_home_path + SAVED_MAPPING_FILE_PATH;
          if (!navigation_client_.call(navigation_interface_srv.request, navigation_interface_srv.response) ||
              navigation_interface_srv.response.status !=
                  generic_lib::StatusCode::OK) {
            TEMI_LOG(error) << "Error saving navigation node state. Response "
                               "from navigation map node: "
                            << static_cast<int>(navigation_interface_srv.response.status);
            response.status = generic_lib::StatusCode::CANCELLED;
            return false;
          } else {
            TEMI_LOG(info) << "Navigation node state saved successfuly. "
                              "Response from navigation map node: "
                           << static_cast<int>(navigation_interface_srv.response.status);
          }
          response.status = generic_lib::StatusCode::OK;
        } else {
          response.status = generic_lib::StatusCode::INVALID_ARGUMENT;
        }
        break;
      }
      case generic_lib::SlamInterface::Request::RELOCALIZE_STATE: {
        try {
          TEMI_LOG(info) << "RELOCALIZE_STATE request, current trajectory id : " << current_trajectory_id_;
          if (true/*trajectory_options_.at(current_trajectory_id_).trajectory_builder_options.has_pure_localization_trimmer()*/)
          {
            HandleReLocalization(request, response);
          } else {
            TEMI_LOG(error) << "Requested action is unavailable(current slam mode), rejecting it";
            response.status = generic_lib::StatusCode::UNAVAILABLE;
          }

        } catch (const std::exception & err)
        {
          TEMI_LOG(error) << "Requested action caused exception: " << err.what();
        }
        break;
      }
      default: {
        response.status = generic_lib::StatusCode::UNAVAILABLE;
        TEMI_LOG(error) << "Requested action is unavailable, rejecting it";
      }
    }
    return true;
}

void Node::HandleLoadState(generic_lib::SlamInterface::Request& request,
                           generic_lib::SlamInterface::Response& response, bool pure_localization) {
  std::string user_home_path = generic_lib::tools::get_user_home_folder();
  std::string pbstream_full_file_path = user_home_path + SAVED_SLAM_FILE_PATH;
  if (generic_lib::tools::is_file_exists(pbstream_full_file_path)) {
    boost::filesystem::path path_to_slam_state_pbstream{pbstream_full_file_path};
    boost::uintmax_t slam_state_pbstream_filesize = boost::filesystem::file_size(path_to_slam_state_pbstream);
    if(slam_state_pbstream_filesize > kMaxAllowedSlamStatePbstreamFileSize) { // larger then 100MB
        health_monitor_manager_.set_event(EVNT_SLAM_LARGE_PBSTREAM_FILE);
    }
    FinishAllTrajectories();
    TrajectoryOptions trajectory_options = default_trajectory_options_;
    NodeOptions node_options = node_options_;
    if (pure_localization) {
      std::tie(node_options, trajectory_options) =
          LoadOptions(user_home_path + SLAM_CONF_DIR, PURE_LOCALIZATION_LUA_FILE);
    }
    Reset(pure_localization, node_options);
    if (!LoadState(pbstream_full_file_path, pure_localization)) {
      response.status = generic_lib::StatusCode::FAILED_DUE_TO_CORRUPT_PBSTREAM;
      health_monitor_manager_.set_event(EVNT_SLAM_CORRUPTED_PBSTREAM);
      return;
    }
    TEMI_LOG(info) << "Slam finished loading pbstream, calling navigation node to load";
    generic_lib::SlamInterface navigation_interface_srv;
    navigation_interface_srv.request = request;
    calling_navigation_node_to_load_ = true;
    if (!navigation_client_.call(navigation_interface_srv.request, navigation_interface_srv.response) ||
        navigation_interface_srv.response.status !=
            generic_lib::StatusCode::OK) {
      TEMI_LOG(error) << "Error loading navigation map state. Response from "
                         "navigation map node: "
                      << static_cast<int>(
                             navigation_interface_srv.response.status);
    } else {
      TEMI_LOG(info)
          << "Navigation map loaded succesfuly. Response from Navigation map node: "
          << static_cast<int>(navigation_interface_srv.response.status);
    }
    calling_navigation_node_to_load_ = false;
    // More information here: https://github.com/googlecartographer/cartographer_ros/issues/913
    ::cartographer::transform::Rigid3d pose_3d = ToRigid3d(request.initial_trajectory_pose);
    if (pose_3d.IsValid()) {
      ::cartographer::mapping::proto::InitialTrajectoryPose init_pose;
      init_pose.set_to_trajectory_id(0);
      *init_pose.mutable_relative_pose() = ::cartographer::transform::ToProto(pose_3d);
      init_pose.set_timestamp(::cartographer::common::ToUniversal(FromRos(::ros::Time(0))));
      // convert the global pose to relative trjectory pose
      ConvertGlobalPoseToTrajectoryPose(pose_3d, init_pose);
      *trajectory_options.trajectory_builder_options.mutable_initial_trajectory_pose() = init_pose;
    } else {
      TEMI_LOG(error) << "Got an invalid initial pose: "<<pose_3d.DebugString();
    }
    StartTrajectoryWithDefaultTopics(trajectory_options);
    TEMI_LOG(info) << "StartTrajectoryWithDefaultTopics.";
    map_builder_bridge_.ValidateReadyAfterReset();
    if (pure_localization) {
      health_monitor_manager_.set_event(EVNT_SLAM_PURE_LOCALIZATION);
    } else {
      health_monitor_manager_.clear_event(EVNT_SLAM_PURE_LOCALIZATION);
    }
    health_monitor_manager_.clear_event(EVNT_SLAM_LOST_MAP);
    syscommand_manager_.publish_msg(SLAM_POSITION_CHANGED_UPDATE, LOAD_COMMAND);
    response.status = generic_lib::StatusCode::OK;
  } else {
    response.status =
        generic_lib::StatusCode::FAILED_DUE_TO_PBSTREAM_NOT_EXIST;
  }
}

void Node::ConvertGlobalPoseToTrajectoryPose(const ::cartographer::transform::Rigid3d global_pose, ::cartographer::mapping::proto::InitialTrajectoryPose& trajectory_relevant_pose)
{
  const auto &node_poses = map_builder_bridge_.GetTrajectoryNodesData();
  TEMI_LOG(debug) << "ConvertGlobalPoseToTrajectoryPose";
  double current_min_distance = std::numeric_limits<double>::infinity();
  int relative_trajectory_id = -1;
  ::cartographer::transform::Rigid3d relative_trajectory_node_global_pose;
  
  ::cartographer::common::Time relative_trajectory_node_time;
  int relative_trajectory_node_index;
  const auto trajectory_states = map_builder_bridge_.GetTrajectoryStates();

  for (const int trajectory_id : node_poses.trajectory_ids()) {
    const auto it = trajectory_states.find(trajectory_id);
    TEMI_LOG(info) << "ConvertGlobalPoseToTrajectoryPose, trajectory " << trajectory_id << ": " << TrajectoryStateToString(it->second);
    if (it->second == TrajectoryState::FINISHED) {
      TEMI_LOG(info) << "ConvertGlobalPoseToTrajectoryPose skip trajectory " << trajectory_id;
      continue;
    }
    for (const auto &node_id_data : node_poses.trajectory(trajectory_id)) {
      ::cartographer::transform::Rigid3d
        current_trajectory_node_global_pose = node_id_data.data.global_pose;
      double node_distance = (current_trajectory_node_global_pose.translation()
                          - global_pose.translation()).norm();
      // get a nearest neighbor, TODO: try speed this up maybe?
      if (node_distance < current_min_distance)
      {
        current_min_distance = node_distance;
        relative_trajectory_id = trajectory_id;
        relative_trajectory_node_global_pose = current_trajectory_node_global_pose;
        relative_trajectory_node_time = node_id_data.data.time();
        relative_trajectory_node_index = node_id_data.id.node_index;
        // maybe consider break or return here directly?
      }
    }

    // only search for one trajectory, most likely this is the loaded one
    if (relative_trajectory_id != -1) {
      trajectory_relevant_pose.set_to_trajectory_id(relative_trajectory_id);
      *trajectory_relevant_pose.mutable_relative_pose() = ::cartographer::transform::ToProto(relative_trajectory_node_global_pose.inverse() * global_pose);
      trajectory_relevant_pose.set_timestamp(::cartographer::common::ToUniversal(relative_trajectory_node_time));
      TEMI_LOG(info) << "Found closest trajectory node: "<<relative_trajectory_id << ", " << relative_trajectory_node_index << "; distance: " << current_min_distance;
      return;
    }
  }

  TEMI_LOG(fatal) << "Failed in ConvertGlobalPoseToTrajectoryPose";

}

void Node::SerializeState(const std::string& filename,
                          const bool include_unfinished_submaps) {
  absl::MutexLock lock(&mutex_);
  CHECK(
      map_builder_bridge_.SerializeState(filename, include_unfinished_submaps))
      << "Could not write state.";
}

bool Node::LoadState(const std::string& state_filename,
                     const bool load_frozen_state) {
  absl::MutexLock lock(&mutex_);
  if (!map_builder_bridge_.LoadState(state_filename, load_frozen_state)) {
    return false;
  } else {
    return true;
  }
}

void Node::Reset(bool pure_localization, NodeOptions node_options) {
  absl::MutexLock lock(&mutex_);
  TEMI_LOG(info) << "Node::Reset call map_builder_bridge_ to reset.";
  map_builder_bridge_.Reset(pure_localization, node_options);
  TEMI_LOG(info) << "Node::Reset start clearing other resources.";
  sensor_samplers_.clear();
  extrapolators_.clear();
  last_added_pose_.clear();
  trajectory_options_.clear();
  trajectories_scheduled_for_finish_.clear(); //Anders: this is important
}

#ifdef ROS_MSG_SYS
void Node::MaybeWarnAboutTopicMismatch(
    const ::ros::WallTimerEvent& unused_timer_event) {
  ::ros::master::V_TopicInfo ros_topics;
  ::ros::master::getTopics(ros_topics);
  std::set<std::string> published_topics;
  std::stringstream published_topics_string;
  for (const auto& it : ros_topics) {
    std::string resolved_topic = GenericBase::nh_->resolveName(it.name, false);
    published_topics.insert(resolved_topic);
    published_topics_string << resolved_topic << ",";
  }
  bool print_topics = false;
  for (const auto& entry : subscribers_) {
    int trajectory_id = entry.first;
    for (const auto& subscriber : entry.second) {
      std::string resolved_topic = GenericBase::nh_->resolveName(subscriber->topic);
      if (published_topics.count(resolved_topic) == 0) {
        LOG(WARNING) << "Expected topic \"" << subscriber->topic
                     << "\" (trajectory " << trajectory_id << ")"
                     << " (resolved topic \"" << resolved_topic << "\")"
                     << " but no publisher is currently active.";
        print_topics = true;
      }
    }
  }
  if (print_topics) {
    LOG(WARNING) << "Currently available topics are: "
                 << published_topics_string.str();
  }
}
#endif

void Node::syscommand_callback(
    const generic_lib::nodeCommand::ConstPtr& command) {
    if ((command->command.compare(CMD_LOW_POWER_MODE_START) == 0) ||
        (command->command.compare(CMD_LOW_POWER_MODE_STOP) == 0)) {
      low_power_mode_routine(command->command, command->subCommand);
    } else if (command->command == RE_INIT_LOGGER) {
      init_logger(node_name_);
      TEMI_LOG(info) << "init_logger " << node_name_;
    } else if (command->command == SLAM_ENTER_LPM_MODE_COMMAND) {
      low_power_mode_routine(CMD_LOW_POWER_MODE_START, command->subCommand);
      slam_entered_lpm_special_mode_ = true;
    } else if (command->command == SLAM_EXIT_LPM_MODE_COMMAND) {
      slam_entered_lpm_special_mode_ = false;
      low_power_mode_routine(CMD_LOW_POWER_MODE_STOP, command->subCommand);
    }
}

void Node::HandleReLocalization(generic_lib::SlamInterface::Request& request,
      generic_lib::SlamInterface::Response& response){
  auto command_pose_3d = ToRigid3d(request.initial_trajectory_pose);
  if (low_power_mode_ || !command_pose_3d.IsValid()) {
    TEMI_LOG(info) << "failed, LPM: " << low_power_mode_
                   << ", pose: " << command_pose_3d;
    response.status = generic_lib::StatusCode::UNAVAILABLE;
    return;
  }
  TEMI_LOG(info)<<"re trajectory start and current trajectory id is: "<<current_trajectory_id_;
  if(!FinishTrajectory(current_trajectory_id_)){
    TEMI_LOG(warning)<<"FinishTrajectory failed, the current trajectory is not established now";
  };
  auto original_options = trajectory_options_[current_trajectory_id_];
  ::cartographer::mapping::proto::InitialTrajectoryPose init_pose;
  ConvertGlobalPoseToTrajectoryPose(command_pose_3d, init_pose);
  TEMI_LOG(info) << "Re trajectory procedure use original pose: " << command_pose_3d;
  *original_options.trajectory_builder_options.mutable_initial_trajectory_pose() = init_pose;
  StartTrajectoryWithDefaultTopics(original_options);
  map_builder_bridge_.ValidateReadyAfterReset();
  syscommand_manager_.publish_msg(SLAM_POSITION_CHANGED_UPDATE, RePose_CMD);
  response.status = generic_lib::StatusCode::OK;
  TEMI_LOG(info) << "ReLocalization process finished";
}

void Node::low_power_mode_routine(std::string command, std::string sub_command) {
  absl::MutexLock lock(&mutex_);
  if (slam_entered_lpm_special_mode_) {
    TEMI_LOG(info) << "slam_entered_lpm_special_mode_, ignore general LPM command";
    return;
  }
  if (!low_power_mode_ && (command.compare(CMD_LOW_POWER_MODE_START) == 0)) {
    map_builder_bridge_.PauseAccumulatingPoseGraphSensorData(); // Pause accumulating in both LPM modes
    for (const auto& entry : subscribers_) {
      int trajectory_id = entry.first;
      for (const std::string& topic : ComputeRepeatedTopicNames(
              kLaserScanTopic, trajectory_options_.at(trajectory_id).num_laser_scans)) {
          map_builder_bridge_.sensor_bridge(trajectory_id)->PauseCollating(topic);
        }
    }
    low_power_mode_ = true;
    health_monitor_manager_.send_low_power_mode_start();
  } else if (low_power_mode_ && (command.compare(CMD_LOW_POWER_MODE_STOP) == 0)) {
    const auto predicate = [this]() {
            return (scan_stable_);
        };
    while(!mutex_.AwaitWithTimeout(
        absl::Condition(&predicate),
        absl::FromChrono(cartographer::common::FromSeconds(1.)))) {
        TEMI_LOG(info) << "Scan not stable, waiting for stable scan";
    }
    map_builder_bridge_.ResumeAccumulatingPoseGraphSensorData(); // Resume accumulating in both LPM modes
    for (const auto& entry : subscribers_) {
      int trajectory_id = entry.first;
      for (const std::string& topic : ComputeRepeatedTopicNames(
              kLaserScanTopic, trajectory_options_.at(trajectory_id).num_laser_scans)) {
          map_builder_bridge_.sensor_bridge(trajectory_id)->ResumeCollating(topic);
        }
    }
    low_power_mode_ = false;
    health_monitor_manager_.send_low_power_mode_stop();
  }
}


void Node::init_all_generic_tools() {
  absl::MutexLock lock(&mutex_);
  // Initialize bureaucracy stuff
  health_monitor_manager_.init_params(N_SLAM);
  syscommand_manager_.init_params(&Node::syscommand_callback, this, N_SLAM);
  health_monitor_manager_.clear_event(EVNT_SLAM_PURE_LOCALIZATION);

  // Initialize the publishers
  submap_list_publisher_.init_publisher(kSubmapListTopic,
                                        kLatestOnlyPublisherQueueSize);
  // Visualization stuff.
  if (node_options_.publish_landmark_poses_list) {
    landmark_poses_list_publisher_.init_publisher(
          kLandmarkPosesListTopic, kLatestOnlyPublisherQueueSize);
    wall_timers_.push_back(
          GenericWallTimer(node_options_.trajectory_publish_period_sec,
                           &Node::PublishLandmarkPosesList, this));
  }
  if (node_options_.publish_constraint_list) {
    constraint_list_publisher_.init_publisher(kConstraintListTopic,
                                              kLatestOnlyPublisherQueueSize);
    wall_timers_.push_back(GenericWallTimer(
                             kConstraintPublishPeriodSec,
                             &Node::PublishConstraintList, this));
  }
  if (node_options_.publish_scan_matched_point_cloud) {
    scan_matched_point_cloud_publisher_.init_publisher(kScanMatchedPointCloudTopic,
                                                       kLatestOnlyPublisherQueueSize);
  }
  if (node_options_.publish_trajectory_node_list) {
    trajectory_node_list_publisher_.init_publisher(
          kTrajectoryNodeListTopic, kLatestOnlyPublisherQueueSize);
    wall_timers_.push_back(
          GenericWallTimer(node_options_.trajectory_publish_period_sec,
                           &Node::PublishTrajectoryNodeList, this));
  }
  if (node_options_.publish_tracked_pose) {
    tracked_pose_publisher_.init_publisher(
            kTrackedPoseTopic, kLatestOnlyPublisherQueueSize);
  }
  if (node_options_.publish_trajectory_node_data_viz) {
    trajectory_node_data_viz_publisher_.init_publisher(
        kTrajectoryNodesDataVizTopic, kLatestOnlyPublisherQueueSize);
  }
  // Initialize the timers.
  wall_timers_.push_back(
        GenericWallTimer(node_options_.submap_publish_period_sec,
                         &Node::PublishSubmapList, this));
  wall_timers_.push_back(
        GenericWallTimer(node_options_.pose_publish_period_sec,
                         &Node::PublishLocalTrajectoryData, this));

  // Register a new tf broadcaster.
  tf_broadcaster_.init_tf(default_trajectory_options_.odom_frame,
                          node_options_.map_frame, TF_PORT_MAP_ODOM);
  if (default_trajectory_options_.provide_odom_frame){
    tf_broadcaster_odom_.init_tf(default_trajectory_options_.published_frame,
                          default_trajectory_options_.odom_frame, TF_PORT_ODOM_WHEEL);
  } else {
    add_dynamic_tf_to_node(ODOM_FRAME, WHEEL_FRAME);
  }

  // Initialize the services
  service_servers_.push_back(GenericServiceServer(
                               kSubmapQueryServiceName, &Node::HandleSubmapQuery, this));
  service_servers_.push_back(GenericServiceServer(
                               kSlamInterfaceServiceName, &Node::HandleCommand, this));

  service_servers_.push_back(GenericServiceServer(
                               kReadMetricsServiceName, &Node::HandleReadMetrics, this));

  service_servers_.push_back(GenericServiceServer(kTrajectoryQueryServiceName, &Node::HandleTrajectoryQuery, this));

  navigation_client_.init_service_client<generic_lib::SlamInterface>(NAVIGATION_MAP_STATE);
  validate_init_ok();
  health_monitor_manager_.set_event(EVNT_SLAM_LOST_MAP); // after init OK
}


}  // namespace cartographer_ros
