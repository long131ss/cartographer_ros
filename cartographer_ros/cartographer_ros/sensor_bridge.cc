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

#include "cartographer_ros/sensor_bridge.h"

#include "absl/memory/memory.h"
#include "cartographer_ros/msg_conversion.h"
#include "cartographer_ros/slam_exception.h"
#include "cartographer_ros/time_conversion.h"

namespace cartographer_ros {

namespace carto = ::cartographer;

using carto::transform::Rigid3d;

namespace {

const std::string& CheckNoLeadingSlash(const std::string& frame_id) {
  if (frame_id.size() > 0) {
    CHECK_NE(frame_id[0], '/') << "The frame_id " << frame_id
                               << " should not start with a /. See 1.7 in "
                                  "http://wiki.ros.org/tf2/Migration.";
  }
  return frame_id;
}

}  // namespace

SensorBridge::SensorBridge(
    const int num_subdivisions_per_laser_scan,
    const std::string& tracking_frame,
    const double lookup_transform_timeout_sec, TfWrapper* tf_wrapper,
    carto::mapping::TrajectoryBuilderInterface* const trajectory_builder,
    const bool& pure_state)
    : num_subdivisions_per_laser_scan_(num_subdivisions_per_laser_scan),
      pure_state_(pure_state),
      tf_bridge_(tracking_frame, lookup_transform_timeout_sec, tf_wrapper),
      tf_wrapper_(tf_wrapper),
      trajectory_builder_(trajectory_builder) {}

std::unique_ptr<carto::sensor::OdometryData> SensorBridge::ToOdometryData(
    const nav_msgs::Odometry::ConstPtr& msg) {
  const carto::common::Time time = FromRos(msg->header.stamp);
  const auto sensor_to_tracking = tf_bridge_.LookupToTracking(
      time, CheckNoLeadingSlash(msg->child_frame_id));
  if (sensor_to_tracking == nullptr) {
    return nullptr;
  }
  return absl::make_unique<carto::sensor::OdometryData>(
      carto::sensor::OdometryData{
          time, ToRigid3d(msg->pose.pose) * sensor_to_tracking->inverse()});
}

void SensorBridge::HandleOdometryMessage(
    const std::string& sensor_id, const nav_msgs::Odometry::ConstPtr& msg) {
  std::unique_ptr<carto::sensor::OdometryData> odometry_data =
      ToOdometryData(msg);
  if (odometry_data != nullptr) {
    trajectory_builder_->AddSensorData(
        sensor_id,
        carto::sensor::OdometryData{odometry_data->time, odometry_data->pose});
  }
}

void SensorBridge::HandleNavSatFixMessage(
    const std::string& sensor_id, const sensor_msgs::NavSatFix::ConstPtr& msg) {
  const carto::common::Time time = FromRos(msg->header.stamp);
  if (msg->status.status == sensor_msgs::NavSatStatus::STATUS_NO_FIX) {
    trajectory_builder_->AddSensorData(
        sensor_id,
        carto::sensor::FixedFramePoseData{time, absl::optional<Rigid3d>()});
    return;
  }

  if (!ecef_to_local_frame_.has_value()) {
    ecef_to_local_frame_ =
        ComputeLocalFrameFromLatLong(msg->latitude, msg->longitude);
    LOG(INFO) << "Using NavSatFix. Setting ecef_to_local_frame with lat = "
              << msg->latitude << ", long = " << msg->longitude << ".";
  }

  trajectory_builder_->AddSensorData(
      sensor_id, carto::sensor::FixedFramePoseData{
                     time, absl::optional<Rigid3d>(Rigid3d::Translation(
                               ecef_to_local_frame_.value() *
                               LatLongAltToEcef(msg->latitude, msg->longitude,
                                                msg->altitude)))});
}

void SensorBridge::HandleLandmarkMessage(
    const std::string& sensor_id,
    const cartographer_ros_msgs::LandmarkList::ConstPtr& msg) {
  auto landmark_data = ToLandmarkData(*msg);

  auto tracking_from_landmark_sensor = tf_bridge_.LookupToTracking(
      landmark_data.time, CheckNoLeadingSlash(msg->header.frame_id));
  if (tracking_from_landmark_sensor != nullptr) {
    for (auto& observation : landmark_data.landmark_observations) {
      observation.landmark_to_tracking_transform =
          *tracking_from_landmark_sensor *
          observation.landmark_to_tracking_transform;
    }
  }
  trajectory_builder_->AddSensorData(sensor_id, landmark_data);
}

std::unique_ptr<carto::sensor::ImuData> SensorBridge::ToImuData(
    const sensor_msgs::Imu::ConstPtr& msg) {
  if (msg->angular_velocity_covariance[0] == -1) {
    throw SensorException(
        "Your IMU data claims to not contain angular velocity measurements "
        "by setting angular_velocity_covariance[0] to -1. Cartographer "
        "requires this data to work. See "
        "http://docs.ros.org/api/sensor_msgs/html/msg/Imu.html.");
  }
  if (msg->linear_acceleration_covariance[0] == -1) {
    throw SensorException(
        "Your IMU data claims to not contain linear acceleration measurements "
        "by setting acceleration_covariance[0] to -1. Cartographer "
        "requires this data to work. See "
        "http://docs.ros.org/api/sensor_msgs/html/msg/Imu.html.");
  }
  const carto::common::Time time = FromRos(msg->header.stamp);
  const auto sensor_to_tracking = tf_bridge_.LookupToTracking(
      time, CheckNoLeadingSlash(msg->header.frame_id));
  if (sensor_to_tracking == nullptr) {
    return nullptr;
  }
  CHECK(sensor_to_tracking->translation().norm() < 1e-5)
      << "The IMU frame must be colocated with the tracking frame. "
         "Transforming linear acceleration into the tracking frame will "
         "otherwise be imprecise.";
  return absl::make_unique<carto::sensor::ImuData>(carto::sensor::ImuData{
      time, sensor_to_tracking->rotation() * ToEigen(msg->linear_acceleration),
      sensor_to_tracking->rotation() * ToEigen(msg->angular_velocity)});
}

void SensorBridge::HandleImuMessage(const std::string& sensor_id,
                                    const sensor_msgs::Imu::ConstPtr& msg) {
  std::unique_ptr<carto::sensor::ImuData> imu_data = ToImuData(msg);
  if (imu_data != nullptr) {
    trajectory_builder_->AddSensorData(
        sensor_id,
        carto::sensor::ImuData{imu_data->time, imu_data->linear_acceleration,
                               imu_data->angular_velocity});
  }
}

void SensorBridge::HandleLaserScanMessage(
    const std::string& sensor_id, const sensor_msgs::LaserScan::ConstPtr& msg) {
  carto::sensor::PointCloudWithIntensities point_cloud;//带时间(相对时间)和强度的点云
  carto::common::Time time;
  std::tie(point_cloud, time) = ToPointCloudWithIntensities(*msg);
  HandleLaserScan(sensor_id, time, msg->header.frame_id, point_cloud);
}

void SensorBridge::HandleMultiEchoLaserScanMessage(
    const std::string& sensor_id,
    const sensor_msgs::MultiEchoLaserScan::ConstPtr& msg) {
  carto::sensor::PointCloudWithIntensities point_cloud;
  carto::common::Time time;
  std::tie(point_cloud, time) = ToPointCloudWithIntensities(*msg);
  HandleLaserScan(sensor_id, time, msg->header.frame_id, point_cloud);
}

void SensorBridge::HandlePointCloud2Message(
    const std::string& sensor_id,
    const sensor_msgs::PointCloud2::ConstPtr& msg) {
  carto::sensor::PointCloudWithIntensities point_cloud;
  carto::common::Time time;
  std::tie(point_cloud, time) = ToPointCloudWithIntensities(*msg);
  HandleRangefinder(sensor_id, time, msg->header.frame_id, point_cloud.points);
}

const TfBridge& SensorBridge::tf_bridge() const { return tf_bridge_; }

void SensorBridge::HandleLaserScan(
    const std::string& sensor_id, const carto::common::Time time,
    const std::string& frame_id,
    const carto::sensor::PointCloudWithIntensities& points) {
  if (points.points.empty()) {
    return;
  }
  //points.points.back().time应该是等于0的
  if (points.points.back().time > 0) {
    throw SensorException("HandleLaserScan -> points.points.back().time > 0");
  }
  // TODO(gaschler): Use per-point time instead of subdivisions.
  //将一帧激光分成若干份,可配置
  for (int i = 0; i != num_subdivisions_per_laser_scan_; ++i) {
    const size_t start_index =
        points.points.size() * i / num_subdivisions_per_laser_scan_;
    const size_t end_index =
        points.points.size() * (i + 1) / num_subdivisions_per_laser_scan_;
    carto::sensor::TimedPointCloud subdivision(
        points.points.begin() + start_index, points.points.begin() + end_index);
    if (start_index == end_index) {
      continue;
    }
    //
    const double time_to_subdivision_end = subdivision.back().time;//本小块subdivision最后一个点的相对整个一帧的最后一个点的相对时间
    // `subdivision_time` is the end of the measurement so sensor::Collator will
    // send all other sensor data first.
    const carto::common::Time subdivision_time =
        time + carto::common::FromSeconds(time_to_subdivision_end); //subdivision的最后一个点的绝对时间戳
    auto it = sensor_to_previous_subdivision_time_.find(sensor_id);
    //上一子块和当前子块的在时间戳上有重叠,则忽略当前子块
    //为什么会有这种情况呢?因为假如有两个雷达,发出来的topic都是scan,sensor_id都是laser,可能会出现子块重叠的情况
    if (it != sensor_to_previous_subdivision_time_.end() &&
        it->second >= subdivision_time) {
      LOG(WARNING) << "Ignored subdivision of a LaserScan message from sensor "
                   << sensor_id << " because previous subdivision time "
                   << it->second << " is not before current subdivision time "
                   << subdivision_time;
      continue;
    }
    sensor_to_previous_subdivision_time_[sensor_id] = subdivision_time;
    bool is_far_distance_point;
    bool do_not_need_this_point;
    int count = 0;
    for (auto& point : subdivision) {
      point.time -= time_to_subdivision_end;//因为我们是一个分块一个分块的处理的,所以要把每个分块里面的点的相对时间戳转换为相对本分块最后一个点的时间错
      if (pure_state_) {
        continue;
      }
      count++;
      is_far_distance_point =
          hypot(point.position[0], point.position[1]) > 7;  // beyond 7m filter
      do_not_need_this_point =
          is_far_distance_point &&
          count % 3 != 0;  // remove 2 point for each 3 points
      if (do_not_need_this_point) {
        point.position[0] = point.position[1] = 0.0;
      }
    }
    if (subdivision.back().time != 0) {
      throw SensorException("HandleLaserScan -> subdivision.back().time != 0");
    }
    HandleRangefinder(sensor_id, subdivision_time, frame_id, subdivision);
  }
}

void SensorBridge::HandleRangefinder(
    const std::string& sensor_id, const carto::common::Time time,
    const std::string& frame_id, const carto::sensor::TimedPointCloud& ranges) {
  if (!ranges.empty()) {
    CHECK_LE(ranges.back().time, 0.f);//小分块最后一个点的相对时间戳应该小于等于0
  }
  //sensor可以理解为laser坐标系,tracking可以理解为建图要跟踪的坐标系
  //假如我们跟踪的坐标系是base_link,那么sensor_to_tracking就是laser坐标系到base_link坐标系的变换
  const auto sensor_to_tracking =
      tf_bridge_.LookupToTracking(time, CheckNoLeadingSlash(frame_id));
  if (sensor_to_tracking != nullptr) {
    trajectory_builder_->AddSensorData(
        sensor_id, carto::sensor::TimedPointCloudData{
                       time, sensor_to_tracking->translation().cast<float>(),
                       carto::sensor::TransformTimedPointCloud(
                           ranges, sensor_to_tracking->cast<float>())});
  }
}

void SensorBridge::PauseCollating(const std::string& sensor_id) {
  trajectory_builder_->PauseCollating(sensor_id);
}

void SensorBridge::ResumeCollating(const std::string& sensor_id) {
  trajectory_builder_->ResumeCollating(sensor_id);
}

}  // namespace cartographer_ros
