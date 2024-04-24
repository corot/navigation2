// Copyright (c) 2020, Samsung Research America
// Copyright (c) 2023, Open Navigation LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. Reserved.

#include <string>
#include <memory>
#include <vector>
#include <limits>

#include "Eigen/Core"

#include "mbf_msgs/GetPathResult.h"

#include "nav2_smac_planner/smac_planner_hybrid.hpp"

// #define BENCHMARK_TESTING

namespace nav2_smac_planner
{

SmacPlannerHybrid::SmacPlannerHybrid()
: _a_star(nullptr),
  _collision_checker(nullptr, 1),
  _smoother(nullptr),
  _costmap(nullptr),
  _costmap_ros(nullptr),
  _costmap_downsampler(nullptr)
{
}

SmacPlannerHybrid::~SmacPlannerHybrid()
{
  ROS_INFO("Destroying plugin %s of type SmacPlannerHybrid",
    _name.c_str());
}

void SmacPlannerHybrid::initialize(
  std::string name,
  costmap_2d::Costmap2DROS* costmap_ros) {
  _name = name;
  _costmap = costmap_ros->getCostmap();
  _costmap_ros = std::shared_ptr<costmap_2d::Costmap2DROS>(costmap_ros);
  _global_frame = costmap_ros->getGlobalFrameID();

  ROS_INFO("Initializing %s of type SmacPlannerHybrid", name.c_str());

  ros::NodeHandle parent_nh("~");
  ros::NodeHandle private_nh(parent_nh, name);
  _raw_plan_publisher = private_nh.advertise<nav_msgs::Path>("unsmoothed_plan", 1);
  _expansions_publisher = private_nh.advertise<geometry_msgs::PoseArray>("expansions", 1);
  _planned_footprints_publisher = private_nh.advertise<visualization_msgs::MarkerArray>(
      "planned_footprints", 1);

  dsrv_ = std::make_unique<dynamic_reconfigure::Server<SmacPlannerHybridConfig>>(private_nh);
  dsrv_->setCallback(boost::bind(&SmacPlannerHybrid::reconfigureCB, this, _1, _2));
}

void SmacPlannerHybrid::reconfigureCB(SmacPlannerHybridConfig& config, uint32_t level)
{
  _angle_bin_size = 2.0 * M_PI / config.angle_quantization_bins;
  _angle_quantizations = static_cast<unsigned int>(config.angle_quantization_bins);
  _tolerance = config.tolerance;
  _motion_model = static_cast<MotionModel>(config.motion_model_for_search);

  _downsample_costmap = config.downsample_costmap;
  _downsampling_factor = config.downsampling_factor;

  _allow_unknown = config.allow_unknown;
  _max_iterations = config.max_iterations;
  _max_on_approach_iterations = config.max_on_approach_iterations;
  _terminal_checking_interval = config.terminal_checking_interval;

  _max_planning_time = config.max_planning_time;
  _lookup_table_size = config.lookup_table_size;
  _minimum_turning_radius_global_coords = config.minimum_turning_radius;
  _debug_visualizations = config.debug_visualizations;

  _search_info.non_straight_penalty = config.non_straight_penalty;
  _search_info.change_penalty = config.change_penalty;
  _search_info.reverse_penalty = config.reverse_penalty;
  _search_info.cost_penalty = config.cost_penalty;
  _search_info.retrospective_penalty = config.retrospective_penalty;
  _search_info.analytic_expansion_ratio = config.analytic_expansion_ratio;
  _search_info.analytic_expansion_max_length = config.analytic_expansion_max_length / _costmap->getResolution();
  _search_info.analytic_expansion_max_cost = config.analytic_expansion_max_cost;
  _search_info.analytic_expansion_max_cost_override = config.analytic_expansion_max_cost_override;
  _search_info.cache_obstacle_heuristic = config.cache_obstacle_heuristic;
  _search_info.allow_primitive_interpolation = config.allow_primitive_interpolation;
  _search_info.downsample_obstacle_heuristic = config.downsample_obstacle_heuristic;
  _search_info.use_quadratic_cost_penalty = config.use_quadratic_cost_penalty;

  if (_max_on_approach_iterations <= 0) {
    ROS_WARN( "On approach iteration selected as <= 0, "
      "disabling tolerance and on approach iterations.");
    _max_on_approach_iterations = std::numeric_limits<int>::max();
  }

  if (_max_iterations <= 0) {
    ROS_WARN( "maximum iteration selected as <= 0, "
      "disabling maximum iterations.");
    _max_iterations = std::numeric_limits<int>::max();
  }

  if (_minimum_turning_radius_global_coords < _costmap->getResolution() * _downsampling_factor) {
    ROS_WARN( "Min turning radius cannot be less than the search grid cell resolution!");
    _minimum_turning_radius_global_coords = _costmap->getResolution() * _downsampling_factor;
  }

  // convert to grid coordinates
  if (!_downsample_costmap) {
    _downsampling_factor = 1;
  }
  _search_info.minimum_turning_radius =
    _minimum_turning_radius_global_coords / (_costmap->getResolution() * _downsampling_factor);
  _lookup_table_dim =
    static_cast<float>(_lookup_table_size) /
    static_cast<float>(_costmap->getResolution() * _downsampling_factor);

  // Make sure its a whole number
  _lookup_table_dim = static_cast<float>(static_cast<int>(_lookup_table_dim));

  // Make sure its an odd number
  if (static_cast<int>(_lookup_table_dim) % 2 == 0) {
    ROS_INFO("Even sized heuristic lookup table size set %f, increasing size by 1 to make odd",
      _lookup_table_dim);
    _lookup_table_dim += 1.0;
  }

  // Initialize collision checker
  _collision_checker = GridCollisionChecker(_costmap_ros, _angle_quantizations);
  _collision_checker.setFootprint(
    _costmap_ros->getRobotFootprint(),
    _costmap_ros->getUseRadius(),
    findCircumscribedCost(_costmap_ros.get()));

  // Initialize A* template
  _a_star = std::make_unique<AStarAlgorithm<NodeHybrid>>(_motion_model, _search_info);
  _a_star->initialize(
    _allow_unknown,
    _max_iterations,
    _max_on_approach_iterations,
    _terminal_checking_interval,
    _max_planning_time,
    _lookup_table_dim,
    _angle_quantizations);

  // Initialize path smoother
  _smoother = std::make_unique<Smoother>();
  _smoother->initialize(_minimum_turning_radius_global_coords);

  // Initialize costmap downsampler
  if (_downsample_costmap && _downsampling_factor > 1) {
    _costmap_downsampler = std::make_unique<CostmapDownsampler>();
    std::string topic_name = "downsampled_costmap";
    _costmap_downsampler->on_configure(
      _global_frame, topic_name, _costmap, _downsampling_factor);
  }

  ROS_INFO("Configured plugin %s of type SmacPlannerHybrid with "
    "maximum iterations %i, max on approach iterations %i, and %s. Tolerance %.2f."
    "Using motion model: %s.",
    _name.c_str(), _max_iterations, _max_on_approach_iterations,
    _allow_unknown ? "allowing unknown traversal" : "not allowing unknown traversal",
    _tolerance, toString(_motion_model).c_str());
}

uint32_t SmacPlannerHybrid::makePlan(
    const geometry_msgs::PoseStamped & start,
    const geometry_msgs::PoseStamped & goal,
    double tolerance,
    std::vector<geometry_msgs::PoseStamped> & plan,
    double &cost,
    std::string &message)
{
  _planning_canceled = false;

  std::lock_guard<std::mutex> lock_reinit(_mutex);
  ros::Time a = ros::Time::now();

  std::unique_lock<costmap_2d::Costmap2D::mutex_t> lock(*(_costmap->getMutex()));

  // Downsample costmap, if required
  costmap_2d::Costmap2D * costmap = _costmap;
  if (_costmap_downsampler) {
    costmap = _costmap_downsampler->downsample(_downsampling_factor);
    _collision_checker.setCostmap(costmap);
  }

  // Set collision checker and costmap information
  _collision_checker.setFootprint(
    _costmap_ros->getRobotFootprint(),
    _costmap_ros->getUseRadius(),
    findCircumscribedCost(_costmap_ros.get()));
  _a_star->setCollisionChecker(&_collision_checker);

  // Set starting point, in A* bin search coordinates
  float mx, my;
  if (!costmap->worldToMapContinuous(start.pose.position.x, start.pose.position.y, mx, my)) {
    message = "Start Coordinates of(" + std::to_string(start.pose.position.x) + ", " +
            std::to_string(start.pose.position.y) + ") was outside bounds";
    return mbf_msgs::GetPathResult::OUT_OF_MAP;
  }

  double orientation_bin = tf2::getYaw(start.pose.orientation) / _angle_bin_size;
  while (orientation_bin < 0.0) {
    orientation_bin += static_cast<float>(_angle_quantizations);
  }
  // This is needed to handle precision issues
  if (orientation_bin >= static_cast<float>(_angle_quantizations)) {
    orientation_bin -= static_cast<float>(_angle_quantizations);
  }
  unsigned int orientation_bin_id = static_cast<unsigned int>(floor(orientation_bin));

  if (_collision_checker.inCollision(mx, my, orientation_bin_id, _allow_unknown)) {
    message = "Start pose is blocked";
    return mbf_msgs::GetPathResult::BLOCKED_START;
  }

  _a_star->setStart(mx, my, orientation_bin_id);

  // Set goal point, in A* bin search coordinates
  if (!costmap->worldToMapContinuous(goal.pose.position.x, goal.pose.position.y, mx, my)) {
    message = "Goal Coordinates of(" + std::to_string(goal.pose.position.x) + ", " +
            std::to_string(goal.pose.position.y) + ") was outside bounds";
    return mbf_msgs::GetPathResult::OUT_OF_MAP;
  }

  orientation_bin = tf2::getYaw(goal.pose.orientation) / _angle_bin_size;
  while (orientation_bin < 0.0) {
    orientation_bin += static_cast<float>(_angle_quantizations);
  }
  // This is needed to handle precision issues
  if (orientation_bin >= static_cast<float>(_angle_quantizations)) {
    orientation_bin -= static_cast<float>(_angle_quantizations);
  }
  orientation_bin_id = static_cast<unsigned int>(floor(orientation_bin));

  if (_collision_checker.inCollision(mx, my, orientation_bin_id, _allow_unknown)) {
    message = "Goal pose is blocked";
    return mbf_msgs::GetPathResult::BLOCKED_GOAL;
  }

  _a_star->setGoal(mx, my, orientation_bin_id);

  // Setup message
  nav_msgs::Path output_path;
  output_path.header.stamp = ros::Time::now();
  output_path.header.frame_id = _global_frame;
  geometry_msgs::PoseStamped pose;
  pose.header = output_path.header;
  pose.pose.position.z = 0.0;
  pose.pose.orientation.x = 0.0;
  pose.pose.orientation.y = 0.0;
  pose.pose.orientation.z = 0.0;
  pose.pose.orientation.w = 1.0;

  // Compute output_path
  NodeHybrid::CoordinateVector path;
  int num_iterations = 0;
  std::string error;
  std::unique_ptr<std::vector<std::tuple<float, float, float>>> expansions = nullptr;
  if (_debug_visualizations) {
    expansions = std::make_unique<std::vector<std::tuple<float, float, float>>>();
  }
  // Note: All exceptions thrown are handled by the planner server and returned to the action
  if (!_a_star->createPath(
      path, num_iterations,
      _tolerance / static_cast<float>(costmap->getResolution()), [&](){ return _planning_canceled; }, expansions.get()))
  {
    if (_debug_visualizations) {
      geometry_msgs::PoseArray msg;
      geometry_msgs::Pose msg_pose;
      msg.header.stamp = ros::Time::now();
      msg.header.frame_id = _global_frame;
      for (auto & e : *expansions) {
        msg_pose.position.x = std::get<0>(e);
        msg_pose.position.y = std::get<1>(e);
        msg_pose.orientation = getWorldOrientation(std::get<2>(e));
        msg.poses.push_back(msg_pose);
      }
      _expansions_publisher.publish(msg);
    }

    if (_planning_canceled) {
      message = "Planner was cancelled";
      return mbf_msgs::GetPathResult::CANCELED;
    }

    // Note: If the start is blocked only one iteration will occur before failure,
    // but this should not happen because we check the start pose before planning
    if (num_iterations == 1) {
      message = "Start pose is blocked";
      return mbf_msgs::GetPathResult::BLOCKED_START;
    }

    if (num_iterations < _a_star->getMaxIterations()) {
      message = "No valid path found";
    } else {
      message = "Exceeded maximum iterations";
    }
    return mbf_msgs::GetPathResult::NO_PATH_FOUND;
  }

  // Convert to world coordinates
  output_path.poses.reserve(path.size());
  for (int i = path.size() - 1; i >= 0; --i) {
    pose.pose = getWorldCoords(path[i].x, path[i].y, costmap);
    pose.pose.orientation = getWorldOrientation(path[i].theta);
    output_path.poses.push_back(pose);
  }

  // Publish raw path for debug
  if (_raw_plan_publisher.getNumSubscribers() > 0) {
    _raw_plan_publisher.publish(output_path);
  }

  if (_debug_visualizations) {
    // Publish expansions for debug
    geometry_msgs::PoseArray msg;
    geometry_msgs::Pose msg_pose;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = _global_frame;
    for (auto & e : *expansions) {
      msg_pose.position.x = std::get<0>(e);
      msg_pose.position.y = std::get<1>(e);
      msg_pose.orientation = getWorldOrientation(std::get<2>(e));
      msg.poses.push_back(msg_pose);
    }
    _expansions_publisher.publish(msg);

    // plot footprint path planned for debug
    if (_planned_footprints_publisher.getNumSubscribers() > 0) {
      visualization_msgs::MarkerArray marker_array;
      for (size_t i = 0; i < output_path.poses.size(); i++) {
        const std::vector<geometry_msgs::Point> edge =
          transformFootprintToEdges(output_path.poses[i].pose, _costmap_ros->getRobotFootprint());
        marker_array.markers.push_back(createMarker(edge, i, _global_frame, ros::Time::now()));
      }

      if (marker_array.markers.empty()) {
        visualization_msgs::Marker clear_all_marker;
        clear_all_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(clear_all_marker);
      }
      _planned_footprints_publisher.publish(marker_array);
    }
  }

  // Find how much time we have left to do smoothing
  ros::Time b = ros::Time::now();
  double time_remaining = _max_planning_time - (b - a).toSec();

#ifdef BENCHMARK_TESTING
  std::cout << "It took " << time_span.count() * 1000 <<
    " milliseconds with " << num_iterations << " iterations." << std::endl;
#endif

  // Smooth output_path
  if (_smoother && num_iterations > 1) {
    _smoother->smooth(output_path, costmap, time_remaining);
  }

#ifdef BENCHMARK_TESTING
  ros::Time c = ros::Time::now();
  std::cout << "It took " << (c - b).toSec() * 1000 <<
    " milliseconds to smooth path." << std::endl;
#endif

  plan = std::move(output_path.poses);
  return mbf_msgs::GetPathResult::SUCCESS;
}

bool SmacPlannerHybrid::cancel() {
  _planning_canceled = true;
  return true;
};

/* TODO reinitializing all with every reconfig!!!  restore this if too slow
rcl_interfaces::SetParametersResult
SmacPlannerHybrid::dynamicParametersCallback(std::vector<ros::Parameter> parameters)
{
  rcl_interfaces::SetParametersResult result;
  std::lock_guard<std::mutex> lock_reinit(_mutex);

  bool reinit_collision_checker = false;
  bool reinit_a_star = false;
  bool reinit_downsampler = false;
  bool reinit_smoother = false;

  for (auto parameter : parameters) {
    const auto & type = parameter.get_type();
    const auto & name = parameter.get_name();

    if (type == ParameterType::PARAMETER_DOUBLE) {
      if (name == _name + ".max_planning_time") {
        reinit_a_star = true;
        _max_planning_time = parameter.as_double();
      } else if (name == _name + ".tolerance") {
        _tolerance = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".lookup_table_size") {
        reinit_a_star = true;
        _lookup_table_size = parameter.as_double();
      } else if (name == _name + ".minimum_turning_radius") {
        reinit_a_star = true;
        if (_smoother) {
          reinit_smoother = true;
        }

        if (parameter.as_double() < _costmap->getResolution() * _downsampling_factor) {
          ROS_ERROR("Min turning radius cannot be less than the search grid cell resolution!");
          result.successful = false;
        }

        _minimum_turning_radius_global_coords = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".reverse_penalty") {
        reinit_a_star = true;
        _search_info.reverse_penalty = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".change_penalty") {
        reinit_a_star = true;
        _search_info.change_penalty = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".non_straight_penalty") {
        reinit_a_star = true;
        _search_info.non_straight_penalty = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".cost_penalty") {
        reinit_a_star = true;
        _search_info.cost_penalty = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".analytic_expansion_ratio") {
        reinit_a_star = true;
        _search_info.analytic_expansion_ratio = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".analytic_expansion_max_length") {
        reinit_a_star = true;
        _search_info.analytic_expansion_max_length =
          static_cast<float>(parameter.as_double()) / _costmap->getResolution();
      } else if (name == _name + ".analytic_expansion_max_cost") {
        reinit_a_star = true;
        _search_info.analytic_expansion_max_cost = static_cast<float>(parameter.as_double());
      } else if (name == "resolution") {
        // Special case: When the costmap's resolution changes, need to reinitialize
        // the controller to have new resolution information
        ROS_INFO("Costmap resolution changed. Reinitializing SmacPlannerHybrid.");
        reinit_collision_checker = true;
        reinit_a_star = true;
        reinit_downsampler = true;
        reinit_smoother = true;
      }
    } else if (type == ParameterType::PARAMETER_BOOL) {
      if (name == _name + ".downsample_costmap") {
        reinit_downsampler = true;
        _downsample_costmap = parameter.as_bool();
      } else if (name == _name + ".allow_unknown") {
        reinit_a_star = true;
        _allow_unknown = parameter.as_bool();
      } else if (name == _name + ".cache_obstacle_heuristic") {
        reinit_a_star = true;
        _search_info.cache_obstacle_heuristic = parameter.as_bool();
      } else if (name == _name + ".allow_primitive_interpolation") {
        _search_info.allow_primitive_interpolation = parameter.as_bool();
        reinit_a_star = true;
      } else if (name == _name + ".smooth_path") {
        if (parameter.as_bool()) {
          reinit_smoother = true;
        } else {
          _smoother.reset();
        }
      } else if (name == _name + ".analytic_expansion_max_cost_override") {
        _search_info.analytic_expansion_max_cost_override = parameter.as_bool();
        reinit_a_star = true;
      }
    } else if (type == ParameterType::PARAMETER_INTEGER) {
      if (name == _name + ".downsampling_factor") {
        reinit_a_star = true;
        reinit_downsampler = true;
        _downsampling_factor = parameter.as_int();
      } else if (name == _name + ".max_iterations") {
        reinit_a_star = true;
        _max_iterations = parameter.as_int();
        if (_max_iterations <= 0) {
          ROS_INFO("maximum iteration selected as <= 0, "
            "disabling maximum iterations.");
          _max_iterations = std::numeric_limits<int>::max();
        }
      } else if (name == _name + ".max_on_approach_iterations") {
        reinit_a_star = true;
        _max_on_approach_iterations = parameter.as_int();
        if (_max_on_approach_iterations <= 0) {
          ROS_INFO("On approach iteration selected as <= 0, "
            "disabling tolerance and on approach iterations.");
          _max_on_approach_iterations = std::numeric_limits<int>::max();
        }
      } else if (name == _name + ".terminal_checking_interval") {
        reinit_a_star = true;
        _terminal_checking_interval = parameter.as_int();
      } else if (name == _name + ".angle_quantization_bins") {
        reinit_collision_checker = true;
        reinit_a_star = true;
        int angle_quantizations = parameter.as_int();
        _angle_bin_size = 2.0 * M_PI / angle_quantizations;
        _angle_quantizations = static_cast<unsigned int>(angle_quantizations);
      }
    } else if (type == ParameterType::PARAMETER_STRING) {
      if (name == _name + ".motion_model_for_search") {
        reinit_a_star = true;
        _motion_model = fromString(parameter.as_string());
        if (_motion_model == MotionModel::UNKNOWN) {
          ROS_WARN("Unable to get MotionModel search type. Given '%s', "
            "valid options are MOORE, VON_NEUMANN, DUBIN, REEDS_SHEPP.",
            _motion_model_for_search.c_str());
        }
      }
    }
  }

  // Re-init if needed with mutex lock (to avoid re-init while creating a plan)
  if (reinit_a_star || reinit_downsampler || reinit_collision_checker || reinit_smoother) {
    // convert to grid coordinates
    if (!_downsample_costmap) {
      _downsampling_factor = 1;
    }
    _search_info.minimum_turning_radius =
      _minimum_turning_radius_global_coords / (_costmap->getResolution() * _downsampling_factor);
    _lookup_table_dim =
      static_cast<float>(_lookup_table_size) /
      static_cast<float>(_costmap->getResolution() * _downsampling_factor);

    // Make sure its a whole number
    _lookup_table_dim = static_cast<float>(static_cast<int>(_lookup_table_dim));

    // Make sure its an odd number
    if (static_cast<int>(_lookup_table_dim) % 2 == 0) {
      ROS_INFO(
        "Even sized heuristic lookup table size set %f, increasing size by 1 to make odd",
        _lookup_table_dim);
      _lookup_table_dim += 1.0;
    }

    auto node = _node.lock();

    // Re-Initialize A* template
    if (reinit_a_star) {
      _a_star = std::make_unique<AStarAlgorithm<NodeHybrid>>(_motion_model, _search_info);
      _a_star->initialize(
        _allow_unknown,
        _max_iterations,
        _max_on_approach_iterations,
        _terminal_checking_interval,
        _max_planning_time,
        _lookup_table_dim,
        _angle_quantizations);
    }

    // Re-Initialize costmap downsampler
    if (reinit_downsampler) {
      if (_downsample_costmap && _downsampling_factor > 1) {
        std::string topic_name = "downsampled_costmap";
        _costmap_downsampler = std::make_unique<CostmapDownsampler>();
        _costmap_downsampler->on_configure(
          node, _global_frame, topic_name, _costmap, _downsampling_factor);
      }
    }

    // Re-Initialize collision checker
    if (reinit_collision_checker) {
      _collision_checker = GridCollisionChecker(_costmap_ros, _angle_quantizations, node);
      _collision_checker.setFootprint(
        _costmap_ros->getRobotFootprint(),
        _costmap_ros->getUseRadius(),
        findCircumscribedCost(_costmap_ros));
    }

    // Re-Initialize smoother
    if (reinit_smoother) {
      SmootherParams params;
      params.get(node, _name);
      _smoother = std::make_unique<Smoother>(params);
      _smoother->initialize(_minimum_turning_radius_global_coords);
    }
  }
  result.successful = true;
  return result;
}*/

}  // namespace nav2_smac_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_smac_planner::SmacPlannerHybrid, mbf_costmap_core::CostmapPlanner)
