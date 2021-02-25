# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import math
import numpy as np
import gym
import cv2
from scipy.spatial import distance
from benchmark.common import ActionAdapter

from smarts.core.agent_interface import (
    AgentInterface,
    AgentType,
    OGM,
    Waypoints,
    NeighborhoodVehicles,
)

from smarts.core.controllers import ActionSpaceType
from smarts.core.agent import AgentSpec
from ultra.baselines.adapter import BaselineAdapter
from ultra.env.ultra_env import UltraEnv

from ultra.utils.common import (
    ego_social_safety,
    get_closest_waypoint,
    get_path_to_goal,
    rotate2d_vector,
)

num_lookahead = 100


class UltraGym(UltraEnv):
    def __init__(
        self,
        max_episode_steps,
        action_type,
        scenario_info,
        headless,
        timestep_sec,
        seed,
        eval_mode=False,
        ordered_scenarios=False,
    ):

        self.timestep_sec = timestep_sec
        self.headless = headless
        self.scenario_info = scenario_info
        self.scenarios = self.get_task(scenario_info[0], scenario_info[1])

        adapter = GymAdapter()
        if action_type == "discrete":
            action_type = ActionSpaceType.Lane
            self.action_space = gym.spaces.Discrete(4)
        elif action_type == "continuous":
            action_type = ActionSpaceType.Continuous

        spec = AgentSpec(
            interface=AgentInterface(
                waypoints=Waypoints(lookahead=20),
                neighborhood_vehicles=NeighborhoodVehicles(200),
                action=action_type,
                rgb=True,
                max_episode_steps=max_episode_steps,
                debug=True,
            ),
            observation_adapter=adapter.observation_adapter,
            reward_adapter=adapter.reward_adapter,
            action_adapter=ActionAdapter.from_type(action_type),
        )

        super().__init__(
            agent_specs={"007": spec},
            scenario_info=scenario_info,
            headless=headless,
            timestep_sec=timestep_sec,
            seed=seed,
        )


class GymAdapter:
    def __init__(self):
        pass

    def observation_adapter(self, env_observation):
        ego_state = env_observation.ego_vehicle_state
        start = env_observation.ego_vehicle_state.mission.start
        goal = env_observation.ego_vehicle_state.mission.goal
        path = get_path_to_goal(
            goal=goal, paths=env_observation.waypoint_paths, start=start
        )
        closest_wp, _ = get_closest_waypoint(
            num_lookahead=num_lookahead,
            goal_path=path,
            ego_position=ego_state.position,
            ego_heading=ego_state.heading,
        )
        signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
        lane_width = closest_wp.lane_width * 0.5
        ego_dist_center = signed_dist_from_center / lane_width

        relative_goal_position = np.asarray(goal.position[0:2]) - np.asarray(
            ego_state.position[0:2]
        )
        relative_goal_position_rotated = rotate2d_vector(
            relative_goal_position, -ego_state.heading
        )

        state = dict(
            speed=ego_state.speed,
            relative_goal_position=relative_goal_position_rotated,
            distance_from_center=ego_dist_center,
            steering=ego_state.steering,
            angle_error=closest_wp.relative_heading(ego_state.heading),
            social_vehicles=env_observation.neighborhood_vehicle_states,
            road_speed=closest_wp.speed_limit,
            # ----------
            # dont normalize the following,
            start=start.position,
            goal=goal.position,
            heading=ego_state.heading,
            goal_path=path,
            ego_position=ego_state.position,
            waypoint_paths=env_observation.waypoint_paths,
            events=env_observation.events,
        )

        def rgb2gray(rgb):
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            gray = 0.2125 * r + 0.7154 * g + 0.0721 * b

            return gray

        rgb_ndarray = env_observation.top_down_rgb.data
        gray_scale = rgb2gray(rgb_ndarray)
        gray_scale = (
            cv2.resize(
                rgb2gray(rgb_ndarray),
                rgb_ndarray.shape[0:2],
                interpolation=cv2.INTER_CUBIC,
            )
            / 255.0
        )
        return gray_scale

    def reward_adapter(self, observation, reward):
        env_reward = reward
        ego_events = observation.events
        ego_observation = observation.ego_vehicle_state
        start = observation.ego_vehicle_state.mission.start
        goal = observation.ego_vehicle_state.mission.goal
        path = get_path_to_goal(
            goal=goal, paths=observation.waypoint_paths, start=start
        )

        linear_jerk = np.linalg.norm(ego_observation.linear_jerk)
        angular_jerk = np.linalg.norm(ego_observation.angular_jerk)

        # Distance to goal
        ego_2d_position = ego_observation.position[0:2]
        goal_dist = distance.euclidean(ego_2d_position, goal.position)

        closest_wp, _ = get_closest_waypoint(
            num_lookahead=num_lookahead,
            goal_path=path,
            ego_position=ego_observation.position,
            ego_heading=ego_observation.heading,
        )
        angle_error = closest_wp.relative_heading(
            ego_observation.heading
        )  # relative heading radians [-pi, pi]

        # Distance from center
        signed_dist_from_center = closest_wp.signed_lateral_error(
            observation.ego_vehicle_state.position
        )
        lane_width = closest_wp.lane_width * 0.5
        ego_dist_center = signed_dist_from_center / lane_width

        # number of violations
        (ego_num_violations, social_num_violations,) = ego_social_safety(
            observation,
            d_min_ego=1.0,
            t_c_ego=1.0,
            d_min_social=1.0,
            t_c_social=1.0,
            ignore_vehicle_behind=True,
        )

        speed_fraction = max(0, ego_observation.speed / closest_wp.speed_limit)
        ego_step_reward = 0.02 * min(speed_fraction, 1) * np.cos(angle_error)
        ego_speed_reward = min(
            0, (closest_wp.speed_limit - ego_observation.speed) * 0.01
        )  # m/s
        ego_collision = len(ego_events.collisions) > 0
        ego_collision_reward = -1.0 if ego_collision else 0.0
        ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
        ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
        ego_wrong_way = -0.02 if ego_events.wrong_way else 0.0
        ego_goal_reward = 0.0
        ego_time_out = 0.0
        ego_dist_center_reward = -0.002 * min(1, abs(ego_dist_center))
        ego_angle_error_reward = -0.005 * max(0, np.cos(angle_error))
        ego_reached_goal = 1.0 if ego_events.reached_goal else 0.0
        ego_safety_reward = -0.02 if ego_num_violations > 0 else 0
        social_safety_reward = -0.02 if social_num_violations > 0 else 0
        ego_lat_speed = 0.0  # -0.1 * abs(long_lat_speed[1])
        ego_linear_jerk = -0.0001 * linear_jerk
        ego_angular_jerk = -0.0001 * angular_jerk * math.cos(angle_error)
        env_reward /= 100
        # DG: Different speed reward
        ego_speed_reward = -0.1 if speed_fraction >= 1 else 0.0
        ego_speed_reward += -0.01 if speed_fraction < 0.01 else 0.0

        rewards = [
            ego_goal_reward,
            ego_collision_reward,
            ego_off_road_reward,
            ego_off_route_reward,
            ego_wrong_way,
            ego_speed_reward,
            # ego_time_out,
            ego_dist_center_reward,
            ego_angle_error_reward,
            ego_reached_goal,
            ego_step_reward,
            env_reward,
            # ego_linear_jerk,
            # ego_angular_jerk,
            # ego_lat_speed,
            # ego_safety_reward,
            # social_safety_reward,
        ]
        return sum(rewards)