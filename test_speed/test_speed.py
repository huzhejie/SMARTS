import argparse
from pathlib import Path

import gym

from smarts.core.agent import AgentPolicy
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType


class KeeplanePolicy(AgentPolicy):
    def act(self, obs):
        return 0


def parse_args():
    parser = argparse.ArgumentParser("run simple keep lane agent")
    # env setting
    parser.add_argument("--scenario", "-s", type=str, help="Path to scenario")
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )

    return parser.parse_args()


def main(args):
    scenario_path = Path(args.scenario).absolute()

    AGENT_ID = "AGENT-007"

    agent_interface = AgentInterface(
        max_episode_steps=10000,
        action=ActionSpaceType.LaneWithContinuousSpeed,
    )

    agent_spec = AgentSpec(
        interface=agent_interface,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario_path],
        agent_specs={AGENT_ID: agent_spec},
        # set headless to false if u want to use envision
        headless=False,
        visdom=False,
        seed=42,
        timestep_sec=0.08,
        # sumo_port=sumo_port,
        # num_external_sumo_clients=1,
        # sumo_headless=False,
        # sumo_auto_start=True,
    )

    # agent = agent_spec.build_agent()

    for speed in [40, 80, 120, 160]:
        step = 0
        observations = env.reset()
        total_reward = 0.0
        dones = {"__all__": False}

        while not dones["__all__"] and step < 50:
            agent_action = (speed / 3.6, 0)
            observations, rewards, dones, _ = env.step({AGENT_ID: agent_action})
            print(
                f"target_speed: {agent_action[0]}, actual_speed: {observations[AGENT_ID].ego_vehicle_state.speed}"
            )
            total_reward += rewards[AGENT_ID]
        print("*" * 100)
        print("end", step)
        print("Accumulated reward:", total_reward)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
