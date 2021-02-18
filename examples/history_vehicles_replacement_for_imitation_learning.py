import logging

from smarts.core.smarts import SMARTS
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.scenario import Scenario
from smarts.core.scenario import Mission
from envision.client import Client as Envision

from examples import default_argument_parser


logging.basicConfig(level=logging.INFO)


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def main(scenarios, headless, seed):
    scenarios_iterator = Scenario.scenario_variations(scenarios, [])
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(headless=True, auto_start=True),
        envision=Envision(),
    )
    for _ in scenarios:
        scenario = next(scenarios_iterator)
        agent_missions = scenario.discover_missions_of_traffic_histories()

        for agent_id, mission in agent_missions.items():
            agent_spec = AgentSpec(
                interface=AgentInterface.from_type(
                    AgentType.Laner, max_episode_steps=None
                ),
                agent_builder=KeepLaneAgent,
            )
            agent = agent_spec.build_agent()

            smarts.switch_ego_agent({agent_id: agent_spec.interface})
            smarts.history_set_start_elapsed_time(mission.start_time)
            modified_mission = Mission(
                goal=agent_missions[agent_id].goal,
                start_time=.0,
                start=agent_missions[agent_id].start,
            )
            scenario.set_ego_missions({agent_id: modified_mission})
            observations = smarts.reset(scenario)

            dones = {agent_id: False}
            while not dones[agent_id]:
                agent_obs = observations[agent_id]
                agent_action = agent.act(agent_obs)

                observations, rewards, dones, infos = smarts.step(
                    {agent_id: agent_action}
                )

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("history-vehicles-replacement-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios, headless=args.headless, seed=args.seed,
    )
