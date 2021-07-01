import logging
from smarts.core.controllers import ActionSpaceType

import gym

from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles, Waypoints
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from ultra.baselines.adapter import BaselineAdapter
from ultra.baselines.ppo.ppo.policy import PPOPolicy

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.edge_id != obs.via_data.near_via_points[0].edge_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    # agent_spec = AgentSpec(
    #     interface=AgentInterface.from_type(
    #         AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
    #     ),
    #     agent_builder=ChaseViaPointsAgent,
    # )

    adapter = BaselineAdapter("ppo")
    agent_spec = AgentSpec(
        interface=AgentInterface(
            waypoints=Waypoints(lookahead=20),
            neighborhood_vehicles=NeighborhoodVehicles(radius=200),
            action=ActionSpaceType.Continuous,
            max_episode_steps=max_episode_steps,
        ),
        agent_builder=PPOPolicy,
        agent_params={"policy_params": adapter.policy_params},
        observation_adapter=adapter.observation_adapter,
        reward_adapter=adapter.reward_adapter,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        # envision_record_data_replay_path="./data_replay",
    )

    agent = agent_spec.build_agent()

    for episode in episodes(n=num_episodes):
        observations = env.reset()
        print(observations)
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]

            agent_action = agent.act(agent_obs)
            next_observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            # print(infos)
            reached_max_episode_steps = infos[AGENT_ID]["env_obs"].events.reached_max_episode_steps
            if reached_max_episode_steps:
                dones[AGENT_ID] = False
            agent.step(
                state = observations[AGENT_ID],
                action = agent_action,
                reward = rewards[AGENT_ID],
                next_state = next_observations[AGENT_ID],
                done=dones[AGENT_ID],
                info = infos[AGENT_ID],
            )
            episode.record_step(observations, rewards, dones, infos)
            observations = next_observations

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
