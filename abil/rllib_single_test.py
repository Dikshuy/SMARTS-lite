import argparse
import logging
from typing import Dict

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID

from rllib_agent import TrainingModel, rllib_agent

import gym
from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)


# Add custom metrics to your tensorboard using these callbacks
# See: https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
class Callbacks(DefaultCallbacks):
    @staticmethod
    def on_episode_start(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):

        episode.user_data["ego_speed"] = []

    @staticmethod
    def on_episode_step(
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):

        single_agent_id = list(episode._agent_to_last_obs)[0]
        obs = episode.last_raw_obs_for(single_agent_id)
        episode.user_data["ego_speed"].append(obs["speed"])

    @staticmethod
    def on_episode_end(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):

        mean_ego_speed = np.mean(episode.user_data["ego_speed"])
        print(
            f"ep. {episode.episode_id:<12} ended;"
            f" length={episode.length:<6}"
            f" mean_ego_speed={mean_ego_speed:.2f}"
        )
        episode.custom_metrics["mean_ego_speed"] = mean_ego_speed


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["rollout_fragment_length"] * 2:
        config["train_batch_size"] = config["rollout_fragment_length"] * 2
    return config


def main(
    scenario,
    headless,
    sumo_headless,
    seed,
):
    agent_specs = {'AGENT-0': rllib_agent['agent_spec']}
    agent_params=rllib_agent['agent_spec'].agent_params
    print(f'\n\nagent_spec.agent_params: {agent_params}\n\n')
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario],
        agent_specs=agent_specs,
        sim_name="test_game_of_tag",
        headless=headless,
        sumo_headless=sumo_headless,
        seed=seed,
    )

    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    score_sum = 0
    num_episodes = 50
    for episode in episodes(n=num_episodes):
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        score = 0
        while not dones["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }

            observations, rewards, dones, infos = env.step(actions)
            episode.record_step(observations, rewards, dones, infos)
            score += rewards['AGENT-0']
        score_sum += score
    print(f"Average Score: {score_sum/num_episodes}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("rllib-example")
    parser.add_argument(
        "scenario",
        help="Scenario to run (see scenarios/ for some samples you can use)",
        type=str,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run Envision simulation in headless mode",
    )
    parser.add_argument(
        "--sumo_headless",
        action="store_true",
        default=True,
        help="Run sumo simulation in headless mode",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The base random seed to use, intended to be mixed with --num_samples",
    )

    args = parser.parse_args()
    main(
        scenario=args.scenario,
        headless=args.headless,
        sumo_headless=args.sumo_headless,
        seed=args.seed,
    )
