# SMARTS
![SMARTS CI](https://github.com/junluo-huawei/SMARTS/workflows/SMARTS%20CI/badge.svg?branch=master) ![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

SMARTS (Scalable Multi-Agent RL Training School) is a simulation platform for reinforcement learning and multi-agent research on autonomous driving. Its focus is on realistic and diverse interactions. It is part of the [XingTian](https://github.com/huawei-noah/xingtian/) suite of RL platforms from Huawei Noah's Ark Lab.

Check out the paper at [SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving](https://arxiv.org/abs/2010.09776) for background on some of the project goals.

![](docs/_static/smarts_envision.gif)

## Multi-Agent experiment as simple as...

```python
import gym

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent

class SimpleAgent(Agent):
    def act(self, obs):
        return "keep_lane"

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
    agent_builder=SimpleAgent,
)

agent_specs = {
    "Agent-007": agent_spec,
    "Agent-008": agent_spec,
}

env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/loop"],
    agent_specs=agent_specs,
)

agents = {
    agent_id: agent_spec.build_agent()
    for agent_id, agent_spec in agent_specs.items()
}
observations = env.reset()

for _ in range(1000):
    agent_actions = {
        agent_id: agents[agent_id].act(agent_obs)
        for agent_id, agent_obs in observations.items()
    }
    observations, _, _, _ = env.step(agent_actions)
```

## Setup

```bash
# For Mac OS X users, make sure XQuartz is pre-installed as SUMO's dependency

# git clone ...
cd <project>

# Follow the instructions given by prompt for setting up the SUMO_HOME environment variable
./install_deps.sh

# verify sumo is >= 1.5.0
# if you have issues see ./doc/SUMO_TROUBLESHOOTING.md
sumo

# setup virtual environment; presently only Python 3.7.x is officially supported
python3.7 -m venv .venv

# enter virtual environment to install all dependencies
source .venv/bin/activate

# upgrade pip, a recent version of pip is needed for the version of tensorflow we depend on
pip install --upgrade pip

# install [train] version of python package with the rllib dependencies
pip install -e .[train]

# make sure you can run sanity-test (and verify they are passing)
# if tests fail, check './sanity_test_result.xml' for test report. 
pip install -e .[test]
make sanity-test

# then you can run a scenario, see following section for more details
```

## Running

We use [supervisord](http://supervisord.org/introduction.html) to run SMARTS together with it's supporting processes. To run the default example simply build a scenario and start supervisord:

```bash
# build scenarios/loop
scl scenario build --clean scenarios/loop

# start supervisord
supervisord
```

With `supervisord` running, visit http://localhost:8081/ in your browser to view your experiment.

See [./envision/README.md](./envision/README.md) for more information on Envision, our front-end visualization tool.

Several example scripts are provided under [`SMARTS/examples`](./examples), as well as a handful of scenarios under [`SMARTS/scenarios`](./scenarios). You can create your own scenarios using the [Scenario Studio](./smarts/sstudio). Here's how you can use one of the example scripts with a scenario.

```bash
# Update the command=... in ./supervisord.conf
#
# [program:smarts]
# command=python examples/single_agent.py scenarios/loop
# ...
```

## Documentation

Documentation is available at [smarts.readthedocs.io](https://smarts.readthedocs.io/en/latest)

## CLI tool

SMARTS provides a command-line tool to interact with scenario studio and Envision.

Usage
```
scl COMMAND SUBCOMMAND [OPTIONS] [ARGS]...
```

Commands:
* envision
* scenario
* zoo

Subcommands of scenario:
* build-all: Generate all scenarios under the given directories
* build: Generate a single scenario
* clean: Clean generated artifacts

Subcommands of envision:
* start: start envision server

Subcommands of zoo:
* zoo: Build an agent, used for submitting to the agent-zoo

## Examples:

```
# Start envision, serve scenario assets out of ./scenarios
scl envision start --scenarios ./scenarios

# Build all scenario under given directories
scl scenario build-all ./scenarios ./eval_scenarios

# Rebuild a single scenario, replacing any existing generated assets
scl scenario build --clean scenarios/loop

# Clean generated scenario artifacts
scl scenario clean scenarios/loop
```

## Interfacing with Gym

See the provided ready-to-go scripts under the [examples/](./examples) directory.

## Contributing

Please read [Contributing](CONTRIBUTING.md)

## Bug reports

Please read [how to create a bug report](https://github.com/huawei-noah/SMARTS/wiki/How-To-Make-a-Bug-Report) and then open an issue [here](https://github.com/huawei-noah/SMARTS/issues).

## Building Docs Locally

Assuming you have run `pip install .[dev]`.

```bash
make docs

python -m http.server -d docs/_build/html
# Open http://localhost:8000 in your browser
```

## Extras

## Visualizing Agent Observations
If you want to easily visualize observations you can use our [Visdom](https://github.com/facebookresearch/visdom) integration. Start the visdom server before running your scenario,

```bash
visdom
# Open the printed URL in your browser
```

And in your experiment, start your environment with `visdom=True`

```python
env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/loop"],
    agent_specs=agent_specs,
    visdom=True,
)
```

## Interfacing w/ PyMARL and malib

[PyMARL](https://github.com/oxwhirl/pymarl) and [malib](https://github.com/ying-wen/malib) have been open-sourced. You can run them via,

```bash
git clone git@github.com:ying-wen/pymarl.git

ln -s your-project/scenarios ./pymarl/scenarios

cd pymarl

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/main.py --config=qmix --env-config=smarts
```

```bash
git clone git@github.com:ying-wen/malib.git

ln -s your-project/scenarios ./malib/scenarios

cd malib

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python examples/run_smarts.py --algo SAC --scenario ./scenarios/loop --n_agents 5
```

## Using Docker

If you're comfortable using docker or are on a platform without suitable support to easily run SMARTS (e.g. an older version of Ubuntu) you can run the following,

```bash
$ cd /path/to/SMARTS
$ docker run --rm -it -v $PWD:/src -p 8081:8081 huaweinoah/smarts:<version>
# E.g. docker run --rm -it -v $PWD:/src -p 8081:8081 huaweinoah/smarts:v0.4.12
# <press enter>

# Run Envision server in the background
# This will only need to be run if you want visualisation
$ scl envision start -s ./scenarios -p 8081 &

# Build an example
# This needs to be done the first time and after changes to the example
$ scl scenario build scenarios/loop --clean

# Run an example
# add --headless if you do not need visualisation
$ python examples/single_agent.py scenarios/loop

# On your host machine visit http://localhost:8081 to see the running simulation in
# Envision.
```

(For those who have permissions:) if you want to push new images to our [public dockerhub registry](https://hub.docker.com/orgs/huaweinoah) run,

```bash
# For this to work, your account needs to be added to the huaweinoah org
docker login

export VERSION=v0.4.3-pre
docker build --no-cache -t smarts:$VERSION .
docker tag smarts:$VERSION huaweinoah/smarts:$VERSION
docker push huaweinoah/smarts:$VERSION
```

## Troubleshooting

```bash
ps -ef | grep ray && ps -ef | grep smarts && ps -ef | grep ultra
pkill -f -9 smarts && pkill -f -9 ray

python ultra/train.py --task 1 --level easy --policy ppo --eval-rate 5 --eval-episodes 2 --episodes 20

start of command: nohup python -u 
end of command: > sample.txt &

tail -f sample.txt
```

## Setting up SMARTS/ULTRA on Compute Canada

### Running SMARTS on Compute Canada

```bash
# Login to Compute Canada with Trusted X11 Forwarding and the forwarded port for Envision.
$ ssh <user-name>@<cluster-name>.computecanada.ca -Y -L localhost:8081:localhost:8081

# On your Compute Canada login node, obtain the Docker image for SMARTS and compress it (taken from
# https://docs.computecanada.ca/wiki/Singularity#Creating_an_image_using_Docker_Hub_and_Dockerfile).
$ cd ~/scratch
$ wget https://raw.githubusercontent.com/moby/moby/master/contrib/download-frozen-image-v2.sh
$ sh download-frozen-image-v2.sh smarts-0416_docker huaweinoah/smarts:v0.4.16
$ cd smarts-0416_docker && tar cf ../smarts-0416_docker.tar * && cd ..

# Start an interactive job and build the Singularity container.
$ cd ~/scratch
$ salloc --mem-per-cpu=2000 --cpus-per-task=4 --time=2:0:0
$ module load singularity
$ singularity build smarts-0416_singularity.sif docker-archive://smarts-0416_docker.tar
$ exit  # Exit out of the interactive job once the Singularity container is built.

# Move the Singularity container back to your projects directory and clone SMARTS.
$ cd ~/scratch
$ mv smarts-0416_singularity.sif ~/projects/<sponsor-name>/<user-name>/
$ cd ~/projects/<sponsor-name>/<user-name>/
$ git clone https://github.com/Dikshuy/SMARTS-lite.git

# Execute the Singularity container and bind your SMARTS directory to the /SMARTS directory in the container.
# After, go to your the SMARTS directory in the container, modify the PYTHONPATH, and run an example!
$ cd ~/projects/<sponsor-name>/<user-name>/
$ singularity shell --bind SMARTS-lite/:/SMARTS-lite --env DISPLAY=$DISPLAY smarts-0416_singularity.sif
Singularity> cd /SMARTS-lite
Singularity> export PYTHONPATH=/SMARTS-lite:$PYTHONPATH
Singularity> supervisord
```

### Running ULTRA on Compute Canada

Follow the steps above to obtain `smarts-0416_singularity.sif` and `SMARTS-lite/`.

```bash
# Start an interactive job to run an ULTRA experiment.
$ salloc --time=1:0:0 --mem=16G --cpus-per-task=8 --ntasks=1
$ module load singularity
$ singularity shell --bind SMARTS-lite/:/SMARTS-lite --env DISPLAY=$DISPLAY smarts-0416_singularity.sif
Singularity> cd /SMARTS-lite/ultra
Singularity> export PYTHONPATH=/SMARTS-lite/ultra:/SMARTS-lite/:$PYTHONPATH

# Follow instructions in https://github.com/huawei-noah/SMARTS/blob/master/ultra/docs/getting_started.md to
# run the experiment.
```

### Running jobs on CC using `sbatch`

```bash
#!/bin/bash
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --account=def-mtaylor3
#SBATCH --mem=32000M
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2

module load singularity
singularity exec -B ../SMARTS-lite:/SMARTS-lite --env DISPLAY=$DISPLAY,PYTHONPATH=/SMARTS-lite/ultra:/SMARTS-lite:$PYTHONPATH --home /SMARTS-lite/ultra ../smarts-0416_singularity.sif python ultra/hammer_train.py --task 0-3agents --level easy --policy ppo,ppo,ppo --headless
```

## Note
You can shut down this Envision process by running `pkill -f -9 ultra` (notice that `ps -ef | grep ultra` will output the Envision process that you started with the command `./ultra/env/envision_base.sh`). But if you kill this Envision process, you will have to rerun `./ultra/env/envision_base.sh` if you want to be able to visualize the training through Envision again.

## Adapters

An adapter is a function that receives a environment observation, environment reward,
and/or action from an agent, and then manipulates them (often by extracting or adding
relevant information) so that they can be processed by the agent or the environment.

### Action Adapters

An action adapter takes an action from an agent and adapts it so that it conforms to the
SMARTS simulator's action format. ULTRA has two default action adapters, one for
continuous action, and another for discrete action.

#### [ultra.adapters.default_action_continuous_adapter](../ultra/adapters/default_action_continuous_adapter.py)

The default continuous action adapter requires the agent has a "continuous" action space
as defined by SMARTS. Therefore, when using this adapter, the [`AgentInterface`](../../smarts/core/agent_interface.py)
of your agent needs its `action` parameter to be `ActionSpaceType.Continuous`. This
requirement is outlined in this module's `required_interface`.

SMARTS' `ActionSpaceType.Continuous` accepts actions in the form of a NumPy array with
shape (3,). That is, the action is a NumPy array of the form
`[throttle, brake, steering]` where throttle is a `float` in the range `[0, 1]`, brake 
is a `float` in the range `[0, 1]`, and steering is a `float` in the range `[-1, 1]`
(see [SMARTS action documentation](https://smarts.readthedocs.io/en/latest/sim/observations.html#actions)
for more on the behaviour of each part of the action). This action space is outlined in
this module's `gym_space`.

The behaviour of this action adapter is simply to return the action it was given,
without adapting it any further. It expects that the action outputted by the agent
already conforms to this NumPy array of shape (3,). The behaviour of this adapter is
fully defined in this module's `adapt` function.

#### [ultra.adapters.default_action_discrete_adapter](../ultra/adapters/default_action_discrete_adapter.py)

The default discrete action adapter requires the agent has a "lane" action space as
defined by SMARTS. Therefore, when using this adapter, the [`AgentInterface`](../../smarts/core/agent_interface.py)
of your agent needs its `action` parameter to be `ActionSpaceType.Lane`. This
requirement is outlined in this module's `required_interface`.

SMARTS' `ActionSpaceType.Lane` accepts actions in the form of a string. The string must
either be `"keep_lane"`, `"slow_down"`, `"change_lane_left"`, or `"change_lane_right"`
where the behaviour of each action can be inferred from the action name (see
[SMARTS action documentation](https://smarts.readthedocs.io/en/latest/sim/observations.html#actions)
for more on the behaviour of each action). This action space is outlined in this
module's `gym_space`.

The behaviour of this action adapter is simply to return the action it was given,
without adapting it any further. It expects that the action outputted by the agent
already is one of the four available strings. The behaviour of this adapter is fully
defined in this module's `adapt` function.

### Info Adapters

An info adapter takes an [observation](https://smarts.readthedocs.io/en/latest/sim/observations.html#id1), reward, and info dictionary from the environment and adapts them to include
more relevant information about the agent at each step. By default, the ULTRA
environment includes the ego vehicle's raw observation, and its score in the info
dictionary.
```python
info = {
    "score": ...,  # A float, the total distance travelled by the ego vehicle.
    "env_obs": ...,  # A smarts.core.sensors.Observation, the raw observation received by the ego vehicle.
}
```

ULTRA has a default info adapter that is used to include more data about the
agent that can be used to track the agent's learning progress and monitor the agent
during training and evaluation.

#### [ultra.adapters.default_info_adapter](../ultra/adapters/default_info_adapter.py)

The default info adapter requires that the SMARTS environment include the next 20
waypoints in front of the ego vehicle, and all neighborhood (social) vehicles within a
radius of 200 meters around the ego vehicle. Therefore, when using this adapter, the
[AgentInterface](../../smarts/core/agent_interface.py) of your agent needs its
`waypoints` parameter to be `Waypoints(lookahead=20)` and its `neighborhood_vehicles`
parameter to be `NeighborhoodVehciles(radius=200.0)`. This requirement is outlined in
this module's `required_interface`.

The default info adapter modifies the given info dictionary passed to it by the
environment. Specifically, it adds another key, "logs", to the info dictionary. This
key's values is another dictionary that contains information about the agent:
```python
info = {
    "score": ...,  # A float, the total distance travelled by the ego vehicle.
    "env_obs": ...,  # A smarts.core.sensors.Observation, the raw observation received by the ego vehicle.
    "logs": {
        "position": ...,  # A np.ndarray with shape (3,), the x, y, z position of the ego vehicle.
        "speed": ...,  # A float, the speed of the ego vehicle in meters per second.
        "steering": ...,  # A float, the angle of the front wheels in radians.
        "heading": ...,  # A smarts.core.coordinates.Heading, the vehicle's heading in radians.
        "dist_center": ...,  # A float, the distance in meters from the center of the lane of the closest waypoint.
        "start": ...,  # A smarts.core.scenario.Start, the start of the ego evhicle's mission.
        "goal": ...,  # A smarts.core.scenario.PositionalGoal, the goal of the ego vehicle's mission.
        "closest_wp": ...,  # A smarts.core.waypoints.Waypoint, the closest waypoint to the ego vehicle.
        "events": ...,  # A smarts.core.events.Events, the events of the ego vehicle.
        "ego_num_violations": ...,  # An int, the number of violations committed by the ego vehicle (see ultra.utils.common.ego_social_safety).
        "social_num_violations": ...,  # An int, the number of violations committed by social vehicles (see ultra.utils.common.ego_social_safety).
        "goal_dist": ...,  # A float, the euclidean distance between the ego vehicle and its goal.
        "linear_jerk": ...,  # A float, the magnitude of the ego vehicle's linear jerk.
        "angular_jerk": ...,  # A float, the magnitude of the ego vehicle's angular jerk.
        "env_score": ...,  # A float, the ULTRA environment's reward obtained from the default reward adapter (see ultra.adapters.default_reward_adapter).
    }
}
```

This information contained in logs can ultimately be used by ULTRA's [Episode](../ultra/utils/episode.py)
object that is used to record this data to Tensorboard and also save this data to a
serializable object.

### Observation Adapters

An observation adapter takes an [observation](https://smarts.readthedocs.io/en/latest/sim/observations.html#id1)
from the environment and adapts it so that it can be processed by the agent. ULTRA has
two default observation adapters, one that adapts the observation containing a top-down
RGB image into a gray-scale version of the same image, and another that adapts the
observation into a dictionary of vectors.

#### [ultra.adapters.default_observation_image_adapter](../ultra/adapters/default_observation_image_adapter.py)

The default image observation adapter requires that the SMARTS environment include the
top-down RGB image in the agent's observation. Specifically, the image should be of
shape 64x64 with a resolution of 50 / 64. Therefore, when using this adapter, the
[`AgentInterface`](../../smarts/core/agent_interface.py) of your agent needs its `rgb`
parameter to be `RGB(width=64, height=64, resolution=(50 / 64))`. This requirement is
outlined in this module's `required_interface`.

This default image observation adapter produces a NumPy array of type `float32` and with
shape `(4, 64, 64)`. Each element of the array is normalized to be in the range
`[0, 1]`. This observation space is outlined in this module's `gym_space`.

This adapter receives an observation from the environment that contains a
`smarts.core.sensors.TopDownRGB` instance in the observation. The `data` attribute of
this class is a NumPy array of type `uint8` and shape `(4, 64, 64, 3)`. The adapter
converts this array to gray-scale by dotting it with `(0.1, 0.8, 0.1)`, resulting in the
value of each gray-scale pixel to be a linear combination of the red (R), green (G), and
blue (B) components of that pixel: `0.1 * R + 0.8 * G + 0.1 * B`. This gray-scale
weighting was chosen to accentuate the differences in gray values between the ego
vehicle, social vehicles, and the road. The gray-scale image is then normalized by
dividing the array by `255.0`. The output is a NumPy array of type `float32` and with
shape `(4, 64, 64)`. The most recent frame is at the highest index of this array. The
behaviour of this adapter is fully defined in this module's `adapt` function.

#### [ultra.adapters.default_observation_vector_adapter](../ultra/adapters/default_observation_vector_adapter.py)

The default vector observation adapter requires that the SMARTS environment include the
next 20 waypoints in front of the ego vehicle, and all neighborhood (social) vehicles
within a radius of 200 meters around the ego vehicle. Therefore, when using this
adapter, the [`AgentInterface`](../../smarts/core/agent_interface.py) of your agent
needs its `waypoints` parameter to be `Waypoints(lookahead=20)` and its
`neighborhood_vehicles` parameter to be `NeighborhoodVehicles(radius=200.0)`. This
requirement is outlined in this module's `required_interface`.

In addition to these aforementioned requirements, the observation, by default, contains
information about the ego vehicle's state. Provided that the observation has the
aforementioned requirements and the ego vehicle's state, this adapter adapts this
observation to a dictionary:
```python
{
    "low_dim_states": [
        ego_vehicle_speed / 30.0,
        distance_from_center / 1.0,
        steering / 3.14,
        angle_error / 3.14,
        relative_goal_position_x / 100.0,
        relative_goal_position_y / 100.0,
        relative_waypoint_position_x / 10.0,  # Nearest waypoint.
        relative_waypoint_position_y / 10.0,  # Nearest waypoint.
        relative_waypoint_position_x / 10.0,  # 2nd closest waypoint.
        relative_waypoint_position_y / 10.0,  # 2nd closest waypoint.
        ...
        relative_waypoint_position_x / 10.0,  # 20th closest waypoint.
        relative_waypoint_position_y / 10.0,  # 20th closest waypoint.
        road_speed / 30.0,
    ],
    "social_vehicles": [
        [
            relative_vehicle_position_x / 100.0,
            relative_vehicle_position_y / 100.0,
            heading_difference / 3.14,
            social_vehicle_speed / 30.0
        ],  # Closest social vehicle.
        [
            relative_vehicle_position_x / 100.0,
            relative_vehicle_position_y / 100.0,
            heading_difference / 3.14,
            social_vehicle_speed / 30.0
        ],  # 2nd closest social vehicle.
        ...
        [
            relative_vehicle_position_x / 100.0,
            relative_vehicle_position_y / 100.0,
            heading_difference / 3.14,
            social_vehicle_speed / 30.0
        ],  # 10th closest social vehicle.
    ]
}
```

Where:
- `ego_vehicle_speed` is the speed of the ego vehicle in meters per second.
- `distance_from_center` is the lateral distance between the center of the closest
waypoint's lane and the ego vehicle's position, divided by half of that lane's width.
- `steering` is the angle of ego vehicle's front wheels in radians.
- `angle_error` is the closest waypoint's heading minus the ego vehicle's heading.
- `relative_goal_position_x` is the x component of the vector obtained by calculating
the goal's `(x, y)` position minus the ego vehicle's `(x, y)` position, and rotating
that difference by the negative of the ego vehicle's heading. All in all, this keeps
this component completely relative from the ego vehicle's perspective.
- `relative_goal_position_y` is the y component of the vector obtained by calculating
the goal's `(x, y)` position minus the ego vehicle's `(x, y)` position, and rotating
that difference by the negative of the ego vehicle's heading. All in all, this keeps
this component completely relative from the ego vehicle's perspective.
- `relative_waypoint_position_x` is the x component of the vector obtained by
calculating the waypoint's `(x, y)` position minus the ego vehicle's `(x, y)` position,
and rotating that difference by the negative of the ego vehicle's heading. All in all,
this keeps the component completely relative from the ego vehicle's perspective.
- `relative_waypoint_position_y` is the y component of the vector obtained by
calculating the waypoint's `(x, y)` position minus the ego vehicle's `(x, y)` position,
and rotating that difference by the negative of the ego vehicle's heading. All in all,
this keeps the component completely relative from the ego vehicle's perspective.
- `road_speed` is the speed limit of the closest waypoint.
- `relative_vehicle_position_x` is the x component of vector obtained by calculating the
social vehicle's `(x, y)` position minus the ego vehicle's `(x, y)` position, and
rotating that difference by the negative of the ego vehicle's heading. All in all, this
keeps this component completely relative from the ego vehicle's perspective.
- `relative_vehicle_position_y` is the y component of vector obtained by calculating the
social vehicle's `(x, y)` position minus the ego vehicle's `(x, y)` position, and
rotating that difference by the negative of the ego vehicle's heading. All in all, this
keeps this component completely relative from the ego vehicle's perspective.
- `heading_difference` is the heading of the social vehicle minus the heading of the ego
vehicle.
- `social_vehicle_speed` is the speed of the social vehicle in meters per second.

Notice that the social vehicles are sorted by relative distance to the ego vehicle. This
was chosen under the assumption that the closest social vehicles are the ones that the
ego vehicle should pay attention to. While this is likely true in most situations, this
assumption may not be the most accurate in all cases. For example, if all the nearest
social vehicles are behind the ego vehicle, the ego vehicle will not observe any social
vehicles ahead of itself.

If the observation provided by the environment contains less than 10 social vehicles
(that is, there are less than 10 social vehicles in a 200 meter radius around the ego
vehicle), this adapter will pad the social vehicle adaptation with zero-vectors for the
remaining rows. For example, if there are no social vehicles in the observation from the
environment, the social vehicle adaptation would be a `(10, 4)` NumPy array with data:
`[[0., 0., 0., 0.], [0., 0., 0., 0.], ..., [0., 0., 0., 0.]]`.

If there are more than 10 social vehicles, this adapter will truncate the social vehicle
adaptation to only include 10 rows - the features of the 10 nearest social vehicles.

### Reward Adapters

A reward adapter takes an [observation](https://smarts.readthedocs.io/en/latest/sim/observations.html#id1)
and the [environment reward](https://smarts.readthedocs.io/en/latest/sim/observations.html#rewards)
as arguments from the environment and adapts them, acting as a custom reward function.
ULTRA has one default reward adapter that uses elements from the agent's observation, as
well as the environment reward, to develop a custom reward.

#### [ultra.adapters.default_reward_adapter](../ultra/adapters/default_reward_adapter.py)

The default reward adapter requires that the SMARTS environment include the next 20
waypoints in front of the ego vehicle in the ego vehicle's observation. Therefore, when
using this adapter, the [`AgentInterface`](../../smarts/core/agent_interface.py) of your
agent needs its `waypoints` parameter to be `Waypoints(lookahead=20)`. This requirement
is outlined in this module's `required_interface`.

This default reward adapter combines elements of the agent's observation along with the
environment reward to create a custom reward. This custom reward consists of multiple
components:
```python
custom_reward = (
    ego_goal_reward +
    ego_collision_reward +
    ego_off_road_reward +
    ego_off_route_reward +
    ego_wrong_way_reward +
    ego_speed_reward +
    ego_distance_from_center_reward +
    ego_angle_error_reward +
    ego_reached_goal_reward +
    ego_step_reward +
    environment_reward
)
```

Where:
- `ego_goal_reward` is `0.0`
- `ego_collison_reward` is `-1.0` if the ego vehicle has collided, else `0.0`.
- `ego_off_road_reward` is `-1.0` if the ego vehicle is off the road, else `0.0`.
- `ego_off_route_reward` is `-1.0` if the ego vehicle is off its route, else `0.0`.
- `ego_wrong_way_reward` is `-0.02` if the ego vehicle is facing the wrong way, else
`0.0`.
- `ego_speed_reward` is `0.01 * (speed_limit - ego_vehicle_speed)` if the ego vehicle is
going faster than the speed limit, else `0.0`.
- `ego_distance_from_center_reward` is `-0.002 * min(1, abs(ego_distance_from_center))`.
- `ego_angle_error_reward` is `-0.0005 * max(0, cos(angle_error))`.
- `ego_reached_goal_reward` is `1.0` if the ego vehicle has reached its goal, else
`0.0`.
- `ego_step_reward` is
`0.02 * min(max(0, ego_vehicle_speed / speed_limit), 1) * cos(angle_error)`.
- `environment_reward` is `the_environment_reward / 100`.

And `speed_limit` is the speed limit of the nearest waypoint to the ego vehicle in
meters per second; the `ego_vehicle_speed` is the speed of the ego vehicle in meters per
second; the `angle_error` is the closest waypoint's heading minus the ego vehicle's
heading; the `ego_distance_from_center` is the lateral distance between the center
of the closest waypoint's lane and the ego vehicle's position, divided by half of that
lane's width; and `the_environment_reward` is the raw reward received from the SMARTS
simulator.

### General
In many cases additinal run logs are located at '~/.smarts'. These can sometimes be helpful.

### SUMO
SUMO can have some problems in setup. Please look through the following for support for SUMO:
* If you are having issues see: **[SETUP](docs/setup.rst)** and **[SUMO TROUBLESHOOTING](docs/SUMO_TROUBLESHOOTING.md)**
* If you wish to find binaries: **[SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php)**
* If you wish to compile from source see: **[SUMO Build Instructions](https://sumo.dlr.de/docs/Developer/Main.html#build_instructions)**. 
    * **Please note that building SUMO may not install other vital dependencies that SUMO requires to run.**
    * If you build from the git repository we recommend you use: **[SUMO version 1.7.0](https://github.com/eclipse/sumo/commits/v1_7_0)** or higher

## Citing SMARTS

If you use SMARTS in your research, please cite the [paper](https://arxiv.org/abs/2010.09776). In BibTeX format:

```bibtex
@misc{zhou2020smarts,
      title={SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving},
      author={Ming Zhou and Jun Luo and Julian Villella and Yaodong Yang and David Rusu and Jiayu Miao and Weinan Zhang and Montgomery Alban and Iman Fadakar and Zheng Chen and Aurora Chongxi Huang and Ying Wen and Kimia Hassanzadeh and Daniel Graves and Dong Chen and Zhengbang Zhu and Nhat Nguyen and Mohamed Elsayed and Kun Shao and Sanjeevan Ahilan and Baokuan Zhang and Jiannan Wu and Zhengang Fu and Kasra Rezaee and Peyman Yadmellat and Mohsen Rohani and Nicolas Perez Nieves and Yihan Ni and Seyedershad Banijamali and Alexander Cowen Rivers and Zheng Tian and Daniel Palenicek and Haitham bou Ammar and Hongbo Zhang and Wulong Liu and Jianye Hao and Jun Wang},
      url={https://arxiv.org/abs/2010.09776},
      primaryClass={cs.MA},
      booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
      year={2020},
      month={11}
 }
```
