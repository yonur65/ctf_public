# Capture the Flag Gridworld Environment (gym-cap)

This gridworld environment is specifically designed to simulate the multi-agent adversarial environment.
The environment mimics the capture the flag setup, where the main objective is to capture the enemy's flag as soon as possible.
The game also ends when all the enemy's agents are captured.
The game has many stochastic transitions, such as the interaction between agents, territorial advantage, or random initialization of the map.
A reinforcement learning implementation can be found [here](https://github.com/raide-project/ctf_RL).

## Package Install

The package and required dependencies can be installed using pip:

``` sh
pip install gym-cap
```

## Requirements

All the dependencies will be installed if the package is installed with pip.
If the package is installed using source code, following packages are required:

- Python 3.7+
- Numpy 1.18+
- OpenAI Gym 0.16+

* Gym might require additional packages depending on OS.

## Publications and Submissions

Please use this bibtex if you want to cite this environment package in your publications:

```
@misc{gym-cap,
    author = {Seung Hyun Kim, Neale van Stralen, Tran Research Group},
    title = {Gridworld Capture the Flag Environment},
    year = {2019},
    url = {\url{https://github.com/raide-project/ctf_public}},
}
```

- [Evaluating Adaptation Performance of Hierarchical Deep ReinforcementLearning]() (ICRA 2020)
    - [Demonstration Video]()
    - [Presentation Video]()

## Preparation

Run the example code [(demo run)](demo/test.py) to test if the package is installed correctly.
Basic policies are provided in `gym_cap.heuristic`.

## Environment Description

![Rendering Example](figures/rendering_example.png)

- The environment takes input as a tuple.
    - The number of element in tuple must match the total number of blue agent. (UAV+UGB)
    - The action must be in range between 0 and 4.
- If UAV is included, the UAV's action comes __in front__ of UGV's action.
    - ex) To make UAV to hover (fix): action = [0, 0] + [UGV's action]

## Custom Policy

To control the __blue agent__, the action can be specified at each step `env.step([0,3,2,4])`.
A custom policy could also be created in module to play the game. 
The example of custom policy can be found in [(custom policy)](demo/demo_policy.py).

## Environment Configurations

### Environment Parameters

The environment is mostly fixed with the default configuration parameters, but some parameters are possible to be modified.
When the environment method `env.reset()` is called, the environment is initialized as same as the previous configuration.
Prescribed parameters can be modified by passing `config_path=custom_config.ini` argument.

``` py
observation = env.reset(config_path='custom_config.ini')
```

Here is the example of custom configuration file `custom_config.ini`.

``` py
# Controllable Variables

[elements]
NUM_BLUE = 4                # number of ground blue agent
NUM_RED = 4                 # number of ground red agent
NUM_BLUE_UAV = 2            # number of air blue agent
NUM_RED_UAV = 2             # number of air red agent


[control]
MAX_STEP = 150              # maximum number of steps in each game
BLUE_ADV_BIAS = 0           # team blue advantage in the interaction
RED_ADV_BIAS = 0            # team red advantage in the interaction

[memory]
TEAM_MEMORY = None          # if set to fog, team observation includes previously visited static environment
RENDER_TEAM_MEMORY = False  # on/off: render team memory

[settings]
STOCH_TRANSITIONS = False   # on/off: stochastic transition
STOCH_TRANSITIONS_EPS = 0.1 # stochastic transition rate
STOCH_ATTACK = True         # on/off: stochastic interaction
STOCH_ATTACK_BIAS = 1       # territorial advantage in the stochastic interaction
STOCH_ZONES = False         # on/off: randomize map generation. (if custom_board is give, this parameter is ignored)
BLUE_PARTIAL = True         # on/off: partial observation for team blue
RED_PARTIAL = True          # on/off: partial observation for team red
```

### Custom Map

The environment can be re-initialized to custom board by passing the `.txt` file.
If customized board is given, prescribed number of agents and terrain information will be ignored.

The path of the customized board can be given as
```py
observation = env.reset(custom_board='test_maps/board2.txt')
```

The example of the custom board can be found in `demo/test_map` directory.
```py
0 0 2 4 1 4 1 1 1
2 2 8 8 4 1 1 1 1
0 0 8 8 1 1 1 1 1
6 0 0 1 1 7 0 0 0
0 0 0 1 8 8 0 0 0
0 0 2 4 8 8 0 0 0
1 1 1 0 0 0 1 1 1
1 1 1 0 0 0 1 1 1
1 1 1 0 0 0 1 1 1
```

* board elements are separated by space.

```
# Element ID
TEAM1_BACKGROUND = 0
TEAM2_BACKGROUND = 1
TEAM1_UGV = 2
TEAM1_UAV = 3
TEAM2_UGV = 4
TEAM2_UAV = 5
TEAM1_FLAG = 6
TEAM2_FLAG = 7
OBSTACLE = 8
```

* The background of the agents is set to team background.

## Advanced Features

### Multi-Agent Communication Settings (work in progress)

```py
agent.get_obs(self, env, com_ground=False, com_air=False, distance=None, freq=1.0, *args)
```

The method returns the observation for a specific agent. If communication is allowed between ground or air, observation for that agent is expanded to include vision from other agents.

Parameters:

- com_ground and com_air (boolean): toggle communication between ground/air units. 
- distance (int): the maximum distance between units for which communication is  
- freq (0-1.0): the probability that communication goes through    

### Policy Evaluation

The demo script `policy_eval.py` provides basic analysis between two policies.

Example)
```bash
$ python policy_eval.py --episode 50 --blue_policy roomba

Episodes Progress Bar

100%|██████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 14.66it/s]
-------------------------------------------- Statistics --------------------------------------
win # overall in 50 episodes: {'BLUE': 31, 'NEITHER': 1, 'RED': 18}
win # in capturing flag    : {'BLUE': 4, 'NEITHER': 15, 'RED': 31}
win # in killing other team: {'BLUE': 14, 'NEITHER': 36}
time per episode: 0.06638088703155517 s
total time: 3.5574886798858643 s
mean steps: 3318.1
```

Valid Arguments:

- episode: number of iterations to analyze (default: 1000)
- blue_policy: policy to be implmented for blue team (default: random)
- red_policy: policy to be implmented for blue team (default: random)
- num_blue: number of blue ugv agents (default: 4)
- num_red: number of red ugv agents (default: 4)
- num_uav: number of uav agents (default: 0)
- map_size: size of map (default: 20)
- time_step: maximum number of steps per iteration to be completed by the teams (default: 150)


### Advanced Agent Types (work in progress)

### Rendering (work in progress)

### Multi-processing (work in progress)

### Gazebo (work in progress)

## Acknowledgement

## License

This project is licensed under the terms of the [University of Illinois/NCSA Open Source License](./LICENSE.md).

