import __future__

import io
import configparser
import os
import pkg_resources

import random
import sys
import traceback
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from .agent import *
from .create_map import gen_random_map, custom_map
from .const import *

"""
Requires that all units initially exist in home zone.
"""


class CapEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        'video.frames_per_second' : 50
    }

    ACTION = ["X", "N", "E", "S", "W"]
    CHANNEL = {
            "UNKNOWN": 0,
            "FOG": 0,
            "DEAD": 0,
            "TEAM1_BACKGROUND": 1,
            "TEAM2_BACKGROUND": 1,
            "TEAM1_FLAG": 2,
            "TEAM2_FLAG": 2,
            "OBSTACLE": 3,
            "TEAM1_UGV": 4,
            "TEAM2_UGV": 4,
            "TEAM1_UAV": 5,
            "TEAM2_UAV": 5,
            "TEAM1_UGV2": 4,
            "TEAM2_UGV2": 4,
            "TEAM1_UGV3": 4,
            "TEAM2_UGV3": 4,
            "TEAM1_UGV4": 4,
            "TEAM2_UGV4": 4,
       }

    # Represented constant
    REPRESENT = {
            "UNKNOWN": 1,
            "DEAD": 0,
            "FOG": -1,
            "OBSTACLE": 1,
            "TEAM1_BACKGROUND": 1,
            "TEAM2_BACKGROUND": -1,
            "TEAM1_FLAG": 1,
            "TEAM2_FLAG": -1,
            "TEAM1_UGV": 1,
            "TEAM2_UGV": -1,
            "TEAM1_UAV": 1,
            "TEAM2_UAV": -1,
            "TEAM1_UGV2": 1,
            "TEAM2_UGV2": -1,
            "TEAM1_UGV3": 1,
            "TEAM2_UGV3": -1,
            "TEAM1_UGV4": 1,
            "TEAM2_UGV4": -1,
        }

    def __init__(self, map_size=20, mode="random", **kwargs):
        """
        Parameters
        ----------
        self    : object
            CapEnv object
        """
        self.seed()
        self.viewer = None

        # Default Configuration
        config_path = pkg_resources.resource_filename(__name__, 'default.ini')
        self.config_path = config_path
        self._parse_config(config_path)

        self.blue_memory = np.zeros((map_size, map_size), dtype=bool)
        self.red_memory = np.zeros((map_size, map_size), dtype=bool)

        self._policy_blue = None
        self._policy_red = None

        self._blue_trajectory = []
        self._red_trajectory = []

        self.reset(map_size, mode=mode, **kwargs)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _parse_config(self, config_path=None):
        """
        Parse Configuration

        If configuration file is explicitly provided, any given settings overide the default parameters.
        Default parameters can be found in const.py module

        To add additional configuration parameter, make sure to add the default value in const.py

        The parameter is accessible as class variable.
        """
        # Configurable parameters
        config_param = {
                'elements': [
                    'NUM_BLUE',
                    'NUM_RED',
                    'NUM_BLUE_UAV',
                    'NUM_RED_UAV',
                    'NUM_GRAY',
                    'NUM_BLUE_UGV2',
                    'NUM_RED_UGV2',
                    'NUM_BLUE_UGV3',
                    'NUM_RED_UGV3',
                    'NUM_BLUE_UGV4',
                    'NUM_RED_UGV4',
                    ],
                'control': [
                    'CONTROL_ALL',
                    'MAX_STEP',
                    'RED_STEP',
                    'RED_DELAY',
                    ],
                'communication': [
                    'COM_GROUND',
                    'COM_AIR',
                    'COM_DISTANCE',
                    'COM_FREQUENCY'
                    ],
                'memory': [
                    'INDIV_MEMORY',
                    'TEAM_MEMORY',
                    'RENDER_INDIV_MEMORY',
                    'RENDER_TEAM_MEMORY'
                    ],
                'settings': [
                    'RL_SUGGESTIONS',
                    'STOCH_TRANSITIONS',
                    'STOCH_TRANSITIONS_EPS',
                    'STOCH_TRANSITIONS_MOD',
                    'STOCH_ATTACK',
                    'STOCH_ATTACK_BIAS',
                    'STOCH_ZONES',
                    'RED_PARTIAL',
                    'BLUE_PARTIAL',
                    'MAP_MODE',
                    'MAP_POOL_SIZE',
                    'AGENT_KILL',
                    'FINAL_KILL'
                    'BLUE_FLAG_VAL'
                    'RED_FLAG_VAL'
                    ]
            }
        config_datatype = {
                'elements': [int, int, int ,int, int,
                        int, int,
                        int, int,
                        int, int],
                'control': [bool, int, int, int, int, int],
                'communication': [bool, bool, int, float],
                'memory': [str, str, bool, bool],
                'settings': [bool, bool, float, str,
                        bool, int, bool, bool, bool, str, int, float, float]
            }

        if config_path is None and self.config_path is not None:
            return
        self.config_path = config_path
        config = configparser.ConfigParser()
        config.read(config_path)

        # Set environment attributes
        for section in config_param:
            for option, datatype in zip(config_param[section], config_datatype[section]):
                if not config.has_section(section) or not config.has_option(section, option):
                    if hasattr(self, option):
                        continue
                    else:
                        raise KeyError('Configuration import fails: double check whether all config variables are included')
                if datatype is bool:
                    value = config.getboolean(section, option)
                elif datatype is int:
                    value = config.getint(section, option)
                elif datatype is float:
                    value = config.getfloat(section, option)
                elif datatype is str:
                    value = config.get(section, option)
                else:
                    raise Exception('Unsupported datatype')
                setattr(self, option, value)

    def reset(self, map_size=None, mode="random", policy_blue=None, policy_red=None,
            custom_board=None, config_path=None):
        """
        Resets the game

        Parameters
        ----------------

        map_size : [int]
        mode : [str]
        policy_blue : [policy]
        policy_red : [policy]
        custom_board : [str, numpy.ndarray]
        config_path : [str]

        """

        # WARNINGS

        # STORE ARGUMENTS
        self.mode = mode

        # LOAD DEFAULT PARAMETERS
        if config_path is not None:
            self._parse_config(config_path)
        if map_size is None:
            map_size = self.map_size
        elif type(map_size) is int:
            map_size = (map_size, map_size)
        elif type(map_size) is list and len(map_size) == 2:
            map_size = tuple(map_size)
        else:
            raise TypeError("Invalid type for map_size")

        # INITIALIZE MAP
        self.custom_board = custom_board
        if custom_board is None:
            # Random Generated Map
            map_obj = {
                    (TEAM1_UGV, TEAM2_UGV): (self.NUM_BLUE, self.NUM_RED),
                    (TEAM1_UAV, TEAM2_UAV): (self.NUM_BLUE_UAV, self.NUM_RED_UAV),
                    (TEAM1_UGV2, TEAM2_UGV2): (self.NUM_BLUE_UGV2, self.NUM_RED_UGV2),
                    (TEAM1_UGV3, TEAM2_UGV3): (self.NUM_BLUE_UGV3, self.NUM_RED_UGV3),
                    (TEAM1_UGV4, TEAM2_UGV4): (self.NUM_BLUE_UGV4, self.NUM_RED_UGV4)
                }

            self._env, self._static_map, agent_locs = gen_random_map('map',
                    map_size, rand_zones=self.STOCH_ZONES, np_random=self.np_random, map_obj=map_obj)
            self.map_size = map_size
        else:
            # Read map from existing file
            if type(custom_board) is str:
                board = np.loadtxt(custom_board, dtype = int, delimiter = " ")
            elif type(custom_board) is np.ndarray:
                board = custom_board
            else:
                raise AttributeError("Provided board must be either path(str) or matrix(np array).")
            self._env, self._static_map, map_obj, agent_locs = custom_map(board)
            self.NUM_BLUE, self.NUM_BLUE_UAV, self.NUM_BLUE_UGV2, self.NUM_BLUE_UGV3, self.NUM_BLUE_UGV4 = map_obj[TEAM1_BACKGROUND]
            self.NUM_RED, self.NUM_RED_UAV, self.NUM_RED_UGV2, self.NUM_RED_UGV3, self.NUM_RED_UGV4 = map_obj[TEAM2_BACKGROUND]
            self.map_size = tuple(self._static_map.shape)

        h, w = self.map_size
        Y, X = np.ogrid[:2*h, :2*w]
        self._radial = (X-w)**2 + (Y-h)**2

        # INITIALIZE TEAM
        self._team_blue, self._team_red = self._construct_agents(agent_locs, self._static_map)
        self._agents = self._team_blue+self._team_red

        self.action_space = spaces.Discrete(len(self.ACTION) ** len(self._team_blue))
        self.observation_space = Board(shape=[self.map_size[0], self.map_size[1], NUM_CHANNEL])

        # INITIATE POLICY
        if policy_blue is not None:
            self._policy_blue = policy_blue
        if self._policy_blue is not None:
            self._policy_blue.initiate(self._static_map, self._team_blue)
        if len(self._team_red) == 0:
            self.mode = "sandbox"
        else:
            if policy_red is not None:
                self._policy_red = policy_red
            if self._policy_red is not None:
                self._policy_red.initiate(self._static_map, self._team_red)

        # INITIALIZE MEMORY
        if self.TEAM_MEMORY == "fog":
            self.blue_memory = np.ones_like(self._static_map, dtype=bool)
            self.red_memory = np.ones_like(self._static_map, dtype=bool)

        if self.INDIV_MEMORY == "fog":
            for agent in self._team_blue + self._team_red:
                agent.memory[:] = UNKNOWN
                agent.memory_mode = "fog"

        # INITIALIZE TRAJECTORY (DEBUG)
        self._blue_trajectory = []
        self._red_trajectory = []

        self._create_observation_mask()

        self.blue_win = False
        self.red_win = False
        self.red_flag_captured = False
        self.blue_flag_captured = False
        self.red_eliminated = False
        self.blue_eliminated = False


        # Necessary for human mode
        self.first = True
        self.run_step = 0  # Number of step of current episode

        return self.get_obs_blue

    def _construct_agents(self, agent_coords, static_map):
        """
        From given coordinates, it generates objects of agents and make them into the list.

        team_blue --> [air1, air2, ... , ground1, ground2, ...]
        team_red  --> [air1, air2, ... , ground1, ground2, ...]

        complete_map    : 2d numpy array
        static_map      : 2d numpy array

        """
        team_blue = []
        team_red = []

        Class = {
            TEAM1_UAV : (AerialVehicle, TEAM1_BACKGROUND),
            TEAM2_UAV : (AerialVehicle, TEAM2_BACKGROUND),
            TEAM1_UGV : (GroundVehicle, TEAM1_BACKGROUND),
            TEAM2_UGV : (GroundVehicle, TEAM2_BACKGROUND),
            TEAM1_UGV2: (GroundVehicle_Tank, TEAM1_BACKGROUND),
            TEAM2_UGV2: (GroundVehicle_Tank, TEAM2_BACKGROUND),
            TEAM1_UGV3: (GroundVehicle_Scout, TEAM1_BACKGROUND),
            TEAM2_UGV3: (GroundVehicle_Scout, TEAM2_BACKGROUND),
            TEAM1_UGV4: (GroundVehicle_Clocking, TEAM1_BACKGROUND),
            TEAM2_UGV4: (GroundVehicle_Clocking, TEAM2_BACKGROUND),
         }

        for element, coords in agent_coords.items():
            if coords is None: continue
            for coord in coords:
                Vehicle, team_id = Class[element]
                cur_ent = Vehicle(coord, static_map, team_id, element)
                if team_id == TEAM1_BACKGROUND:
                    team_blue.append(cur_ent)
                elif team_id == TEAM2_BACKGROUND:
                    team_red.append(cur_ent)

        return team_blue, team_red

    def _create_vision_mask(self, centers, radii):
        h, w = self._static_map.shape
        mask = np.zeros([h,w], dtype=bool)
        for center, radius in zip(centers, radii):
            y,x = center
            mask += (self._radial[h-y:2*h-y, w-x:2*w-x]) <= radius ** 2
        return ~mask

    def _create_observation_mask(self):
        """
        Creates the mask

        Mask is True(1) for the location where it CANNOT see.
        For full observation setting, mask is zero matrix

        Parameters
        ----------
        self    : object
            CapEnv object
        team    : int
            Team to create obs space for
        """


        if self.BLUE_PARTIAL:
            centers, radii = [], []
            for agent in self._team_blue:
                if not agent.isAlive: continue
                centers.append(agent.get_loc())
                radii.append(agent.range)
            self._blue_mask = self._create_vision_mask(centers, radii)
            if self.TEAM_MEMORY == "fog":
                self.blue_memory = np.logical_and(self.blue_memory, self._blue_mask)
        else:
            self._blue_mask = np.zeros_like(self._static_map, dtype=bool)

        if self.RED_PARTIAL:
            centers, radii = [], []
            for agent in self._team_red:
                if not agent.isAlive: continue
                centers.append(agent.get_loc())
                radii.append(agent.range)
            self._red_mask = self._create_vision_mask(centers, radii)
            if self.TEAM_MEMORY == "fog":
                self.red_memory = np.logical_and(self.red_memory, self._red_mask)
        else:
            self._red_mask = np.zeros_like(self._static_map, dtype=bool)

    def step(self, entities_action=None, cur_suggestions=None):
        """
        Takes one step in the capture the flag game

        :param
            entities_action: contains actions for entity 1-n
            cur_suggestions: suggestions from rl to human
        :return:
            state    : object
            CapEnv object
            reward  : float
            float containing the reward for the given action
            isDone  : bool
            decides if the game is over
            info    :
        """

        self.run_step += 1
        indiv_action_space = len(self.ACTION)
        blue_point = 0.0
        red_point = 0.0

        if self.CONTROL_ALL:
            assert self.RED_STEP == 1
            assert entities_action is not None, 'Under CONTROL_ALL setting, action must be specified'
            assert (type(entities_action) is list) or (type(entities_action) is np.ndarray), \
                    'CONTROLL_ALL setting requires list (or numpy array) type of action'
            assert len(entities_action) == len(self._team_blue+self._team_red), \
                    'You entered wrong number of moves.'

            move_list_blue = entities_action[:len(self._team_blue)]
            move_list_red  = entities_action[-len(self._team_red):]

            # Move team1
            positions = []
            for idx, act in enumerate(move_list_blue):
                if self.STOCH_TRANSITIONS and self.np_random.random() < self.STOCH_TRANSITIONS_EPS:
                    if self._policy_blue is not None and not self._policy_blue._random_transition_safe:
                        act = 0
                    else:
                        act = self._stoch_transition(self._team_blue[idx].get_loc())
                self._team_blue[idx].move(self.ACTION[act], self._env, self._static_map)
                positions.append((self._team_blue[idx].get_loc(), self._team_blue[idx].isAlive))
            self._blue_trajectory.append(positions)


            # Move team2
            if self.mode == "sandbox":
                move_list_red = []
            positions = []
            for idx, act in enumerate(move_list_red):
                if self.STOCH_TRANSITIONS and self.np_random.random() < self.STOCH_TRANSITIONS_EPS:
                    if self._policy_red is not None and not self._policy_red._random_transition_safe:
                        act = 0
                    else:
                        act = self._stoch_transition(self._team_red[idx].get_loc())
                self._team_red[idx].move(self.ACTION[act], self._env, self._static_map)
                positions.append((self._team_red[idx].get_loc(), self._team_red[idx].isAlive))
            self._red_trajectory.append(positions)

        else:
            # Move team1
            if entities_action is None:
                # Use predefined policy
                try:
                    move_list_blue = self._policy_blue.gen_action(self._team_blue, self.get_obs_blue)
                except Exception as e:
                    print("No valid policy for blue team and no actions provided", e)
                    traceback.print_exc()
                    exit()
            elif type(entities_action) is int:
                # Action given in Integer
                move_list_blue = []
                if entities_action >= len(self.ACTION) ** len(self._team_blue):
                    sys.exit("ERROR: You entered too many moves. There are " + str(len(self._team_blue)) + " entities.")
                while len(move_list_blue) < len(self._team_blue):
                    move_list_blue.append(entities_action % indiv_action_space)
                    entities_action = int(entities_action / indiv_action_space)
            else:
                # Action given in array
                if len(entities_action) != len(self._team_blue):
                    sys.exit("ERROR: You entered wrong number of moves. There are " + str(len(self._team_blue)) + " entities.")
                move_list_blue = entities_action

            positions = []
            for idx, act in enumerate(move_list_blue):
                if self.STOCH_TRANSITIONS and self.np_random.random() < self.STOCH_TRANSITIONS_EPS:
                    if self._policy_blue is not None and not self._policy_blue._random_transition_safe:
                        act = 0
                    else:
                        act = self._stoch_transition(self._team_blue[idx].get_loc())
                self._team_blue[idx].move(self.ACTION[act], self._env, self._static_map)
                positions.append((self._team_blue[idx].get_loc(), self._team_blue[idx].isAlive))
            self._blue_trajectory.append(positions)

            # Move team2
            if self.mode != "sandbox" and self.run_step % self.RED_DELAY == 0:
                for _ in range(self.RED_STEP):
                    try:
                        move_list_red = self._policy_red.gen_action(self._team_red, self.get_obs_red)
                    except Exception as e:
                        print("No valid policy for red team", e)
                        traceback.print_exc()
                        exit()

                    positions = []
                    for idx, act in enumerate(move_list_red):
                        if self.STOCH_TRANSITIONS and self.np_random.random() < self.STOCH_TRANSITIONS_EPS:
                            if self._policy_red is not None and not self._policy_red._random_transition_safe:
                                act = 0
                            else:
                                act = self._stoch_transition(self._team_red[idx].get_loc())
                        self._team_red[idx].move(self.ACTION[act], self._env, self._static_map)
                        positions.append((self._team_red[idx].get_loc(), self._team_red[idx].isAlive))
                    self._red_trajectory.append(positions)

                    finish_move=False
                    for i in self._team_red:
                        if i.isAlive and not i.is_air:
                            locx, locy = i.get_loc()
                            if self._static_map[locx][locy] == TEAM1_FLAG:
                                finish_move=True
                    if finish_move: break

        self._create_observation_mask()

        # Update individual's memory
        for agent in self._agents:
            if agent.memory_mode == "fog":
                agent.update_memory(env=self)

        # Run interaction
        target_agents = [agent for agent in self._agents if agent.isAlive and not agent.is_air]
        survive_list = [agent.isAlive for agent in target_agents]
        new_status = self._interaction(target_agents)
        num_blue_killed = 0
        num_red_killed = 0
        for idx, entity in enumerate(target_agents):
            if survive_list[idx] and not new_status[idx]:
                if entity.team == TEAM1_BACKGROUND:
                    num_blue_killed += 1
                    red_point+= self.AGENT_KILL
                elif entity.team == TEAM2_BACKGROUND:
                    num_red_killed += 1
                    blue_point+= self.AGENT_KILL
        for status, entity in zip(new_status, target_agents):
            entity.isAlive = status

        # Check win and lose conditions
        has_alive_entity = False
        for i in self._team_red:
            if i.isAlive and not i.is_air:
                has_alive_entity = True
                locx, locy = i.get_loc()
                if self._static_map[locx][locy] == TEAM1_FLAG:  # TEAM 1 == BLUE
                    self.red_win = True
                    self.blue_flag_captured = True
                    red_point += self.RED_FLAG_VAL
                    if self.mode == 'continue': # Regenerate
                        self._static_map[locx][locy] = TEAM1_BACKGROUND
                        self._env[locx][locy][2] = 0
                        candidate = np.logical_and(self._env[:,:,1]==REPRESENT[TEAM1_BACKGROUND], self._env[:,:,4]!=REPRESENT[TEAM1_UGV])
                        coords = np.argwhere(candidate)
                        newloc = coords[np.random.choice(len(coords))]
                        self._static_map[newloc[0]][newloc[1]] = TEAM1_FLAG
                        self._env[newloc[0]][newloc[1]][2] = REPRESENT[TEAM1_FLAG]

        # TODO Change last condition for multi agent model
        if not has_alive_entity and self.mode != "sandbox" and self.mode != "human_blue":
            # self.blue_win = True
            if not self.red_eliminated:
                blue_point += self.FINAL_KILL
            self.red_eliminated = True

        has_alive_entity = False
        for i in self._team_blue:
            if i.isAlive and not i.is_air:
                has_alive_entity = True
                locx, locy = i.get_loc()
                if self._static_map[locx][locy] == TEAM2_FLAG:
                    self.blue_win = True
                    self.red_flag_captured = True
                    blue_point += self.BLUE_FLAG_VAL
                    if self.mode == 'continue': # Regenerate
                        self._static_map[locx][locy] = TEAM2_BACKGROUND
                        self._env[locx][locy][2] = 0
                        candidate = np.logical_and(self._env[:,:,1]==REPRESENT[TEAM2_BACKGROUND], self._env[:,:,4]!=REPRESENT[TEAM2_UGV])
                        coords = np.argwhere(candidate)
                        newloc = coords[np.random.choice(len(coords))]
                        self._static_map[newloc[0]][newloc[1]] = TEAM2_FLAG
                        self._env[newloc[0]][newloc[1]][2] = REPRESENT[TEAM2_FLAG]


        if not has_alive_entity:
            if not self.blue_eliminated:
                red_point += self.FINAL_KILL
            self.blue_eliminated = True
        isDone = self.red_win or self.blue_win or self.run_step > self.MAX_STEP or (self.blue_eliminated and self.red_eliminated)
        # if self.run_step > self.MAX_STEP:
            # if blue_point > red_point:
            #     self.blue_win = True
            # elif blue_point < red_point:
            #     self.red_win = True

        # Calculate Reward
        #reward, red_reward = self._create_reward(num_blue_killed, num_red_killed, mode='instant')
        reward, red_reward = blue_point-red_point-0.001, red_point-blue_point-0.001

        # Pass internal info
        info = {
                'blue_trajectory': self._blue_trajectory,
                'red_trajectory': self._red_trajectory,
                'static_map': self._static_map,
                'red_reward': red_reward
            }

        return self.get_obs_blue, reward, isDone, info
    def update_step(self,entity_location,interactions=True):
        """
        Takes an update step in the ctf game where agent locations are updated based on external simulation.
        :param
            entities_location:
        :return:
            state    : object
            CapEnv object
            reward  : float
            float containing the reward for the given action
            isDone  : bool
            decides if the game is over
            info    :
        """
        move_list_blue = []
        move_list_red = []
        if self.run_step == 0:
            agent_list_blue =[]
            agent_list_red = []
            #Creating movelist from the input dictionary
            for name, loc in entity_location.items():
                if "T1" in name:
                    if "UAV" in name:
                        agent_list_blue.insert(0,name)
                    elif "R" in name:
                        agent_list_blue.append(name)
                elif "T2" in name:
                    if "UAV" in name:
                        agent_list_red.insert(0,name)
                    elif "R" in name:
                        agent_list_red.append(name)
            self.agent_list = agent_list_blue + agent_list_red

        move_list  = []
        for name in self.agent_list:
            move_list.append(entity_location[name])
        move_list_blue = move_list[:len(self._team_blue)]
        move_list_red  = move_list[-len(self._team_red):]


        #Update all agent locations.
        for idx, loc in enumerate(move_list_blue):
            self._team_blue[idx].move_abs(loc, self._env, self._static_map)
        for idx, loc in enumerate(move_list_red):
            self._team_red[idx].move_abs(loc, self._env, self._static_map)

        self._create_observation_mask()

        # Update individual's memory
        for agent in self._team_blue + self._team_red:
            if agent.memory_mode == "fog":
                agent.update_memory(env=self)

        # Update team memory
        if self.TEAM_MEMORY == "fog":
            self._update_global_memory(env=self)

        # Run interaction
        if interactions:
            survive_list = []
            for entity in self._team_blue + self._team_red:
                if not entity.isAlive:
                    survive_list.append(False)
                else:
                    survive_list.append(self._interaction(entity))
            for status, entity in zip(survive_list, self._team_blue+self._team_red):
                entity.isAlive = status

            # Check win and lose conditions
            has_alive_entity = False
            for i in self._team_red:
                if i.isAlive and not i.is_air:
                    has_alive_entity = True
                    locx, locy = i.get_loc()
                    if self._static_map[locx][locy] == TEAM1_FLAG:  # TEAM 1 == BLUE
                        self.red_win = True
                        self.blue_flag_captured = True

            # TODO Change last condition for multi agent model
            if not has_alive_entity and self.mode != "sandbox" and self.mode != "human_blue":
                self.blue_win = True
                self.red_eliminated = True

            has_alive_entity = False
            for i in self._team_blue:
                if i.isAlive and not i.is_air:
                    has_alive_entity = True
                    locx, locy = i.get_loc()
                    if self._static_map[locx][locy] == TEAM2_FLAG:
                        self.blue_win = True
                        self.red_flag_captured = True

            if not has_alive_entity:
                self.red_win = True
                self.blue_eliminated = True

        # Calculate Reward
        reward = 0 #self._create_reward()

        isDone = self.red_win or self.blue_win

        # Pass internal info
        info = {
                # 'blue_trajectory': self._blue_trajectory,
                # 'red_trajectory': self._red_trajectory,
                'static_map': self._static_map
            }

        self.run_step += 1

        alive={}
        for idx,i in enumerate(self._team_blue):
            alive[self.agent_list[idx]] = i.isAlive
        for idx,i in enumerate(self._team_red):
            alive[self.agent_list[idx+len(self._team_blue)]] = i.isAlive

        return self.get_obs_blue, self.get_obs_red, reward, isDone, alive, info

    def _stoch_transition(self, loc):
        if self.STOCH_TRANSITIONS_MOD == 'random':
            return self.np_random.randint(0,len(self.ACTION))
        elif self.STOCH_TRANSITIONS_MOD == 'fix':
            return 0
        elif self.STOCH_TRANSITIONS_MOD == 'drift1':
            if loc[0] > self.map_size[0]//2 or loc[1] > self.map_size[1]//2:
                return self.np_random.choice([1,2])
            else:
                return self.np_random.choice([3,4])
        else:
            raise AttributeError('Unknown transition mod is used: {}'.format(self.STOCH_TRANSITIONS_MOD))

    def _interaction(self, entity):
        """
        Interaction

        Checks if a unit is dead
        If configuration parameter 'STOCH_ATTACK' is true, the interaction becomes stochastic

        Parameters
        ----------
        entity       : list
            List of all agents

        Return
        ______
        list[bool] :
            Return true if the entity survived after the interaction
        """

        # Get parameters
        att_range = np.array([agent.a_range for agent in entity], dtype=float)[:,None]
        att_strength = np.array([agent.get_advantage for agent in entity])[:,None]
        team_index = np.array([agent.team for agent in entity])
        alliance_matrix = team_index[:,None]==team_index[None,:]

        # Get distance between all agents
        x, y = np.array([agent.get_loc() for agent in entity]).T
        dx = np.subtract(*np.meshgrid(x,x))
        dy = np.subtract(*np.meshgrid(y,y))
        distance = np.hypot(dx, dy)

        # Get influence matrix
        infl_matrix = np.less(distance, att_range)
        infl_matrix = infl_matrix * att_strength
        friend_count = (infl_matrix*alliance_matrix).sum(axis=0)-1 # -1 to eliminate self
        enemy_count = (infl_matrix*~alliance_matrix).sum(axis=0)
        mask = enemy_count == 0

        # Add background advantage bias
        loc_background = [self._static_map[agent.get_loc()] for agent in entity]
        friend_count[loc_background==team_index] += self.STOCH_ATTACK_BIAS
        enemy_count[~(loc_background==team_index)] += self.STOCH_ATTACK_BIAS

        # Interaction
        if self.STOCH_ATTACK:
            result = self.np_random.random(friend_count.shape) < friend_count / (friend_count + enemy_count)
        else:
            result = friend_count > enemy_count
        result[mask] = True

        return result

    def _create_reward(self, num_blue_killed, num_red_killed, mode='dense'):
        """
        Range (-100, 100)

        Parameters
        ----------
        self    : object
            CapEnv object
        """

        assert mode in ['dense', 'flag', 'combat', 'defense', 'capture', 'instant']

        red_alive = sum([entity.isAlive for entity in self._team_red if not entity.is_air])
        blue_alive = sum([entity.isAlive for entity in self._team_blue if not entity.is_air])
        red_total = len([entity for entity in self._team_red if not entity.is_air])
        blue_total = len([entity for entity in self._team_blue if not entity.is_air])

        if mode == 'dense':
            # Dead enemy team gives .5/total units for each dead unit
            # Only count ground unit
            if self.red_win:
                return -100
            if self.blue_win:
                return 100
            reward = 0
            if self.mode != 'sandbox':
                reward += 50.0 * (red_total - red_alive) / red_total
            reward -= (50.0 * (blue_total - blue_alive) / blue_total)
            return reward
        elif mode == 'flag':
            # Flag game reward
            if self.red_flag_captured:
                return 100
            if self.blue_flag_captured:
                return -100
        elif mode == 'combat':
            # Aggressive combat game. Elliminate enemy to win
            return 100 * red_alive / red_total
        elif mode == 'defense':
            # Lose reward if flag is lost.
            if self.blue_flag_captured:
                return -100
        elif mode == 'capture':
            # Reward only by capturing (sparse)
            if self.red_flag_captured:
                return 100
        elif mode == 'instant':
            bias = 0

            # DRAW
            if self.red_flag_captured and self.blue_flag_captured:
                return -1, -1
            elif self.red_win and self.blue_win:
                return -1, -1
            elif self.run_step > self.MAX_STEP:
                return -1, -1

            # TERMINATE
            if self.red_win:
                return -1, 1
            elif self.blue_win:
                return 1, -1

            # INTERMEDIATE
            diff = num_red_killed - num_blue_killed
            lambd = 0.1
            blue_reward = bias + diff * lambd
            red_reward = bias + (-diff) * lambd

            return blue_reward, red_reward

    def render(self, mode='human'):
        """
        Renders the screen options="obs, env"

        Parameters
        ----------
        self    : object
            CapEnv object
        mode    : string
            Defines what will be rendered
        """

        if (self.RENDER_INDIV_MEMORY == True and self.INDIV_MEMORY == "fog") or (self.RENDER_TEAM_MEMORY == True and self.TEAM_MEMORY == "fog"):
            SCREEN_W = 1200
            SCREEN_H = 600

            if self.viewer is None:
                import pygame
                from pygame import gfxdraw
                if mode=="human":
                    self.viewer = pygame.display.set_mode(
                        (SCREEN_W, SCREEN_H)
                    )
                else:
                    self.viewer = pygame.Surface(
                    (SCREEN_W, SCREEN_H)
                    )
                # self.viewer.set_bounds(0, SCREEN_W, 0, SCREEN_H)

            self.viewer.draw_polygon([(0, 0), (SCREEN_W, 0), (SCREEN_W, SCREEN_H), (0, SCREEN_H)], color=(0, 0, 0))

            self._env_render(self._static_map,
                            [7, 7], [SCREEN_H//2-10, SCREEN_H//2-10])
            self._env_render(self.get_obs_blue_render,
                            [7+1.49*SCREEN_H//3, 7], [SCREEN_H//2-10, SCREEN_H//2-10])
            self._env_render(self.get_obs_red_render,
                            [7+1.49*SCREEN_H//3, 7+1.49*SCREEN_H//3], [SCREEN_H//2-10, SCREEN_H//2-10])
            self._env_render(self.get_full_state,
                            [7, 7+1.49*SCREEN_H//3], [SCREEN_H//2-10, SCREEN_H//2-10])

            # ind blue agent memory rendering
            for num_blue, blue_agent in enumerate(self._team_blue):
                if num_blue < 2:
                    blue_agent.INDIV_MEMORY = self.INDIV_MEMORY
                    if blue_agent.INDIV_MEMORY == "fog" and self.RENDER_INDIV_MEMORY == True:
                        self._env_render(blue_agent.memory,
                                         [900+num_blue*SCREEN_H//4, 7], [SCREEN_H//4-10, SCREEN_H//4-10])
                else:
                    blue_agent.INDIV_MEMORY = self.INDIV_MEMORY
                    if blue_agent.INDIV_MEMORY == "fog" and self.RENDER_INDIV_MEMORY == True:
                        self._env_render(blue_agent.memory,
                                         [900+(num_blue-2)*SCREEN_H//4, 7+SCREEN_H//4], [SCREEN_H//4-10, SCREEN_H//4-10])

            # ind red agent memory rendering
            for num_red, red_agent in enumerate(self._team_red):
                if num_red < 2:
                    red_agent.INDIV_MEMORY = self.INDIV_MEMORY
                    if red_agent.INDIV_MEMORY == "fog" and self.RENDER_INDIV_MEMORY == True:
                        self._env_render(red_agent.memory,
                                         [900+num_red*SCREEN_H//4, 7+1.49*SCREEN_H//2], [SCREEN_H//4-10, SCREEN_H//4-10])

                else:
                    red_agent.INDIV_MEMORY = self.INDIV_MEMORY
                    if red_agent.INDIV_MEMORY == "fog" and self.RENDER_INDIV_MEMORY == True:
                        self._env_render(red_agent.memory,
                                         [900+(num_red-2)*SCREEN_H//4, 7+SCREEN_H//2], [SCREEN_H//4-10, SCREEN_H//4-10])

            if self.TEAM_MEMORY == "fog" and self.RENDER_TEAM_MEMORY == True:
                # blue team memory rendering
                blue_visited = np.copy(self._static_map)
                blue_visited[self.blue_memory] = UNKNOWN
                self._env_render(blue_visited,
                                 [7+2.98*SCREEN_H//3, 7], [SCREEN_H//2-10, SCREEN_H//2-10])

                # red team memory rendering
                red_visited = np.copy(self._static_map)
                red_visited[self.red_memory] = UNKNOWN
                self._env_render(red_visited,
                                 [7+2.98*SCREEN_H//3, 7+1.49*SCREEN_H//3], [SCREEN_H//2-10, SCREEN_H//2-10])
        else:
            SCREEN_W = 600
            SCREEN_H = 600

            import pygame
            from pygame import gfxdraw
            if self.viewer is None:
                if mode=="human":
                    self.viewer = pygame.display.set_mode(
                        (SCREEN_W, SCREEN_H)
                    )
                else:
                    self.viewer = pygame.Surface(
                    (SCREEN_W, SCREEN_H)
                    )
                # self.viewer.set_bounds(0, SCREEN_W, 0, SCREEN_H)
            self.surf = pygame.Surface((SCREEN_W, SCREEN_H))
            self.surf.fill((255, 255, 255))

            pygame.display.set_caption("Capture the Flag")

            gfxdraw.filled_polygon(self.surf, [(0, 0), (SCREEN_W, 0), (SCREEN_W, SCREEN_H), (0, SCREEN_H)],  (0,0,0))


            self._env_render(self._static_map,
                            [5, 10], [SCREEN_W//2-10, SCREEN_H//2-10])
            self._env_render(self.get_obs_blue_render,
                            [5+SCREEN_W//2, 10], [SCREEN_W//2-10, SCREEN_H//2-10])
            self._agent_render(self.get_full_state,
                            [5+SCREEN_W//2, 10], [SCREEN_W//2-10, SCREEN_H//2-10], self._team_blue)
            self._env_render(self.get_obs_red_render,
                            [5+SCREEN_W//2, 10+SCREEN_H//2], [SCREEN_W//2-10, SCREEN_H//2-10])
            self._env_render(self.get_full_state,
                            [5, 10+SCREEN_H//2], [SCREEN_W//2-10, SCREEN_H//2-10])
            self._agent_render(self.get_full_state,
                            [5, 10+SCREEN_H//2], [SCREEN_W//2-10, SCREEN_H//2-10])


            if mode=="human":
                self.viewer.blit(self.surf, (0, 0))
                pygame.event.pump()
                pygame.display.flip()

            return np.array(pygame.surfarray.pixels3d(self.viewer))

    def _env_render(self, image, rend_loc, rend_size):
        import pygame
        from pygame import gfxdraw
        map_h, map_w = image.shape
        tile = int(min(rend_size[0]/map_w, rend_size[1]/map_h))

        for y in range(map_h):
            for x in range(map_w):
                locx, locy = rend_loc
                locx += x * tile
                locy += y * tile
                cur_color = COLOR_DICT[image[y][x]]
                gfxdraw.filled_polygon(self.surf,[
                    (locx, locy),
                    (locx + tile, locy),
                    (locx + tile, locy + tile),
                    (locx, locy + tile)], cur_color)
                gfxdraw.aapolygon(self.surf,[
                    (locx, locy),
                    (locx + tile, locy),
                    (locx + tile, locy + tile),
                    (locx, locy + tile)], cur_color)

                if image[y][x] == TEAM1_UAV or image[y][x] == TEAM2_UAV:
                    gfxdraw.line(self.surf,
                        locx, locy,
                        locx + tile, locy + tile,
                        (0,0,0))
                    gfxdraw.line(self.surf,
                        locx + tile, locy,
                        locx, locy + tile,
                        (0,0,0))#col * tile, row * tile

    def _agent_render(self, image, rend_loc, rend_size, agents=None):
        import pygame
        from pygame import gfxdraw
        if agents is None:
            agents = self._team_blue + self._team_red
        map_h, map_w = image.shape
        tile = int(min(rend_size[0]/map_w, rend_size[1]/map_h))

        for entity in agents:
            if not entity.isAlive: continue
            y,x = entity.get_loc()
            locx, locy = rend_loc
            locx += x * tile
            locy += y * tile
            cur_color = COLOR_DICT[entity.unit_type]
            gfxdraw.filled_polygon(self.surf,[
                (locx, locy),
                (locx + tile, locy),
                (locx + tile, locy + tile),
                (locx, locy + tile)], cur_color)

            if type(entity) in [AerialVehicle,GroundVehicle_Tank,GroundVehicle_Scout,GroundVehicle_Clocking]:
                gfxdraw.line(self.surf,
                    locx, locy,
                    locx + tile, locy + tile,
                    (0,0,0))
                gfxdraw.line(self.surf,
                    locx + tile, locy,
                    locx, locy + tile,
                    (0,0,0))#col * tile, row * tile


            if entity.marker is not None:
                ratio = 0.6
                color = np.divide(entity.marker, 255.0)
                gfxdraw.filled_polygon(self.surf,[
                    (locx + tile * ratio, locy + tile * ratio),
                    (locx + tile, locy + tile * ratio),
                    (locx + tile, locy + tile),
                    (locx + tile * ratio, locy + tile)], color=color)

    def close(self):
        if self.viewer: self.viewer.close()

    def _env_flat(self, mask=None):
        # Return 2D representation of the state
        board = np.copy(self._static_map)
        if mask is not None:
            board[mask] = UNKNOWN
        for entity in self._team_blue+self._team_red:
            if not entity.isAlive: continue
            loc = entity.get_loc()
            if mask is not None and mask[loc]: continue
            board[loc] = entity.unit_type
        return board

    @property
    def get_full_state(self, mask=None):
        return self._env_flat()

    @property
    def get_full_state_channel(self):
        return np.copy(self._env)

    @property
    def get_full_state_rgb(self):
        w, h, ch = self._env.shape
        image = np.full(shape=[w, h, 3], fill_value=0, dtype=int)
        for element in CHANNEL.keys():
            if element == FOG:
                continue
            channel = CHANNEL[element]
            rep = REPRESENT[element]
            image[self._env[:,:,channel]==rep] = np.array(COLOR_DICT[element])
        return image

    @property
    def get_team_blue(self):
        return np.copy(self._team_blue)

    @property
    def get_team_red(self):
        return np.copy(self._team_red)

    @property
    def get_team_grey(self):
        return np.copy(self.team_grey)

    @property
    def get_map(self):
        return np.copy(self._static_map)

    @property
    def get_obs_blue(self):
        view = np.copy(self._env)

        if self.BLUE_PARTIAL:
            if self.TEAM_MEMORY == 'fog':
                memory_channel = np.array([CHANNEL[OBSTACLE], CHANNEL[TEAM1_BACKGROUND], CHANNEL[UNKNOWN]])
                immediate_channel = np.array([CHANNEL[TEAM1_UGV], CHANNEL[TEAM1_UAV], CHANNEL[TEAM1_FLAG]])

                mask_represent = REPRESENT[UNKNOWN]
                mask_channel = CHANNEL[UNKNOWN]
                fog_represent = REPRESENT[FOG]
                fog_channel = CHANNEL[FOG]

                for ch in memory_channel:
                    view[self.blue_memory, ch] = 0
                for ch in immediate_channel:
                    view[self._blue_mask, ch] = 0
                view[self._blue_mask, mask_channel] = mask_represent
                view[self.blue_memory, fog_channel] = fog_represent
            else:
                mask_channel = CHANNEL[UNKNOWN]
                mask_represent = REPRESENT[UNKNOWN]

                view[self._blue_mask, :] = 0
                view[self._blue_mask, mask_channel] = mask_represent

        for entity in self._team_red:
            if not entity.is_visible:
                x, y = entity.get_loc()
                view[x, y, entity.channel] = 0

        return view

    @property
    def get_obs_red(self):
        view = np.copy(self._env)

        if self.RED_PARTIAL:
            if self.TEAM_MEMORY == 'fog':
                memory_channel = np.array([CHANNEL[OBSTACLE], CHANNEL[TEAM1_BACKGROUND], CHANNEL[UNKNOWN]])
                immediate_channel = np.array([CHANNEL[TEAM1_UGV], CHANNEL[TEAM1_UAV], CHANNEL[TEAM1_FLAG]])

                mask_represent = REPRESENT[UNKNOWN]
                mask_channel = CHANNEL[UNKNOWN]
                fog_represent = REPRESENT[FOG]
                fog_channel = CHANNEL[FOG]

                for ch in memory_channel:
                    view[self.red_memory, ch] = 0
                for ch in immediate_channel:
                    view[self._red_mask, ch] = 0
                view[self._red_mask, mask_channel] = mask_represent
                view[self.red_memory, fog_channel] = fog_represent
            else:
                mask_channel = CHANNEL[UNKNOWN]
                mask_represent = REPRESENT[UNKNOWN]

                view[self._red_mask, :] = 0
                view[self._red_mask, mask_channel] = mask_represent

        for entity in self._team_blue:
            if entity.is_visible:
                x, y = entity.get_loc()
                view[x, y, entity.channel] = 0

        # Change red's perspective same as blue
        swap = set([CHANNEL[TEAM1_BACKGROUND], CHANNEL[TEAM1_UGV], CHANNEL[TEAM1_UAV], CHANNEL[TEAM1_FLAG],
                CHANNEL[TEAM1_UGV2], CHANNEL[TEAM1_UGV3], CHANNEL[TEAM1_UGV4]])

        for ch in swap:
            view[:,:,ch] *= -1

        return view

    @property
    def get_obs_blue_render(self):
        return self._env_flat(self._blue_mask)

    @property
    def get_obs_red_render(self):
        return self._env_flat(self._red_mask)

    @property
    def get_obs_grey(self):
        return np.copy(self.observation_space_grey)


    # def quit_game(self):
    #     if self.viewer is not None:
    #         self.viewer.close()
    #         self.viewer = None


# Different environment sizes and modes
# Random modes
class CapEnvGenerate(CapEnv):
    def __init__(self):
        super(CapEnvGenerate, self).__init__(map_size=20)


# State space for capture the flag
class Board(spaces.Space):
    """A Board in R^3 used for CtF """
    def __init__(self, shape=None, dtype=np.uint8):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)

        if shape is None:
            self._shape = (20, 20, NUM_CHANNEL)
        else:
            assert shape[2] == NUM_CHANNEL
            self._shape = tuple(shape)
        super(Board, self).__init__(self.shape, self.dtype)

    def __repr__(self):
        return "Board" + str(self._shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape

    def sample(self):
        map_obj = [NUM_BLUE, NUM_BLUE_UAV, NUM_RED, NUM_RED_UAV, NUM_GRAY]
        state, _, _ = gen_random_map('map',
                self.shape[0], rand_zones=False, map_obj=map_obj)
        return state
