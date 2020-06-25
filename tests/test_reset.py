import gym
import gym_cap

from gym_cap.envs.cap_env import *
import gym_cap.heuristic as policy

from pytest import raises
from random import randint
import numpy as np
from numpy import testing

ENV_NAME = 'cap-v0'
env = gym.make(ENV_NAME)

class TestReset:
    """
    Tests the reset function in gym_cap.envs.cap_env using pytest

    """

    # testing map_size

    def test_map_size_none(self):
        " Tests that the default map_size is 20 x 20 "
        map_size = None
        env.reset(map_size)
        assert env.map_size == (20, 20)

    def test_map_size_int(self):
        " Tests that appropriate integer input for map_size variable is successful "
        min_size = 5 # arbitrary value
        max_size = 20 # arbitrary value
        for int in range(min_size, max_size):
            map_size = int
            env.reset(map_size)
            assert env.map_size == (map_size, map_size)

    def test_map_size_list_len_2(self):
        " Tests that list input of length 2 for map_size variable is successful "
        min_size = 5
        max_size = 20
        for int in range(min_size, max_size):
            map_size = [int, int]
            env.reset(map_size)
            assert env.map_size == tuple(map_size)

    def test_map_size_list_non_square(self):
        " Tests that non-square map_size input allows reset function to be successful "
        map_size = [15, 20]
        env.reset(map_size)
        assert env.map_size == tuple(map_size)

    def test_map_size_list_too_long(self):
        " Tests that TypeError is raised if map_size input is a list with "
        " with a length greater than 2 "
        with raises(TypeError) as err:
            map_size = [20, 20, 20]
            env.reset(map_size)
            assert env.map_size == tuple(map_size)
        assert str(err.value) == 'Invalid type for map_size'

    def test_map_size_list_too_short(self):
        " Tests that TypeError is raised if map_size input is a list with "
        " a length less than 2 "
        with raises(TypeError) as err:
            map_size = [20]
            env.reset(map_size)
            assert env.map_size == tuple(map_size)
        assert str(err.value) == 'Invalid type for map_size'

    def test_map_size_negative_value(self):
        " Tests that ValueError is raised if map_size input contains negative values "
        with raises(ValueError):
            map_size = [-1, -1]
            env.reset(map_size)
            assert env.map_size == tuple(map_size)

    def test_map_size_too_small(self):
        " Tests that ValueError is raised if map_size input is less than 5 x 5 "
        with raises(ValueError):
            map_size = [4, 4]
            env.reset(map_size)
            assert env.map_size == tuple(map_size)

    def test_large_map_size(self):
        " Tests that large map_size value allows reset function to be successful "
        map_size = [1000, 1000]
        env.reset(map_size)
        assert env.map_size == tuple(map_size)

    def test_too_large_map_size(self):
        " Tests that MemoryError is raised when map_size value is too large "
        with raises(MemoryError):
            map_size = [10000000, 10000000]
            env.reset(map_size)
            assert env.map_size == tuple(map_size)

    # testing mode

    def test_mode_none_specified(self):
        " Tests that when no mode is specified, self.mode is set to 'random' "
        env = gym.make(ENV_NAME)
        env.reset()
        assert env.mode == 'random'

    # testing policy_blue

    def test_policy_blue_random(self):
        " Tests that random policy is appropriately assigned to the blue team "
        " for the reset function "
        random_policy = policy.Random()
        env.reset(policy_blue=random_policy)
        assert env._policy_blue == random_policy

    def test_policy_blue_none(self):
        " Tests that default value of None is assigned to self._policy_blue "
        " when the reset function is called and policy_blue is not specified "
        env = gym.make(ENV_NAME)
        env.reset()
        assert env._policy_blue == None

    def test_policy_blue_patrol(self):
        " Tests that patrol policy is appropriately assigned to the blue team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        patrol_policy = policy.Patrol()
        env.reset(policy_blue=patrol_policy)
        assert env._policy_blue == patrol_policy

    def test_policy_blue_astar(self):
        " Tests that astar policy is appropriately assigned to the blue team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        astar_policy = policy.AStar()
        env.reset(policy_blue=astar_policy)
        assert env._policy_blue == astar_policy

    def test_policy_blue_defense(self):
        " Tests that defense policy is appropriately assigned to the blue team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        defense_policy = policy.Defense()
        env.reset(policy_blue=defense_policy)
        assert env._policy_blue == defense_policy

    def test_policy_blue_fighter(self):
        " Tests that fighter policy is appropriately assigned to the blue team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        fighter_policy = policy.Fighter()
        env.reset(policy_blue=fighter_policy)
        assert env._policy_blue == fighter_policy

    def test_policy_blue_roomba(self):
        " Tests that roomba policy is appropriately assigned to the blue team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        roomba_policy = policy.Roomba()
        env.reset(policy_blue=roomba_policy)
        assert env._policy_blue == roomba_policy

    def test_policy_blue_spiral(self):
        " Tests that spiral policy is appropriately assigned to the blue team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        spiral_policy = policy.Spiral()
        env.reset(policy_blue=spiral_policy)
        assert env._policy_blue == spiral_policy

    def test_policy_blue_zeros(self):
        " Tests that zeros policy is appropriately assigned to the blue team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        zeros_policy = policy.Zeros()
        env.reset(policy_blue=zeros_policy)
        assert env._policy_blue == zeros_policy

    # testing policy_red

    def test_policy_red_random(self):
        " Tests that random policy is appropriately assigned to the red team "
        " for the reset function "
        random_policy = policy.Random()
        env.reset(policy_red=random_policy)
        assert env._policy_red == random_policy

    def test_policy_red_none(self):
        " Tests that default value of None is assigned to self._policy_red "
        " when the reset function is called and policy_red is not specified "
        env = gym.make(ENV_NAME)
        env.reset()
        assert env._policy_red == None

    def test_policy_red_patrol(self):
        " Tests that patrol policy is appropriately assigned to the red team "
        " for the reset function "
        patrol_policy = policy.Patrol()
        env.reset(policy_red=patrol_policy)
        assert env._policy_red == patrol_policy

    def test_policy_red_astar(self):
        " Tests that astar policy is appropriately assigned to the red team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        astar_policy = policy.AStar()
        env.reset(policy_red=astar_policy)
        assert env._policy_red == astar_policy

    def test_policy_red_defense(self):
        " Tests that defense policy is appropriately assigned to the red team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        defense_policy = policy.Defense()
        env.reset(policy_red=defense_policy)
        assert env._policy_red == defense_policy

    def test_policy_red_fighter(self):
        " Tests that fighter policy is appropriately assigned to the red team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        fighter_policy = policy.Fighter()
        env.reset(policy_red=fighter_policy)
        assert env._policy_red == fighter_policy

    def test_policy_red_roomba(self):
        " Tests that roomba policy is appropriately assigned to the red team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        roomba_policy = policy.Roomba()
        env.reset(policy_red=roomba_policy)
        assert env._policy_red == roomba_policy

    def test_policy_red_spiral(self):
        " Tests that spiral policy is appropriately assigned to the red team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        spiral_policy = policy.Spiral()
        env.reset(policy_red=spiral_policy)
        assert env._policy_red == spiral_policy

    def test_policy_red_zeros(self):
        " Tests that zeros policy is appropriately assigned to the red team "
        " for the reset function "
        env = gym.make(ENV_NAME)
        zeros_policy = policy.Zeros()
        env.reset(policy_red=zeros_policy)
        assert env._policy_red == zeros_policy

    # testing custom_board

    def test_custom_board_none(self):
        " Tests that when custom_board is not specified, self.custom_board is None "
        env.reset()
        assert env.custom_board == None

    def test_custom_board_txt_file(self):
        " Tests that board1 as a string input successfully works as a custom_board "
        env.reset(custom_board='board1.txt')
        custom_board = 'board1.txt'
        board = np.loadtxt(custom_board, dtype = int, delimiter = " ")
        board_1 = np.array([
                [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [8, 8, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8],
                [8, 8, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8],
                [8, 8, 1, 1, 8, 8, 8, 8, 1, 1, 1, 1, 8, 8, 8, 8, 1, 1, 8, 8],
                [8, 8, 1, 1, 8, 8, 8, 8, 1, 1, 1, 1, 8, 8, 8, 8, 1, 1, 8, 8],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7]
            ])
        np.testing.assert_array_equal(board, board_1)

    def test_custom_board_array(self):
        " Tests that np.array input works successfully as a custom_board and that "
        " self.map_size is set to the appropriate value "
        env.custom_board = np.array([
                [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [8, 8, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8],
                [8, 8, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8],
                [8, 8, 1, 1, 8, 8, 8, 8, 1, 1, 1, 1, 8, 8, 8, 8, 1, 1, 8, 8],
                [8, 8, 1, 1, 8, 8, 8, 8, 1, 1, 1, 1, 8, 8, 8, 8, 1, 1, 8, 8],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7]
            ])
        env.reset(custom_board=env.custom_board)
        assert env.map_size == tuple(env._static_map.shape)

    def test_custom_board_invalid_input(self):
        " Tests that specific AttributeError is raised when input for custom_board is invalid "
        with raises(AttributeError) as err:
            env.reset(custom_board=[3,3,3])
        assert str(err.value) == 'Provided board must be either path(str) or matrix(np array).'

    # testing config_path

    def test_config_path_none(self):
        " Tests that no errors are raised when config_path is None "
        env.reset(config_path=None)

    def test_config_path_specified(self):
        " Tests that self.config_path variable is set to config_path input "
        env.reset(config_path='base_settings.ini')
        assert env.config_path == 'base_settings.ini'

    # general testing

    def test_general_self_radial(self):
        " Tests that self.radial is set appropriately when reset function is called "
        env = gym.make(ENV_NAME)
        env.reset()
        h, w = env.map_size
        Y, X = np.ogrid[:2*h, :2*w]
        np.testing.assert_array_equal(env._radial, (X-w)**2 + (Y-h)**2)

    def test_general_initialize_team(self):
        " Tests that some initialize team values are appropriate when reset function is called "
        env = gym.make(ENV_NAME)
        env.reset()
        assert env._agents == env._team_blue + env._team_red
        assert env.action_space == spaces.Discrete(len(env.ACTION) ** len(env._team_blue))

    def test_initialize_trajectory(self):
        " Test that trajectories are properly initialized with reset function "
        env = gym.make(ENV_NAME)
        env.reset()
        assert env._blue_trajectory == []
        assert env._red_trajectory == []

    def test_initial_variable_status(self):
        " Tests that general variables are properly initialized with reset function "
        " before game is over "
        env = gym.make(ENV_NAME)
        env.reset()
        assert env.blue_win == False
        assert env.red_win == False
        assert env.red_flag_captured == False
        assert env.blue_flag_captured == False
        assert env.red_eliminated == False
        assert env.blue_eliminated == False

    def test_initialization_human_mode(self):
        " Tests that variables relevant to human mode are properly initialized "
        " with reset function "
        env = gym.make(ENV_NAME)
        env.reset()
        assert env.first == True
        assert env.run_step == 0
        assert env.is_done == False

    def test_general_reset_output(self):
        " Tests that self.get_obs_blue is returned when the reset function is called "
        env = gym.make(ENV_NAME)
        env.reset()
        np.testing.assert_array_equal(env.reset(), env.get_obs_blue)
