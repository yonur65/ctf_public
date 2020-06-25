import gym
import gym_cap

from gym_cap.envs.cap_env import *
from gym_cap.envs import agent
import gym_cap.heuristic as policy

from pytest import raises

from random import randint

ENV_NAME = 'cap-v0'
env = gym.make(ENV_NAME)

class TestStep:
    """
    Tests the step function in gym_cap.envs.cap_env using pytest

    """

    # testing for self.is_done if statement

    def test_is_done_None(step):
        " Test that no errors occur when self.is_done is activated in cap_env file for None input "
        env.is_done = True
        env.CONTROL_ALL = False
        env.step(None)

    def test_is_done_int(step):
        " Test that no errors occur when self.is_done is activated in cap_env file for int input "
        env.is_done = True
        env.CONTROL_ALL = False
        env.step(3)

    def test_is_done_list(self):
        " Test that no errors occur when self.is_done is activated in cap_env file for list input "
        env.is_done = True
        env.CONTROL_ALL = False
        env.step([3,4,2,1,2,2,3,2,1,0,3,2])

    # testing for self.CONTROL_ALL if statement

    def test_control_all_red_step_one(self):
        " Test if RED_STEP parameter is set to 1 when CONTROL_ALL is activated "
        env.is_done = False
        env.CONTROL_ALL = True
        assert env.RED_STEP == 1

    def test_control_all_red_step_not_one(self):
        " Test that AssertionError is raised if RED_STEP parameter is not set to 1 "
        env.is_done = False
        env.CONTROL_ALL = True
        with raises(AssertionError):
            assert env.RED_STEP == 2

    def test_control_all_none(self):
        " Test that a specific AssertionError is raised when entities_action is None "
        env.is_done = False
        env.CONTROL_ALL = True
        with raises(AssertionError) as err:
            env.step(None)
        assert str(err.value) == 'Under CONTROL_ALL setting, action must be specified'

    def test_control_all_int(self):
        " Test that a specific AssertionError is raised when entities_action is int "
        env.is_done = False
        env.CONTROL_ALL = True
        with raises(AssertionError) as err:
            env.step(3)
        assert str(err.value) == 'CONTROLL_ALL setting requires list (or numpy array) type of action'

    def test_control_all_list_wrong_move(self):
        " Test that a specific AssertionError is raised when wrong number of moves are inputted "
        env.is_done = False
        env.CONTROL_ALL = True
        with raises(AssertionError) as err:
            env.step([0,3,2])
        assert str(err.value) == 'You entered wrong number of moves.'

    def test_control_all_list_right_move(self):
        " Test that step function is a success when input is between 0 and 4 and "
        " the number of moves equals the len(_team_blue + _team_red) "
        env.is_done = False
        env.CONTROL_ALL = True
        env.step([0,2,3,2,2,2,4,1,3,3,1,3]) # 12 moves required, input is from 0 to 4

    def test_control_all_list_invalid_input_range(self):
        " Test that an IndexError is raised when any input value is less than -5 or greater than 4 "
        env.is_done = False
        env.CONTROL_ALL = True
        with raises(IndexError):
            env.step([0,1,1,7,2,3,5,2,9,2,3,5]) # input cannot be greater than 4
        with raises(IndexError):
            env.step([0,-6,-1,4,2,3,4,2,4,2,3,4]) # input cannot be less than -5

    # testing for else statement

    def test_else_none(self):
        " Test that no error occurs when entities_action is None because exception is excepted "
        env.is_done = False
        env.CONTROL_ALL = False
        env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
        env.step(None)

    def test_else_int(self):
        " Test that when an integer is given, no error occurs as long as too many moves have not been entered "
        env.is_done = False
        env.CONTROL_ALL = False
        env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
        env.step(3)

    def test_else_list_wrong_moves(self):
        " Test that SystemExit occurs when incorrect number of moves are entered for else statement "
        env.is_done = False
        env.CONTROL_ALL = False
        env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
        with raises(SystemExit) as sysex:
            env.step([0,1,4])
        assert str(sysex.value) == 'ERROR: You entered wrong number of moves. There are 6 entities.'

    def test_else_list_right_moves(self):
        " Test that when 6 moves are entered for the blue team, the step function is a success "
        env.is_done = False
        env.CONTROL_ALL = False
        env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
        env.step([0,2,1,3,4,2])

    def test_else_list_invalid_input(self):
        " Test that when values that are greater than 4 or less than -5 are entered "
        " an IndexError is raised "
        env.is_done = False
        env.CONTROL_ALL = False
        env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
        with raises(IndexError) as err1:
            env.step([5,6,3,9,100,2])
        with raises(IndexError) as err2:
            env.step([-4,-4,-6,-3,-2,-1])
        assert str(err1.value) == 'list index out of range'
        assert str(err2.value) == 'list index out of range'

    # general testing

    def test_general_blue_action_specified(self):
        " Test that no errors occur when blue action is specified using env.action_space.sample() "
        env = gym.make(ENV_NAME)
        env.reset(policy_red=policy.Random())
        for step in range(1000):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                break

    def test_general_random_actions(self):
        " Test that the step function is successful when random policies are given to both teams "
        env = gym.make(ENV_NAME)
        env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
        for step in range(1000):
            state, reward, done, info = env.step()
            if done:
                break

    def test_general_no_actions_specified(self):
        " Test that SystemExit occurs when no actions are specified for either team "
        with raises(SystemExit):
            env = gym.make(ENV_NAME)
            env.reset()
            env.step()

    def test_general_equal_number_of_agents(self):
        " Test that the step function is successful with different number of agents "
        env = gym.make(ENV_NAME)
        max_agents = 20 # arbitrary value
        for num_agents in range(max_agents):
            env.NUM_BLUE = num_agents
            env.NUM_RED = num_agents
            env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
            env.step()

    def test_general_differing_number_of_agents(self):
        " Test that the step function is successful with different number of agents "
        env = gym.make(ENV_NAME)
        num_tests = 100
        for instance in range(num_tests):
            env.NUM_BLUE = randint(0, 20)
            env.NUM_RED = randint(0, 20)
            env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
            env.step()

    def test_general_num_agents_none(self):
        " Test that TypeError is raised when None value is entered for the number of blue or red agents "
        env = gym.make(ENV_NAME)
        with raises(TypeError):
            env.NUM_BLUE = None
            env.NUM_RED = 3
            env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
            env.step()

    def test_general_num_agents_list(self):
        " Test that TypeError is raised when list value is entered for the number of blue or red agents "
        env = gym.make(ENV_NAME)
        with raises(TypeError):
            env.NUM_BLUE = [3,4,2]
            env.NUM_RED = 3
            env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
            env.step()

    def test_general_different_number_of_agents(self):
        " Test that the step function is successful with different number of agents "
        env = gym.make(ENV_NAME)
        env.NUM_BLUE = 0
        env.NUM_RED = 0
        env.NUM_BLUE_UAV = 0
        env.NUM_RED_UAV = 0
        env.NUM_GRAY = 0
        env.NUM_BLUE_UGV2 = 0
        env.NUM_RED_UGV2 = 0
        env.NUM_BLUE_UGV3 = 0
        env.NUM_RED_UGV3 = 0
        env.NUM_BLUE_UGV4 = 0
        env.NUM_RED_UGV4 = 0
        env.reset(policy_blue=policy.Random(), policy_red=policy.Random())
        env.step()

    # interaction testing

    def test_interaction_target_agents(self):
        " Test that environment variables for target_agents component of step "
        " function are appropriate "
        env.step()
        agent.isAlive = True
        agent.is_air = False
        target_agents = [agent for agent in env._agents if agent.isAlive and not agent.is_air]
        new_status = env._interaction(target_agents)
        for status, entity in zip(new_status, target_agents):
            assert entity.isAlive == status

    # flag conditions testing
