# from gym import error, spaces, utils
# from gym.utils import seeding
# from .cap_view2d import CaptureView2D
from .const import *
import numpy as np
# from .create_map import CreateMap
#from .enemy_ai import EnemyAI
import math

class Agent:
    """This is a parent class for all agents.
    It creates an instance of agent in specific location"""

    def __init__(self, loc, static_map, team_number, unit_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            Agent object
        loc     : list
            [X,Y] location of unit
        """
        self.isAlive = True
        self.x, self.y = loc
        self._initial_loc = loc
        self.static_map = static_map

        self.team = team_number
        self.step = UGV_STEP
        self.range = UGV_RANGE
        self.a_range = UGV_A_RANGE
        self.level = 'ground'
        self.memory = np.zeros_like(static_map, dtype=bool)
        self.update_memory()
        #self.ai = EnemyAI(static_map)
        
        self.marker = None

        self.unit_type = unit_type
        self.channel = CHANNEL[unit_type]
        self.repr = REPRESENT[unit_type]

        # Movement and Interaction
        self.delay_count = 0
        self.delay = 0
        self.advantage = 1
        self.advantage_while_moving = 0

        ## Special Features
        self.visible = True
        self.clocking = False  # Hide with the move 0

    def revive(self):
        self.isAlive = True
        self.x, self.y = self._initial_loc

    def move(self, action, env, static_map):
        """
        Moves each unit individually. Checks if action is valid first.

        Parameters
        ----------
        self        : object
            CapEnv object
        action      : string
            Action the unit is to take
        env         : list
            the environment to move units in
        static_map   : list
            easily place the correct home tiles
        """
        
        if self.team == TEAM1_BACKGROUND:
            enemy_flag = TEAM2_FLAG 
        else:
            enemy_flag = TEAM1_FLAG

        # If agent is dead, dont move
        if not self.isAlive:
            dead_channel = CHANNEL[DEAD]
            if env[self.x][self.y][dead_channel] == REPRESENT[DEAD]:
                env[self.x][self.y][dead_channel] = 0
            env[self.x][self.y][self.channel] = 0
            return

        if self.delay_count < self.delay:
            self.delay_count += 1 
            return
        else:
            self.delay_count = 0

        channel = self.channel
        icon = self.repr
        collision_channels = list(set(CHANNEL[elem] for elem in LEVEL_GROUP[self.level]))
        
        if action == "X":
            if self.clocking:
                self.visible = False
                self.marker = (255,255,255) # If agent is hidden, mark with white 
            return
        
        elif action in ["N", "S", "E", "W"]:
            if self.clocking:
                self.visible = True
                self.marker = None
            dstep = {"N": [0 ,-1],
                     "S": [0 , 1],
                     "E": [1 , 0],
                     "W": [-1, 0]}[action]

            length, width = static_map.shape
            px, py = self.x, self.y
            nx, ny = px, py
            for s in range(self.step):
                px += dstep[0] 
                py += dstep[1]

                if px < 0 or px >= length: break
                if py < 0 or py >= width: break
                collide = False
                for ch in collision_channels:
                    if env[px, py, ch] != 0:
                        collide = True
                        break
                if collide:
                    break

                nx, ny = px, py
                # Interact with flag
                if env[px,py,CHANNEL[enemy_flag]] == REPRESENT[enemy_flag]: 
                    break

            # Not able to move
            if self.x == nx and self.y == ny: return

            # Make a movement
            env[self.x, self.y, channel] = 0
            env[nx, ny, channel] = icon
            self.x, self.y = nx, ny
        else:
            print("error: wrong action selected")
    
    def update_memory(self):
        """
        Update memory mask to current position
        """
        lx, ly = self.get_loc()
        for x in range(lx - self.range, lx + self.range + 1):
            for y in range(ly - self.range, ly + self.range + 1):
                if ((x - lx) ** 2 + (y - ly) ** 2 <= self.range ** 2) and \
                        0 <= x and x < self.memory.shape[1] and \
                        0 <= y and y < self.memory.shape[0]:
                    self.memory[y,x] = True
         
    def share_memory(self, update):
        """
        Update the memory
        """
        self.memory = np.logical_or(self.memory, update)
    
    def individual_reward(self, env):
        """
        Generates reward for individual
        :param self:
        :return:
        """
        # Small reward range [-1, 1]
        lx, ly = self.get_loc()
        small_observation = [[-1 for i in range(2 * self.range + 1)] for j in range(2 * self.range + 1)]
        small_reward = 0
        if self.is_air:
            for x in range(lx - self.range, lx + self.range + 1):
                for y in range(ly - self.range, ly + self.range + 1):
                    if ((x - lx) ** 2 + (y - ly) ** 2 <= self.range ** 2) and \
                            0 <= x < self.map_size[0] and \
                            0 <= y < self.map_size[1]:
                        small_observation[x - lx + self.range][y - ly + self.range] = self._env[x][y]
                        # Max reward for finding red flag
                        if env[x][y] == TEAM2_FLAG:
                            small_reward = .5
                        # Reward for UAV finding enemy wherever
                        elif env[x][y] == TEAM2_UGV:
                            small_reward += .5 / NUM_RED
        else:
            if env[lx][ly] == TEAM2_FLAG:
                small_reward = 1
            elif not self.isAlive:
                small_reward = -1
        return small_reward

    def get_loc(self):
        return self.x, self.y

    def report_loc(self):
        print("report: position x:%d, y:%d" % (self.x, self.y))

    def get_obs(self, env):
        lx, ly = self.get_loc()
        mx = env.map_size[0]-1
        my = env.map_size[1]-1
        obs = env.get_full_state
        obs[~self.memory] = UNKNOWN
        padwidth = ((mx-lx, lx),(my-ly, ly))
        return np.pad(obs, padwidth, mode='constant', constant_values=UNKNOWN)
        """
        com_air = env.COM_AIR
        com_ground = env.COM_GROUND
        com_distance = env.COM_DISTANCE
        com_frequency = env.COM_FREQUENCY

        if self.team == BLUE:
            myTeam = env.get_team_blue
        else:
            myTeam = env.get_team_red

        a = env.map_size[0]*2-1
        b = env.map_size[1]*2-1
        obs = np.full(shape=(a, b), fill_value=UNKNOWN)
        val = env.get_full_state

        if not self.isAlive:        # if target agent is dead, return all -1
            return obs

        x, y = self.get_loc()
        for i in range(-self.range, self.range + 1):
            for j in range(-self.range, self.range + 1):
                locx, locy = i + loc[0], j + loc[1]
                if (i * i + j * j <= self.range ** 2) and \
                        not (locx < 0 or locx > env.map_size[0] - 1) and \
                        not (locy < 0 or locy > env.map_size[1] - 1):
                    obs[locx+int(a/2)-loc[0]][locy+int(b/2)-loc[1]] = val[locx][locy]
                else:
                    obs[locx + int(a/2) - loc[0]][locy + int(b/2) - loc[1]] = UNKNOWN

        if not com_ground and not com_air:
            return obs

        for agent in myTeam:
            if not agent.isAlive:
                continue
            loc = agent.get_loc()
            if not com_distance == -1:
                if math.hypot(loc[0] - x, loc[1] - y) < com_distance:
                    continue
            if not com_air and agent.is_air:
                continue
            elif com_air and agent.is_air:
                for i in range(-agent.range, agent.range + 1):
                    for j in range(-agent.range, agent.range + 1):
                        locx, locy = i + loc[0], j + loc[1]
                        coordx = locx + int(a / 2) - x
                        coordy = locy + int(b / 2) - y
                        if (i * i + j * j <= agent.range ** 2) and \
                                not (locx < 0 or locx > env.map_size[0] - 1) and \
                                not (locy < 0 or locy > env.map_size[1] - 1):
                            obs[coordx][coordy] = val[locx][locy]

                            if com_frequency is not None and np.random.random() > com_frequency:
                                obs[coordx][coordy] = UNKNOWN

                        elif (0 <= coordx < a) and (0 <= coordy < b):
                            obs[coordx][coordy] = OBSTACLE

                            if com_frequency is not None and np.random.random() > com_frequency:
                                obs[coordx][coordy] = UNKNOWN

            elif not com_ground and not agent.is_air:
                continue
            elif com_ground and not agent.is_air:
                for i in range(-agent.range, agent.range + 1):
                    for j in range(-agent.range, agent.range + 1):
                        locx, locy = i + loc[0], j + loc[1]
                        coordx = locx + int(a / 2) - x
                        coordy = locy + int(b / 2) - y
                        if (i * i + j * j <= agent.range ** 2) and \
                                not (locx < 0 or locx > env.map_size[0] - 1) and \
                                not (locy < 0 or locy > env.map_size[1] - 1):
                            obs[coordx][coordy] = val[locx][locy]

                            if com_frequency is not None and np.random.random() > com_frequency:
                                obs[coordx][coordy] = UNKNOWN

                        elif (0 <= coordx < a) and (0 <= coordy < b):
                            obs[coordx][coordy] = OBSTACLE

                            if com_frequency is not None and np.random.random() > com_frequency:
                                obs[coordx][coordy] = UNKNOWN

        return obs
        """

    @property
    def is_air(self):
        return self.level=='air'

    @property
    def is_visible(self):
        return self.visible

    @property
    def get_advantage(self):
        if self.delay_count < self.delay: # moving
            return self.advantage_while_moving
        else:
            return self.advantage

class GroundVehicle(Agent):
    """This is a child class for ground agents. Inherited from Agent class.
    It creates an instance of UGV in specific location"""

    def __init__(self, loc, static_map, team_number, unit_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        Agent.__init__(self, loc, static_map, team_number, unit_type)


# noinspection PyCallByClass
class AerialVehicle(Agent):
    """This is a child class for aerial agents. Inherited from Agent class.
    It creates an instance of UAV in specific location"""

    def __init__(self, loc, static_map, team_number, unit_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        Agent.__init__(self, loc, static_map, team_number, unit_type)
        self.step = UAV_STEP
        self.range = UAV_RANGE
        self.a_range = UAV_A_RANGE
        self.level = 'air'
        self.advantage = 0

class GroundVehicle_Tank(Agent):
    """This is a child class for tank agents. Inherited from Agent class.
    It creates an instance of UGV2 in specific location"""

    def __init__(self, loc, static_map, team_number, unit_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        Agent.__init__(self, loc, static_map, team_number, unit_type)
        self.step = UGV2_STEP
        self.range = UGV2_RANGE
        self.a_range = UGV2_A_RANGE
        self.delay = UGV2_DELAY
        self.advantage = UGV2_ADVANTAGE
        self.advantage_while_moving = UGV2_ADVANTAGE_WHILE_MOVING

class GroundVehicle_Scout(Agent):
    """This is a child class for tank agents. Inherited from Agent class.
    It creates an instance of UGV3 in specific location"""

    def __init__(self, loc, static_map, team_number, unit_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        Agent.__init__(self, loc, static_map, team_number, unit_type)
        self.step = UGV3_STEP
        self.range = UGV3_RANGE
        self.delay = UGV3_DELAY
        self.a_range = UGV3_A_RANGE
        self.advantage = UGV3_ADVANTAGE
        self.advantage_while_moving = UGV3_ADVANTAGE_WHILE_MOVING

class GroundVehicle_Clocking(Agent):
    """This is a child class for tank agents. Inherited from Agent class.
    It creates an instance of UGV4 in specific location"""

    def __init__(self, loc, static_map, team_number, unit_type):
        Agent.__init__(self, loc, static_map, team_number, unit_type)
        self.clocking = True

class CivilAgent(GroundVehicle):
    """This is a child class for civil agents. Inherited from UGV class.
    It creates an instance of civil in specific location"""

    def __init__(self, loc, static_map, team_number, unit_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        Agent.__init__(self, loc, static_map, team_number, unit_type)
        self.direction = [0, 0]
        self.isDone = False
