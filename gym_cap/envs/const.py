# Game Constants
RED = 10
BLUE = 50
GRAY = 90
UAV_STEP = 3
UGV_STEP = 1
UAV_RANGE = 7
UGV_RANGE = 5
UAV_A_RANGE = 0
UGV_A_RANGE = 2

""" Advanced Units (Experiment)
UGV2 : Tank
UGV3 : Advanced Scout
UGV4 : Clocking Agent
"""
UGV2_STEP = 1
UGV2_DELAY = 2
UGV2_RANGE = 3
UGV2_A_RANGE = 2
UGV2_ADVANTAGE = 3
UGV2_ADVANTAGE_WHILE_MOVING = 1

UGV3_STEP = 3
UGV3_DELAY = 0
UGV3_RANGE = 3
UGV3_A_RANGE = 2
UGV3_ADVANTAGE = 1
UGV3_ADVANTAGE_WHILE_MOVING = 0

NP_SEED = None

# Element ID
SUGGESTION = -5
KNOWN = -4
FOG = -3
BLACK = -2
UNKNOWN = -1
TEAM1_BACKGROUND = 0
TEAM2_BACKGROUND = 1
TEAM1_UGV = 2
TEAM1_UAV = 3
TEAM2_UGV = 4
TEAM2_UAV = 5
TEAM1_FLAG = 6
TEAM2_FLAG = 7
OBSTACLE = 8
DEAD = 9
SELECTED = 10
COMPLETED = 11
TEAM3_UGV = 15

TEAM1_UGV2 = 16
TEAM2_UGV2 = 17
TEAM1_UGV3 = 18
TEAM2_UGV3 = 19
TEAM1_UGV4 = 20
TEAM2_UGV4 = 21

# Element Channel
NUM_CHANNEL = 6
CHANNEL = {
        UNKNOWN: 0,
        KNOWN: 0,
        FOG: 0,
        DEAD: 0,
        TEAM1_BACKGROUND: 1,
        TEAM2_BACKGROUND: 1,
        TEAM1_FLAG: 2,
        TEAM2_FLAG: 2,
        OBSTACLE: 3,
        TEAM1_UGV: 4,
        TEAM2_UGV: 4,
        TEAM1_UAV: 5,
        TEAM2_UAV: 5,
        TEAM1_UGV2: 4,
        TEAM2_UGV2: 4,
        TEAM1_UGV3: 4,
        TEAM2_UGV3: 4,
        TEAM1_UGV4: 4,
        TEAM2_UGV4: 4,
   }

# Represented constant
REPRESENT = {
        UNKNOWN: 1,
        KNOWN: 0,
        DEAD: 0,
        FOG: -1,
        OBSTACLE: 1,
        TEAM1_BACKGROUND: 1,
        TEAM2_BACKGROUND: -1,
        TEAM1_FLAG: 1,
        TEAM2_FLAG: -1,
        TEAM1_UGV: 1,
        TEAM2_UGV: -1,
        TEAM1_UAV: 1,
        TEAM2_UAV: -1,
        TEAM1_UGV2: 1,
        TEAM2_UGV2: -1,
        TEAM1_UGV3: 1,
        TEAM2_UGV3: -1,
        TEAM1_UGV4: 1,
        TEAM2_UGV4: -1,
    }

# Interaction Level
LEVEL_GROUP = {
        'ground': [OBSTACLE, TEAM1_UGV, TEAM2_UGV, TEAM1_UGV2, TEAM2_UGV2, TEAM1_UGV3, TEAM2_UGV3, TEAM1_UGV4, TEAM2_UGV4],
        'air'   : [TEAM1_UAV, TEAM2_UAV]
    }

# Rendering
COLOR_DICT = {
        KNOWN : (255, 255, 255),
        UNKNOWN : (200, 200, 200),
        FOG : (200, 200, 200),
        TEAM1_BACKGROUND : (0, 0, 120),
        TEAM2_BACKGROUND : (120, 0, 0),
        TEAM1_UGV : (0, 0, 255),
        TEAM1_UAV : (55, 55, 230),
        TEAM2_UGV : (255, 0, 0),
        TEAM2_UAV :  (230, 55, 55),
        TEAM1_FLAG : (0, 255, 255),
        TEAM2_FLAG : (255, 255, 0),
        OBSTACLE : (120, 120, 120),
        TEAM3_UGV : (180, 180, 180),
        DEAD : (0, 0, 0),
        SELECTED : (122, 77, 25),
        BLACK : (0, 0, 0),
        SUGGESTION : (50, 50, 50),
        COMPLETED : (100, 0, 0),
        TEAM1_UGV2 : (0,0,240),
        TEAM2_UGV2 : (240,0,0),
        TEAM1_UGV3 : (0,0,240),
        TEAM2_UGV3 : (240,0,0),
        TEAM1_UGV4 : (0,0,240),
        TEAM2_UGV4 : (240,0,0),
    }

