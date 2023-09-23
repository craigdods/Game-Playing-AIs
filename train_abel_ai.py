'''
Abel AI - A Double DQN with Dual Input, Temporal LSTM Processing, Heuristic-Guided Mixed Precision, CNN for Spatial Data, and Efficient Dual-Storage Replay Buffer.

This is my reinforcement-learning agent, inspired by the 2017 efforts of Steven Brown (PySC2 Dev) who used Q-Learning in his blog series: https://github.com/skjb/pysc2-tutorial/tree/master
The PySC2 test harness that he built for his agents is leveraged here (somewhat...) so that the DQN can properly interact with the StarCraft II / PySC2 environment.
Out of the 1800~ Lines-of-Code, approximately 1500~ are net-new code written by me, with the 300~ lines involving boilerplate code responsible for basic SC2/PySC2 multi-step functions like creating buildings/units

To modify training behaviour (Self-Reinforcement, Built-in Bots, or Kane, modify the TRAINING_TYPE Global knob on line 61)

To debug (outside of the PySC2 Run Loop Environment) to see Python Tracebacks, use this string:
python -m pysc2.bin.agent --map Simple64 --agent train_abel_ai.DQNAgent --agent_race terran --max_agent_steps 0 --norender --use_feature_units --difficulty very_easy --agent2_race terran --nosave_replay --action_space RGB --rgb_minimap_size 64 --rgb_screen_size 84 --game_steps_per_episode 21000
---- add to readme:
file-descriptor limits need to be artificially raised in /etc/security/limits.conf:
craig		soft	nofile		8192
craig 	hard	nofile		1048576
Additionally, tensorboard needs to be run as root (due to higher FD limits):
ulimit -n 1048576
tensorboard --logdir=runs

Validated with: ulimit -s -H && ulimit -n -S
required as after 1k episodes we run out of FD's:
FILE OPEN ERROR: for /.../StarCraft II/2023-08-14 02T45T30.000.test: Too many open files
'''

# Import our other agents
from Kane_AI import KaneAI
from test_abel_ai import DQNAgent as test_DQN_Agent

import random
from random import sample as random_sample
import time
import math

from absl import app
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# queue used for replay buffer
from collections import deque
# from action-reptition detection in our rewards queue (and tensorboard metrics...)
from collections import Counter

# Using TensorBoard for model performance tracking & visualizations
from torch.utils.tensorboard import SummaryWriter
# Using amp for mixed-precision (FP16) to improve RTX 4090 performance
from torch.cuda.amp import autocast, GradScaler
# Using a learning Rate Scheduler to help the model converge faster / avoid getting stuck
# The static learning rate of 0.01 was...quite poor excellent.
from torch.optim.lr_scheduler import CosineAnnealingLR
# Using itertools to slice deque's efficiently
import itertools

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units
from pysc2.env import sc2_env
from pysc2.env import run_loop

# Global Configuration Knobs
_TRAINING_TYPE = "kane"  # Possible values: "self", "kane", "bot"
# Possible values: "very_easy", "easy", "medium", "hard" (only used if _TRAINING_TYPE = "bot")
_BOT_DIFFICULTY = "easy"
# Enable/Disable Training Guardrails
_TRAINING_GUARDRAILS_ENABLED = False

# There is a predictable SC2 SegFault after every 986th game in multi-custom-agent environments, this code is a workaround
_MAX_GAMES_BEFORE_RESTART = 500


class CustomRestartException(Exception):
    pass

# BoilerPlate code from Steven Brown's Q-Learning Implementation found here:
# https://github.com/skjb/pysc2-tutorial/tree/master
# Specifically, the Q-Learning bot:
# https://github.com/skjb/pysc2-tutorial/blob/master/Refining%20the%20Sparse%20Reward%20Agent/refined_agent.py
# His boilerplate code was written in 2017 and needed some updating (e.g many pandas functions have been deprecated)


################################## Start of BoilerPlate Code #####################################################
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]


ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'
####### CUSTOM CODE ######
#
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
# Specify Previous Network-Compatable Weights:
_INITIAL_WEIGHTS = 'abel_final_weights.pt'
# Specify Naming Scheme of Future Model Checkpoints
_DATA_FILE = 'ddqn-cnn-lstm-agent-model-v22-a.pt'
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id

ACTION_BUILD_SCV = 'buildscv'
ACTION_ATTACK_UNIT = 'attackunit'
ACTION_RESET_CAMERA = 'resetcamera'
ACTION_MOVE_CAMERA_SELF_EXPANSION = 'mcselfexpansion'
ACTION_MOVE_CAMERA_ENEMY_EXPANSION = 'mcenemyexpansion'
ACTION_MOVE_CAMERA_ENEMY_PRIMARY = 'mcenemyprimary'
###### END OF GLOBAL CUSTOM CODE #####

# Steven Brown created the PySC2 concept of smart_actions although for this bot I've expanded them considerably
smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_ATTACK_UNIT,
    ACTION_BUILD_SCV,
    ACTION_RESET_CAMERA,
    ACTION_MOVE_CAMERA_SELF_EXPANSION,
    ACTION_MOVE_CAMERA_ENEMY_EXPANSION,
    ACTION_MOVE_CAMERA_ENEMY_PRIMARY,
]

# DQN Non-Spatial State size
STATE_SIZE = 2

################################## End of BoilerPlate Code #####################################################

# Custom library of all terran buildings
TERRAN_BUILDINGS = [
    18,  # Command Center
    19,  # Supply Depot
    20,  # Refinery
    21,  # Barracks
    22,  # Orbital Command
    23,  # Factory
    24,  # Starport
    25,  # Engineering Bay
    26,  # Fusion Core
    27,  # Tech Lab (Barracks)
    28,  # Tech Lab (Factory)
    29,  # Tech Lab (Starport)
    30,  # Reactor (generic, as the building morphs)
    37,  # Sensor Tower
    38,  # Bunker
    39,  # Missile Turret
    40,  # Auto-turret (from Raven)
    58,  # Planetary Fortress
]

TERRAN_UNITS = {
    # Worker
    45: "SCV",

    # Basic
    48: "MULE",
    51: "Marine",
    53: "Marauder",
    54: "Reaper",
    55: "Ghost",

    # Factory
    57: "Hellion",
    58: "SiegeTank",
    59: "Cyclone",
    62: "Thor",
    64: "Hellbat",

    # Starport
    35: "VikingFighter",
    36: "VikingAssault",
    67: "Medivac",
    68: "Banshee",
    69: "Raven",
    70: "Battlecruiser",
    132: "Liberator",
    71: "AutoTurret",  # Spawned by Raven

    # Other
    102: "WidowMine",
    488: "LiberatorAG"
}

# Define the top-left coordinates for each quadrant
quadrants = [
    (0, 0),       # Top left quadrant
    (32, 0),      # Top right quadrant
    (0, 32),      # Bottom left quadrant
    (32, 32)      # Bottom right quadrant
]

# Calculate the offset points for each quadrant
# This is used in the smart_attack functions (16 locations with offsets)
# Inspiration was Steven Brown's logic, however this is...highly modified (16 locations in mini-quadrants vs 4, complex offsets, etc)


def calculate_quadrant_points(top_left_x, top_left_y, quadrant):
    corner_offset = 3
    mini_quadrant_size = 16  # Each quadrant is divided further into 4

    if quadrant == "top-left":
        points = [
            # Top-left of top-left mini quadrant
            (top_left_x + corner_offset, top_left_y + corner_offset),
            (top_left_x + mini_quadrant_size + corner_offset, top_left_y + \
             corner_offset),  # Top-left of top-right mini quadrant
            (top_left_x + corner_offset, top_left_y + mini_quadrant_size + \
             corner_offset),  # Top-left of bottom-left mini quadrant
            (top_left_x + mini_quadrant_size + corner_offset, top_left_y + \
             mini_quadrant_size + corner_offset)  # Top-left of bottom-right mini quadrant
        ]
    elif quadrant == "top-right":
        points = [
            # Top-right of top-left mini quadrant
            (top_left_x + mini_quadrant_size - \
             corner_offset, top_left_y + corner_offset),
            (top_left_x + 2 * mini_quadrant_size - corner_offset, top_left_y + \
             corner_offset),               # Top-right of top-right mini quadrant
            (top_left_x + mini_quadrant_size - corner_offset, top_left_y + \
             mini_quadrant_size + corner_offset),  # Top-right of bottom-left mini quadrant
            (top_left_x + 2 * mini_quadrant_size - corner_offset, top_left_y + \
             mini_quadrant_size + corner_offset)  # Top-right of bottom-right mini quadrant
        ]
    elif quadrant == "bottom-left":
        points = [
            # Bottom-left of top-left mini quadrant
            (top_left_x + corner_offset, top_left_y + \
             mini_quadrant_size - corner_offset),
            (top_left_x + mini_quadrant_size + corner_offset, top_left_y + \
             mini_quadrant_size - corner_offset),  # Bottom-left of top-right mini quadrant
            (top_left_x + corner_offset, top_left_y + 2 * mini_quadrant_size - \
             corner_offset),               # Bottom-left of bottom-left mini quadrant
            (top_left_x + mini_quadrant_size + corner_offset, top_left_y + 2 * \
             mini_quadrant_size - corner_offset)  # Bottom-left of bottom-right mini quadrant
        ]
    elif quadrant == "bottom-right":
        points = [
            (top_left_x + mini_quadrant_size - corner_offset, top_left_y +
             mini_quadrant_size - corner_offset),     # Bottom-right of top-left mini quadrant
            (top_left_x + 2 * mini_quadrant_size - corner_offset, top_left_y + \
             mini_quadrant_size - corner_offset),  # Bottom-right of top-right mini quadrant
            (top_left_x + mini_quadrant_size - corner_offset, top_left_y + 2 * \
             mini_quadrant_size - corner_offset),  # Bottom-right of bottom-left mini quadrant
            (top_left_x + 2 * mini_quadrant_size - corner_offset, top_left_y + 2 * \
             mini_quadrant_size - corner_offset)  # Bottom-right of bottom-right mini quadrant
        ]

    return points


# For each quadrant, calculate the offset points and append the attack action
quadrant_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
for i, quad in enumerate(quadrants):
    points = calculate_quadrant_points(*quad, quadrant_names[i])
    for x, y in points:
        smart_actions.append(ACTION_ATTACK + '_' + str(x) + '_' + str(y))


# print("smart_actions is set to: ", smart_actions)
print("--------------------")

print("# Action Mapping")
for index, action in enumerate(smart_actions):
    print(f"# {index}: '{action}'")
print("--------------------")

# --------------------
# Action Mapping
# 0: 'donothing'
# 1: 'buildsupplydepot'
# 2: 'buildbarracks'
# 3: 'buildmarine'
# 4: 'attackunit'
# 5: 'buildscv'
# 6: 'resetcamera'
# 7: 'mcselfexpansion'
# 8: 'mcenemyexpansion'
# 9: 'mcenemyprimary'
# 10: 'attack_3_3'
# 11: 'attack_19_3'
# 12: 'attack_3_19'
# 13: 'attack_19_19'
# 14: 'attack_45_3'
# 15: 'attack_61_3'
# 16: 'attack_45_19'
# 17: 'attack_61_19'
# 18: 'attack_3_45'
# 19: 'attack_19_45'
# 20: 'attack_3_61'
# 21: 'attack_19_61'
# 22: 'attack_45_45'
# 23: 'attack_61_45'
# 24: 'attack_45_61'
# 25: 'attack_61_61'
# --------------------


# Custom Dual DQN Agent implementation with a replay buffer of 2M, gamma of 0.90 (hopefully prioritizing longer term play), and batch_size of 512 (confirmed optimal via hyperparameter search)
# Technically I believe this architecture could be called a "Double DQN with Dual Input, Temporal LSTM Processing, Heuristic-Guided Mixed Precision, CNN for Spatial Data, and Efficient Dual-Storage Replay Buffer."
# To solve the replay buffer's O(n) random lookup issue in Python's `deque`, I have created two separate buffers:
# 1. The 1.5M size deque where state/actions are appended/popped directly in O(1) time per game step
# 2. A python list with O(1) time for random lookups. This is copied once from the deque every 10 games/episodes in O(N) time, resulting in large performance improvements
# Particularly as training occurs multiple times per game (every 100 actions), and fetching 1k items in O(n) time dramatically slowed down gameplay
# The drawback of course is that space complexity is 2N...
class DQNModel(nn.Module):
    def __init__(self, actions, non_spatial_state_size, minimap_shape, learning_rate=0.00075, gamma=0.90, e_greedy=0.15, buffer_capacity=2000000, batch_size=512):
        super(DQNModel, self).__init__()
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma
        # Using linear epsilon decay to reduce random action probability over time
        self.epsilon = e_greedy
        # Final decayed epsilon will result in a random action being taken 10% of the time (down from 90%)
        self.final_epsilon = 0.01
        # We decay over 1.5M root-level actions taken (no real way to track per-episode easily with PySC2 from the model's perspective)
        self.epsilon_decay_rate = (
            self.epsilon - self.final_epsilon) / 1000000
        self.action_counter = 0
        #
        self.state_size = non_spatial_state_size
        self.disallowed_actions = {}
        # Replay buffer deque and list
        self.buffer_capacity = buffer_capacity
        self.buffer = deque(maxlen=buffer_capacity)
        self.training_buffer = []
        # self.position = 0
        self.batch_size = batch_size
        # Mixed Precision
        self.scaler = GradScaler()
        # Counter for tracking how frequently training is run
        self.global_training_steps = 0
        # Using a method for Tensorboard writer to avoid having issues when PySC2 crashes...
        self.writer_path = 'runs/ddqn-cnn-lstm-agent-model-v22-d-kane-NGR'
        # Our replay buffer training theshold size
        self.training_buffer_requirement = 30000
        # This is our custom reward/action mapping dictionary
        # This is leveraged within the `get_normalized_reward` function to ensure useful actions are appropriately rewarded
        self.action_rewards = {
            0: -0.25,  # 'donothing' - Need to negatively reinforce 'doing nothing' as the AI sees it as a 'safe' move in perpetuity
            1: 0.9,     # 'buildsupplydepot'
            2: 0.9,  # 'buildbarracks'
            3: 1,   # 'buildmarine'
            4: 0.4,   # 'attackunit' - High reward, attack visible units
            5: 0.9,   # 'buildscv'
            6: 0.4,   # 'resetcamera' - High reward as is required when available
            7: 0.05,  # 'mcselfexpansion' - moves camera - checks our expansion
            8: 0.1,   # 'mcenemyexpansion' - moves camera - checks enemy expansion
            9: 0.15,  # 'mcenemyprimary' - moves camera - checks enemy primary base
            10: 0,    # 'attack_3_3'
            11: 0,    # 'attack_19_3'
            12: 0,    # 'attack_3_19'
            13: 0,    # 'attack_19_19'
            14: 0,    # 'attack_45_3'
            15: 0,    # 'attack_61_3'
            16: 0,    # 'attack_45_19'
            17: 0,    # 'attack_61_19'
            18: 0,    # 'attack_3_45'
            19: 0,    # 'attack_19_45'
            20: 0,    # 'attack_3_61'
            21: 0,    # 'attack_19_61'
            22: 0,    # 'attack_45_45'
            23: 0,    # 'attack_61_45'
            24: 0,    # 'attack_45_61'
            25: 0     # 'attack_61_61'
        }

        # Setting up quadrant mapping to avoid repetitive copies in the hot path
        # Define which actions correspond to which quadrants
        # When issuing an attack-minimap action, locations are normalized for the agent based on spawn location
        # With the function transformLocation()
        self.top_left_actions = [10, 11, 12, 13]
        self.top_right_actions = [14, 15, 16, 17]
        self.bottom_left_actions = [18, 19, 20, 21]
        self.bottom_right_actions = [22, 23, 24, 25]

        # Attempt GPU acceleration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # CNN for RGB minimap with normalisation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ).to(self.device)  # Moving to GPU if available

        # Calculate the size of our flattened output after convolutional layers
        self.conv_out_size = self._get_conv_out(minimap_shape)

        # Fully Connected Network (FCN) for non-spatial data
        # Not using dropout as it loses our limited/sparse data (unfortunately...)
        # Overfitting hasn't been an issue with this particular setup
        self.fc_non_spatial = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
        ).to(self.device)  # Moving to GPU if available

        # Decision-making LSTM pass-through (takes concatenated, processed outputs from CNN and FCN, hits the LSTM, then to a single 4k->action_length layer)
        # Offloads to GPU if available
        self.lstm_decision = nn.LSTM(
            input_size=8896, hidden_size=4096, num_layers=1, batch_first=True).to(self.device)
        self.fc_after_lstm = nn.Linear(4096, len(self.actions)).to(
            self.device)  # to get action probabilities/scores

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Our learning rate scheduler is enabled here (CosineAnnealing)
        # The goal is to help the model move out of local minima (this happened a lot with a static learning rate)
        # Using a min of 0.0001 to avoid zeroing out and potentially regressing...
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=0.0001)

    #
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.size()))

    # This is an implicit PyTorch function that's called automatically to feed output between networks
    def forward(self, non_spatial_data, minimap):
        non_spatial_data = non_spatial_data.to(self.device)
        minimap = minimap.to(self.device)

        # Separate treatments for different types of data
        conv_out = self.conv(minimap).reshape(minimap.size()[0], -1)
        fc_out = self.fc_non_spatial(non_spatial_data).reshape(
            non_spatial_data.size()[0], -1)  # Flatten the tensor to 2D

        # Combine both outputs
        # LSTM requires a sequence dimension...
        combined = torch.cat((conv_out, fc_out), dim=1).unsqueeze(1)

        lstm_out, _ = self.lstm_decision(combined)
        # And now we remove the sequence dimension
        lstm_out = lstm_out.squeeze(1)

        return self.fc_after_lstm(lstm_out)

    # Using a method to handle writes to ensure things are closed properly in the event of a crash
    def get_writer(self):
        return SummaryWriter(self.writer_path)

    # This is where we store items for our replay buffer
    # s: The current state of the environment.
    # a: The action taken by the agent in state s.
    # r: The reward received after taking action a in state s.
    # s_next: The resulting state after taking
    def store_transition(self, s, a, r, s_next):
        # Transition is a tuple (s, a, r, s_next)
        transition = (s, a, r, s_next)
        # Append the transition to the replay buffer
        self.buffer.append(transition)

    def transfer_buffer(self):
        self.training_buffer = list(self.buffer)

        #####
        # Debugging
        # Extract the components of the last element in the buffer
        # s, a, r, s_next = self.training_buffer[-1]

        # # Print the components
        # print("s:", s)
        # print("a:", a)
        # print("r:", r)
        # print("s_next:", s_next)

        # print("\nlast transfer buffer element is set to:", self.training_buffer[-1])
        #####

    # This function goes back and tweaks the rewards associated with a given game
    # Based on the final tangible reward we get from PySC2
    # Win/loss/draw are multipliers
    # This backpropagation is slow O(n) but needs to be done in the deque prior to copying to the list in the current code
    # Otherwise, we'd have to store in multi-game chunks and append it later (will perform better but...complexity/time issues)

    def backpropagate_final_reward(self, final_reward, root_actions_taken_last_game):
        # Calculate the number of steps to iterate over, limited by the length of the buffer to ensure safety
        steps_to_iterate = min(root_actions_taken_last_game, len(self.buffer))

        # Iterate over the last steps_to_iterate in the replay buffer/queue...in reverse order (newest to oldest)
        for i in range(-steps_to_iterate, 0):
            # Skip if the buffer index result is None
            if self.buffer[i] is None:
                continue
            # Fetch the transition
            s, a, r, s_next = self.buffer[i]

            # Modify the reward to include the final reward
            new_reward = r * final_reward
            # print("Before backpropagation: ", r)

            # Replace the transition in the replay buffer
            self.buffer[i] = (s, a, new_reward, s_next)
            # print("After backpropagation: ", self.buffer[i][2])

        print("Replay buffer currently has: ",
              len(self.training_buffer), "entries")

    # # Sample random transitions from replay buffer
    # def random_sample(self, batch_size):
        # return random_sample(self.training_buffer, batch_size)

    def random_sample(self, batch_size):
        """
        This function implements a biased random sampling strategy.
        Instead of uniformly sampling the replay buffer, we introduce a bias
        towards more recent experiences. This has an effect similar to a
        stochastic prioritization, where more recent experiences have a
        higher probability of being sampled compared to older experiences.

        The intuition behind this is that more recent experiences might be
        more relevant to the current policy, and therefore might benefit
        the agent more during training.
        """

        # Determine the count for recent experiences
        # Here, we consider the most recent 20% of the replay buffer.
        recent_count = int(0.2 * len(self.training_buffer))

        # We take 30% of our batch size from recent experiences
        # and the rest from the older experiences.
        num_from_recent = int(0.3 * batch_size)
        num_from_older = batch_size - num_from_recent

        # Sample from the recent experiences segment.
        # Larger index values are newer (it's copied directly from a queue, where the newest entries are the last)
        recent_samples = random.sample(
            self.training_buffer[-recent_count:], num_from_recent)

        # Sample from the older experiences segment.
        older_samples = random.sample(
            self.training_buffer[:-recent_count], num_from_older)

        # Combine and return the samples from both segments.
        return recent_samples + older_samples

    # The basic framework of the function `choose_action`` was leveraged from the initial Q-Learning
    # Implementation that Steven Brown (PySC2 Dev) created here:
    # https://github.com/skjb/pysc2-tutorial/blob/master/Refining%20the%20Sparse%20Reward%20Agent/refined_agent.py
    # I've of course modified it to use linear decay, a DQN with minimap-CNN (via torch) instead of Q-Learning with basic hot-squares, etc but...the initial work is his
    def choose_action(self, current_state, excluded_actions=[]):
        # Extract non_spatial_data and rgb_minimap from current_state
        non_spatial_data = current_state["non_spatial"]
        rgb_minimap = current_state["rgb_minimap"]

        # Debugs
        # print("Non-spatial data shape before unsqueeze:", non_spatial_data.shape)

        # Set eval mode 'on' to avoid breaking BatchNorm - avoids learning
        self.eval()

        # Convert data to tensors and move them to the GPU
        non_spatial_data_tensor = torch.tensor(
            non_spatial_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Minimap needs to have its dimensions changed slightly as PyTorch expects colours first (3,64,64) versus (64,64,3)
        # Unsqueeze is also required as PyTorch expects batch size in the first position. In our case, that's: [1,3,64,64]
        rgb_minimap_tensor = torch.tensor(rgb_minimap, dtype=torch.float32).permute(
            2, 0, 1).unsqueeze(0).to(self.device)

        # Epsilon-based exploration
        if np.random.uniform() < self.epsilon:
            available_actions = [
                a for a in self.actions if a not in excluded_actions]
            # print("Available actions are:", available_actions)
            action = np.random.choice(available_actions)
        else:
            # Pass the minimap data through the CNN
            conv_output = self.conv(rgb_minimap_tensor)
            # No need for the third dimension anymore
            conv_output = conv_output.view(conv_output.size(0), -1)

            # Pass the non-spatial data through its layers
            # print("Shape of non_spatial_data_tensor:", non_spatial_data_tensor.shape)
            non_spatial_output = self.fc_non_spatial(non_spatial_data_tensor)
            non_spatial_output = non_spatial_output.view(
                non_spatial_output.size(0), -1)  # No need for the third dimension

            # Concatenate the two outputs along the second dimension (feature axis)
            combined_output = torch.cat((conv_output, non_spatial_output), dim=1).unsqueeze(
                1)  # Adding sequence dimension for LSTM
            # print("------------Combined tensor shape:", combined_output.shape)

            # Pass through LSTM and subsequent fully connected layer
            lstm_out, _ = self.lstm_decision(combined_output)
            lstm_out = lstm_out.squeeze(1)  # Removing sequence dimension
            q_values = self.fc_after_lstm(lstm_out)

            # Set our excluded actions for the step to negative infinity, ensuring they're not selected by the model
            for action in excluded_actions:
                q_values[0][action] = float('-inf')
            action = torch.argmax(q_values).item()

        # This is where we keep the logic for epsilon decay (linear)
        self.action_counter += 1
        self.epsilon -= self.epsilon_decay_rate
        self.epsilon = max(self.final_epsilon, self.epsilon)

        # Turn training mode back on
        self.train()
        return action

    # This is where we train the model
    # It samples randomly from the replay buffer - relatively simplistic compared to PER but seemingly effective!
    def learn(self, target_network):
        # Check if the replay buffer has enough samples (using 500K as minimum)
        if len(self.training_buffer) < self.training_buffer_requirement:
            print("Replay buffer is currently too small to conduct training...")
            return

        # Increment our training counter
        self.global_training_steps += 1

        # Sample a mini-batch of transitions from the buffer
        transitions = self.random_sample(self.batch_size)
        # Unzip the transitions into separate variables
        states, actions, rewards, next_states = zip(*transitions)

        # Separate non_spatial and rgb_minimap components of states
        non_spatial_states = [state["non_spatial"] for state in states]
        rgb_minimap_states = [state["rgb_minimap"] for state in states]

        non_spatial_next_states = [state["non_spatial"]
                                   for state in next_states]
        rgb_minimap_next_states = [state["rgb_minimap"]
                                   for state in next_states]

        # Convert the zipped values to numpy arrays
        non_spatial_states_np = np.array(non_spatial_states, dtype=np.float32)
        rgb_minimap_states_np = np.array(rgb_minimap_states, dtype=np.float32)
        non_spatial_next_states_np = np.array(
            non_spatial_next_states, dtype=np.float32)
        rgb_minimap_next_states_np = np.array(
            rgb_minimap_next_states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)

        # Convert numpy arrays to tensors
        non_spatial_states = torch.tensor(
            non_spatial_states_np).to(self.device)
        # Minimap requires permutation to meet PyTorch expectations
        rgb_minimap_states = torch.tensor(rgb_minimap_states_np).to(
            self.device).permute(0, 3, 1, 2)
        non_spatial_next_states = torch.tensor(
            non_spatial_next_states_np).to(self.device)
        # Minimap requires permutation to meet PyTorch expectations
        rgb_minimap_next_states = torch.tensor(
            rgb_minimap_next_states_np).to(self.device).permute(0, 3, 1, 2)

        actions = torch.tensor(actions_np).to(self.device)
        rewards = torch.tensor(rewards_np).to(self.device)

        # If this is the first training step, save the model's graph to TensorBoard for visualization purposes
        if self.global_training_steps == 1:
            try:
                with SummaryWriter(self.writer_path) as writer:
                    writer.add_graph(
                        self, (non_spatial_states, rgb_minimap_states))
            except Exception as e:
                print(f"Error logging model graph: {e}")

        # Using autocast for the forward pass for mixed-precision/FP16 performance improvements
        with autocast():
            # Compute the Q-values for the current states
            q_values = self(non_spatial_states, rgb_minimap_states)
            q_predict = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Compute the target Q-values
            # next_q_values = self(non_spatial_next_states,
            #                      rgb_minimap_next_states)
            next_q_values = target_network(
                non_spatial_next_states, rgb_minimap_next_states)
            max_next_q_values = next_q_values.max(1)[0]
            q_target = rewards + self.gamma * max_next_q_values

            # Compute the loss and update the model's weights
            loss = self.loss_fn(q_predict, q_target)

            # Debugs if required...
            #     print("Q-target contains NaN or Infinity!")
            # if torch.isnan(loss).any() or torch.isinf(loss).any():
            #     print("Loss contains NaN or Infinity!")
            # if torch.isnan(q_values).any() or torch.isinf(q_values).any():
            #     print("Q-values contain NaN or Infinity!")
            # if torch.isnan(non_spatial_states).any() or torch.isinf(non_spatial_states).any():
            #     print("Non-spatial states contain NaN or Infinity!")
            # if torch.isnan(rgb_minimap_states).any() or torch.isinf(rgb_minimap_states).any():
            #     print("RGB Minimap states contain NaN or Infinity!")
            # if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            #     print("Rewards contain NaN or Infinity!")
            # if math.isnan(self.gamma) or math.isinf(self.gamma):
            #     print("Gamma contains NaN or Infinity!")
            # if torch.isnan(actions).any() or torch.isinf(actions).any():
            #     print("Actions contain NaN or Infinity!")

        # Clear accumulated gradients before back propagation
        self.optimizer.zero_grad()

        # Backward pass with scaling
        self.scaler.scale(loss).backward()

        # Clip the gradient to avoid huge updates
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)

        # Update the model's weights with scaling
        self.scaler.step(self.optimizer)

        # Update the scale for next iteration
        self.scaler.update()

        # Update the learning rate based on CosineAnnealing scheduling
        self.scheduler.step()

        # These logs are generated less frequently (every 25 training runs)
        # Try/Except blocks as they've routinely crashed the simulation :(
        if self.global_training_steps % 25 == 0:
            # Logging various metrics for visualization and debugging
            try:
                with SummaryWriter(self.writer_path) as writer:
                    writer.add_scalar('Loss/train', loss.item(),
                                      self.global_training_steps)
            except Exception as e:
                print(f"Error logging Loss/train: {e}")

            try:
                with SummaryWriter(self.writer_path) as writer:
                    writer.add_scalar(
                        'Epsilon/value', self.epsilon, self.global_training_steps)
            except Exception as e:
                print(f"Error logging Epsilon/value: {e}")

            try:
                with SummaryWriter(self.writer_path) as writer:
                    writer.add_histogram(
                        'Q-Values', q_values.detach().cpu().numpy(), self.global_training_steps)
            except Exception as e:
                print(f"Error logging Q-Values: {e}")

            try:
                with SummaryWriter(self.writer_path) as writer:
                    writer.add_scalar(
                        'Learning Rate', self.optimizer.param_groups[0]['lr'], self.global_training_steps)
            except Exception as e:
                print(f"Error logging Learning Rate: {e}")

            # Log the histograms of model weights
            try:
                with SummaryWriter(self.writer_path) as writer:
                    for name, param in self.named_parameters():
                        writer.add_histogram(name, param.clone().cpu(
                        ).data.numpy(), self.global_training_steps)
            except Exception as e:
                print(f"Error logging model weights: {e}")

            # Create an action frequency histogram of the last 1000 actions in the replay buffer
            try:
                with SummaryWriter(self.writer_path) as writer:
                    last_1000_actions = list(itertools.islice(self.training_buffer, len(
                        self.training_buffer) - 1000, len(self.training_buffer)))
                    actions = [transition[1]
                               for transition in last_1000_actions]
                    action_frequencies = Counter(actions)

                    # Ensure all actions have an entry in the Counter (histogram wasn't rendering properly...)
                    for i in range(len(smart_actions)):
                        action_frequencies[i] = action_frequencies.get(i, 0)

                    action_list_for_histogram = [
                        action for action, freq in action_frequencies.items() for _ in range(freq)]

                    writer.add_histogram(
                        'Actions/Frequency', np.array(action_list_for_histogram), self.global_training_steps)
            except Exception as e:
                print(f"Error logging Actions/Frequency: {e}")

    def save_model(self, file_path, episode_count, reward):
        # Save our checkpoint weights
        save_path = f"{file_path}_episode_{episode_count}_reward_{reward:.2f}.pt"
        torch.save(self.state_dict(), save_path)

        # Also overwrite the top of tree so that we always have the latest to load if necessary:
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        print("Loading last checkpoint: ", file_path)
        self.load_state_dict(torch.load(file_path))

    # This function identifies all units of a specific desired type
    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    # Provide the x/y coordinates of our command center(s)

    def get_command_center_coordinates(self, obs):
        # Get Command Centers using the get_units_by_type function
        command_centers = self.get_units_by_type(
            obs, units.Terran.CommandCenter)

        # If there's a Command Center, return its coordinates
        if command_centers:
            # Grabbing the first for now (no expansion support for cameras)
            command_center = command_centers[0]

            # Checks to avoid out-of-bounds crashing
            # E.g ValueError: Argument is out of range for 2/select_point (6/select_point_act [4]; 0/screen [0, 0]), got: [[2], (-8, 15)]
            if 0 <= command_center.x < 84 and 0 <= command_center.y < 84:
                return command_center.x, command_center.y
            else:
                print(
                    f"Command Center coordinates out of range: ({command_center.x}, {command_center.y})")
                return None, None
        else:
            # print("Command Center not found.")
            return None, None

    # This function (tries...) to translate from Screen (84x84) -> Minimap (64x64)
    def translate_coordinates(self, x, y, original_size=84, target_size=64):
        scale_factor = target_size / original_size
        return int(x * scale_factor), int(y * scale_factor)

    # To ensure consistency across spawn locations, we invert the minimap locations based on base_top_left
    def transform_minimap(self, minimap_data, base_top_left):
        if not base_top_left:
            # Flip only the spatial dimensions, leaving the color channels unchanged
            # Copy required due to oddities in the return values..
            # PyTorch can't deal with negative strides AFAIK
            transformed_minimap = np.flip(minimap_data, axis=(0, 1)).copy()
        else:
            # Leave the minimap as it is
            transformed_minimap = minimap_data

        return transformed_minimap

    # This creates a fixed-length vector to store non-spatial data in (metrics and unit metadata)
    def transform_non_spatial_to_fixed_length(self, units, base_top_left, non_spatial_data, last_five_actions):

        # Safety checks
        # Check if units are None or empty
        if units is None or len(units) == 0:
            return np.zeros((300, 5))  # return a zero-filled array

        # Check for NaN values
        if np.isnan(np.array(units)).any():
            print("Warning: NaN values detected in units!")
            return np.zeros((300, 5))  # return a zero-filled array

        # Constant indices based on the field names
        UNIT_TYPE_INDEX = 0
        ALLIANCE_INDEX = 1
        HEALTH_INDEX = 2
        X_POS_INDEX = 12
        Y_POS_INDEX = 13

        # Number of units and features
        num_features = 5  # ['unit_type', 'health', 'x', 'y']

        # Create an empty array of shape (300, num_features)
        fixed_length_units = np.zeros((300, num_features))

        # Embed the non-spatial game state information into the first entries
        offset = 0
        if non_spatial_data:
            for key, value in non_spatial_data.items():
                # Input our metric in the first tuple with appropriate padding after
                fixed_length_units[offset] = [value, 0, 0, 0, 0]
                offset += 1
                if offset >= 18:  # Adjust for a maximum 16 non-spatial data entries
                    break

        # Embed the last_five_actions data into our vector
        # First, replace None elements with 0 and flatten the list
        actions_cleaned = [action or 0 for action in last_five_actions]
        # Then, pad the cleaned actions
        actions_padded = actions_cleaned + [0] * (5 - len(actions_cleaned))

        # Embed the flattened and padded actions
        fixed_length_units[offset, :5] = actions_padded
        offset += 1  # Increment the offset by 1, since we've added only one row

        # Then store the units information
        # Start after the reserved entries
        # Adjusted based on the reserved non-spatial entries
        for i in range(min(len(units), 300 - offset)):
            unit = units[i]
            index = i + offset  # Index into fixed_length_units
            fixed_length_units[index, 0] = unit[UNIT_TYPE_INDEX]
            fixed_length_units[index, 1] = unit[ALLIANCE_INDEX]
            fixed_length_units[index, 2] = unit[HEALTH_INDEX]

            # Transform x, y coordinates as required for normalization
            if not base_top_left:
                fixed_length_units[index, 3], fixed_length_units[index,
                                                                 4] = 64 - unit[X_POS_INDEX], 64 - unit[Y_POS_INDEX]
            else:
                fixed_length_units[index, 3], fixed_length_units[index,
                                                                 4] = unit[X_POS_INDEX], unit[Y_POS_INDEX]

        return fixed_length_units

# Agent Implementation


class DQNAgent(base_agent.BaseAgent):

    def __init__(self):
        super(DQNAgent, self).__init__()

        initial_actions = list(range(len(smart_actions)))
        print("Creating the model with the following attributes:")
        print("Available actions are set to:", initial_actions)
        print("State Size:", STATE_SIZE)

        # Our primary DQN
        self.dqn_model = DQNModel(
            actions=initial_actions, non_spatial_state_size=STATE_SIZE, minimap_shape=(3, 64, 64))
        # Our target network used for Q-Value Calculations
        self.target_network = DQNModel(
            actions=initial_actions, non_spatial_state_size=STATE_SIZE, minimap_shape=(3, 64, 64))

        self.previous_action = None
        self.previous_state = {
            "non_spatial": None,
            "rgb_minimap": None
        }

        self.cc_y = None
        self.cc_x = None

        self.move_number = 0

        # Minerals
        self.minerals = 0

        # Used for tracking rewards for use in model saving/checkpointing
        # Keep the last 100 rewards, initialize it to 0's to avoid overweighting early successes
        self.last_rewards = deque([0] * 100, maxlen=100)
        self.training_time = deque([0] * 100, maxlen=100)
        self.episode_count = 0
        self.previous_avg_reward = 0
        self.actual_root_level_steps_taken = 0
        self.in_game_training_iterations = 0

        # TensorBoard Per-Step Reward Metrics
        # Cumulative Counter for single game
        self.total_episode_reward = 0
        # A queue of the last 100 games (used for average tracking)
        self.total_episode_rewards = deque([0] * 100, maxlen=100)
        # List of each reward, displayed as a histogram (sampled every 25 games infrequently)
        self.episode_rewards = []

        # Custom Delay Timers
        self.attack_delay_timer = 0
        self.unit_attack_delay_timer = 0
        self.scv_delay_timer = 0
        self.camera_move_timer = 0
        self.last_camera_action = 0
        self.last_supply_depot_built_step = None

        # Queue that tracks last actions for exclusionary purposes
        # The queue is also fed as current_state info back into the non_spatial FCN
        self.last_five_actions = deque(maxlen=5)
        self.command_center = []

        # Action mapping
        # Define which actions correspond to which quadrants
        self.top_left_actions = [10, 11, 12, 13]
        self.top_right_actions = [14, 15, 16, 17]
        self.bottom_left_actions = [18, 19, 20, 21]
        self.bottom_right_actions = [22, 23, 24, 25]

        if os.path.isfile(_INITIAL_WEIGHTS):
            print("Loading previous model: ", _INITIAL_WEIGHTS)
            self.dqn_model.load_model(_INITIAL_WEIGHTS)
            self.target_network.load_model(_INITIAL_WEIGHTS)

        # Offload to GPU
        self.dqn_model = self.target_network.to(self.dqn_model.device)
        self.target_network = self.target_network.to(self.dqn_model.device)

    # We identify the current per-step reward based on in-game score and normalize it

    def get_normalized_reward(self, obs, previous_action, non_spatial_metrics, last_five_actions):
        # Extract the cumulative score from the observation
        score = obs.observation.score_cumulative.score
        # Anything above 20k score results in a full score being provided to the model
        max_score = 20000
        # Normalize the score to be between 0 and 1 (We use -1 to 1 later)
        normalized_score = min(score / max_score, 1)

        # --------------------
        # Action Mapping Reference from self.action_rewards{}
        # 0: 'donothing'
        # 1: 'buildsupplydepot'
        # 2: 'buildbarracks'
        # 3: 'buildmarine'
        # 4: 'attackunit'
        # 5: 'buildscv'
        # 6: 'resetcamera'
        # 7: 'mcselfexpansion'
        # 8: 'mcenemyexpansion'
        # 9: 'mcenemyprimary'
        # 10: 'attack_4_4'
        # 11: 'attack_12_4'
        # 12: 'attack_4_12'
        # 13: 'attack_12_12'
        # 14: 'attack_36_4'
        # 15: 'attack_44_4'
        # 16: 'attack_36_12'
        # 17: 'attack_44_12'
        # 18: 'attack_4_36'
        # 19: 'attack_12_36'
        # 20: 'attack_4_44'
        # 21: 'attack_12_44'
        # 22: 'attack_36_36'
        # 23: 'attack_44_36'
        # 24: 'attack_36_44'
        # 25: 'attack_44_44'
        # --------------------

        # Incentive multipliers for attack logic
        attack_opposite_quadrant = 0.5
        attack_adjacent_quadrant = 0.25
        penalty_home_quadrant = -1
        penalty_home_expansion = -0.5

        action_reward = 0

        # Incentivize attacks to opposite quadrant & their expansion
        # Also de-incentivize attacking home base & expansion
        if previous_action in self.bottom_right_actions:
            action_reward += attack_opposite_quadrant
        elif previous_action in self.bottom_left_actions:
            action_reward += attack_adjacent_quadrant
        elif previous_action in self.top_left_actions:
            action_reward += penalty_home_quadrant
        elif previous_action in self.top_right_actions:
            action_reward += penalty_home_expansion

        # Armies of 16 units and over will get the full reward
        # Trying to encourage attacking with a reasonably large group versus one at a time...
        max_army_size = 16
        # Calculate the reward accelerator based on army size. This should give a value between 0.15 and 1.
        army_size_accelerator = 0.15 + \
            (0.85 * min(max_army_size,
             non_spatial_metrics['army_count']) / max_army_size)

        # Include the predefined rewards for the specific action
        action_reward += self.dqn_model.action_rewards.get(previous_action, 0)

        # Apply the accelerator ONLY to the attack actions that are not penalties
        # 'attackunit' and 'attack_x_x' actions
        attack_actions_without_penalty = [4] + list(range(10, 26))
        non_penalty_attack_actions = list(set(
            attack_actions_without_penalty) - set(self.top_left_actions) - set(self.top_right_actions))
        # print("Non penalty actions are:" , non_penalty_attack_actions)
        if previous_action in non_penalty_attack_actions:
            action_reward *= army_size_accelerator
            # print("attack action-reward is: ", action_reward)

        # Calculate the total normalized reward by combining the normalized_score and action_reward
        # Clamp it between -1 and 1 as the LSTM performs better due to its use of tahn
        # We'll min/max this once again after further rewards below
        total_reward = min(max(normalized_score + action_reward, -1), 1)

        # print("Total reward is set to:", total_reward, "with the last action being: ", previous_action)

        # Check repetitiveness in the last_five_actions
        # Essentially, penalize reptition unless it's for building units (marines or SCVs)
        action_counts = Counter(last_five_actions)
        for action, count in action_counts.items():
            if action not in [3, 5] and count >= 3:
                total_reward = 0

        # Define a modifier based on the camera position
        camera_modifier = 1 if non_spatial_metrics['home_base_camera'] else 0.5
        # print("camera modifier is set to:", camera_modifier, "and last camera action is: ", non_spatial_metrics['camera_location']  )

        # Special Reward handling to ensure Marines are only being built when there is available build_queue
        if previous_action == 3 and not non_spatial_metrics['barracks_available_queue']:
            total_reward = 0
        # Special Reward Handling for supply depot construction and supply starvation
        elif previous_action == 1:
            if non_spatial_metrics['supply_free'] < 4:
                # Reward for building supply during starvation
                total_reward = camera_modifier * \
                    (1 if not non_spatial_metrics['supply_depot_recently_built'] else total_reward)
            elif non_spatial_metrics['supply_free'] > 16:
                # Reduced reward for building a supply depot if there's more than 16 extra supply
                total_reward = camera_modifier * -0.25
        # Ensure no reward for unit construction when supply is zero
        elif (previous_action == 3 or previous_action == 5) and non_spatial_metrics['supply_free'] <= 0:
            total_reward = 0
        # Reward Shaping for SCV construction (reduced after 16)
        elif previous_action == 5 and non_spatial_metrics['worker_supply'] > 16:
            over_scv = non_spatial_metrics['worker_supply'] - 16
            total_reward -= camera_modifier * 0.25 * over_scv
        # Reward Shaping for Barracks construction (reduced after 6)
        elif non_spatial_metrics['barracks_count'] > 6:
            over_barracks = non_spatial_metrics['barracks_count'] - 6
            total_reward -= camera_modifier * 0.25 * over_barracks

        # Penalize certain camera movements if barracks_count is less than 3
        # Agent should not waste cycles if we haven't built out the base yet
        if non_spatial_metrics['barracks_count'] < 3 and previous_action in [7, 8, 9]:
            total_reward = -1

        # Ensure that if the camera is moved, de-incentivize any camera action that is not 6 (return to home base)
        # 6 is home base and is excluded
        if non_spatial_metrics['camera_location'] in [7, 8, 9] and previous_action in [7, 8, 9]:
            total_reward = -1

        # Clamp the final total_reward between -1 and 1
        total_reward = max(min(total_reward, 1), -1)

        # Update our TensorBoard per-step Reward Metrics
        self.total_episode_reward += total_reward
        # print("self.total_episode_reward is set to:", self.total_episode_reward)
        # Update our Tensorboard per-Game Reward Histogram (if necessary - sampled every 3 games as it's expensive)
        self.episode_rewards.append(total_reward)

        # DEBUGS
        #
        # print("normalized reward is set to: ", total_reward, " and previous_action was: ", previous_action)
        # if total_reward < 1:
        # print("normalized reward is set to: ", total_reward, " and previous_action was: ", previous_action)
        #
        # Return the final total_reward
        return total_reward

    # BOILER PLATE CODE AGAIN

    def transformLocation(self, x, y):
        # Debug
        # print("Before transforming, x and y are set to: ", x , y)
        if not self.base_top_left:
            # Debug
            # print("After transformation, x and y are set tO: ", 64 - x, 64 - y)
            return [64 - x, 64 - y]

        return [x, y]

    # Custom code for transforming Screen instead of Minimap (references transformLocation of course)
    def transformLocationScreen(self, x, y):
        # Debug
        # print("Before transforming, x and y are set to: ", x , y)
        if not self.base_top_left:
            # Debug
            # print("After transformation, x and y are set tO: ", 84 - x, 84 - y)
            return [84 - x, 84 - y]

        return [x, y]

    # Return of boiler plate

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    # CUSTOM

    # This function checks to see if we can add more marines to our build queue
    # Used to improve our reward system for the agent
    def has_room_in_build_queue(self, obs):
        for unit in obs.observation.feature_units:
            if unit.unit_type == _TERRAN_BARRACKS:
                # Get the length of the build_queue tensor.
                build_queue_length = len(obs.observation.build_queue)
                # Check if there's room in the build queue.
                if build_queue_length < 5:  # Assuming 5 is the max length.
                    return True
        # print("build queue is: ", obs.observation.build_queue)
        return False

    # A simple normalize function
    def normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def step(self, obs):
        super(DQNAgent, self).step(obs)
        # Using delay timers to avoid duplicate commands being issued by the AI
        self.attack_delay_timer += 1
        self.unit_attack_delay_timer += 1
        self.scv_delay_timer += 1
        self.camera_move_timer += 1

        # Visuals
        rgb_minimap = obs.observation["rgb_minimap"]

        # Check our current score, just for debugging
        # print("Current score is: ", self.get_normalized_reward(obs))

        # If this is our last step
        if obs.last():
            self.episode_count += 1
            base_reward = obs.reward  # This is a ternary system - -1, 0, 1
            episode_steps = obs.observation.game_loop[0]

            # Apply step-based reward only if the agent won, encouraging the agent to find efficient victories
            # Reward decreases the longer it takes after 10,000 game steps, becoming 0 after 20,000 steps
            if base_reward == 1:  # 1 indicates a win
                extra_steps = max(0, episode_steps - 10000)
                # print("Extra steps are: ", extra_steps)
                # print("total sets were: ", episode_steps)
                step_penalty = extra_steps // 1000 * 0.1
                step_reward = 1.0 - step_penalty
                combined_reward = base_reward + step_reward
                # Ensure combined_reward never goes below 1.1 for a win
                combined_reward = max(combined_reward, 1.2)
                final_reward_multiplier = combined_reward
            elif base_reward == 0:  # 0 indicates a draw
                combined_reward = base_reward
                final_reward_multiplier = 0.7  # Slight decrease in score for a draw
            else:  # -1 indicates a loss
                combined_reward = base_reward
                final_reward_multiplier = 0.25

            self.last_rewards.append(combined_reward)  # Add the latest reward
            # Calculate the rolling average reward
            avg_reward = sum(self.last_rewards) / len(self.last_rewards)

            print("------------------------------------------------------")
            # print("Combined reward is set to: ", combined_reward)

            # Optimal reward is '2' (perfect string of wins at 10K steps or less)
            # Checkpoint the model every 100 games once we've started training
            if self.episode_count % 100 == 0 and len(self.dqn_model.training_buffer) > self.dqn_model.training_buffer_requirement:
                print("Saving our model weights (Checkpoint)...")
                self.dqn_model.save_model(
                    _DATA_FILE, self.episode_count, avg_reward)

            print("Previous average reward was: ",
                  self.previous_avg_reward)
            print("Our rolling-average reward is: ", avg_reward)
            print("Latest game reward was: ", combined_reward)
            print("Number of steps were: ", episode_steps)
            # Backpropagate the final reward multiplier to previous actions
            print(
                f"Backpropagating a {final_reward_multiplier}x reward multiplier across {self.actual_root_level_steps_taken} root level actions...")
            self.dqn_model.backpropagate_final_reward(
                final_reward_multiplier, self.actual_root_level_steps_taken)

            # Copy over our deque replay buffer into our list if it's been 10 games
            if self.episode_count % 10 == 0:
                self.dqn_model.transfer_buffer()

            # Print statements if our buffer is large enough to train on...
            if len(self.dqn_model.training_buffer) > self.dqn_model.training_buffer_requirement:
                print("Number of in-game model updates: ",
                      self.in_game_training_iterations)
                print("Training the model after game completion...")

            # Is our buffer large enough to begin training...
            if len(self.dqn_model.training_buffer) > self.dqn_model.training_buffer_requirement:
                training_start_time = time.time()
                self.dqn_model.learn(self.target_network)
                training_end_time = time.time() - training_start_time
                self.training_time.append(training_end_time)
                avg_training_time = sum(
                    self.training_time) / len(self.training_time)
                # Update our target DQN network every 5 games once training has begun
                if self.episode_count % 5 == 0:
                    self.target_network.load_state_dict(
                        self.dqn_model.state_dict())
                print(
                    f"This training loop took {training_end_time:.4f} seconds.")
                print("Training complete")
                # Log our average reward to TensorBoard
                with SummaryWriter(self.dqn_model.writer_path) as writer:
                    writer.add_scalar(
                        'Average Reward/value', avg_reward, self.dqn_model.global_training_steps)
                # Log training time
                with SummaryWriter(self.dqn_model.writer_path) as writer:
                    writer.add_scalar('Average Training Time in Seconds/train_duration',
                                      avg_training_time, self.dqn_model.global_training_steps)

                # Update our TensorBoard per-step Reward Metrics
                # Takes an average of the last 100 games (queue length = 100)
                self.total_episode_rewards.append(self.total_episode_reward)
                average_total_episode_rewards = sum(
                    self.total_episode_rewards) / len(self.total_episode_rewards)
                with SummaryWriter(self.dqn_model.writer_path) as writer:
                    writer.add_scalar(
                        'Total Per-Step Shaped Reward/value', average_total_episode_rewards, self.dqn_model.global_training_steps)
                # Update our reward histogram every 3 games (sampled due to expense)
                # In try/except block as these routinely fail/crash simulation
                # Update our reward histogram every 3 games (sampled due to expense)
                if self.episode_count % 3 == 0:
                    try:
                        with SummaryWriter(self.dqn_model.writer_path) as writer:
                            # This will log the distribution of rewards per game step
                            writer.add_histogram(
                                'Per-Step Rewards', np.array(self.episode_rewards), self.dqn_model.global_training_steps)
                    except Exception as e:
                        print(f"Error logging Reward Histogram: {e}")

            # Reset remaining, episode-specific counters
            print("------------------------------------------------------")
            self.previous_avg_reward = avg_reward
            self.previous_action = None
            self.previous_state = None
            self.move_number = 0
            self.actual_root_level_steps_taken = 0
            self.in_game_training_iterations = 0
            supply_depot_recently_built = False
            self.last_supply_depot_built_step = None
            # Clear our action queue
            self.last_five_actions.clear()
            self.last_camera_action = 0
            for _ in range(5):
                self.last_five_actions.append(None)
            # TensorBoard Per-Step Reward Metrics
            # Resetting Cumulative Counter for single game
            self.total_episode_reward = 0
            # Resetting List of each reward, displayed as a histogram (sampled every 25 games infrequently)
            self.episode_rewards = []

            # There is a predictable SC2 Segfault after every 986th game currently, this code is a workaround
            if (self.episode_count % _MAX_GAMES_BEFORE_RESTART == 0) and (_TRAINING_TYPE != "bot"):
                print(
                    "----Restarting underlying SC2 Environment for Crash Avoidance!!----")
                # Raising a custom exception to get caught in our main PySC2 run_loop to trigger restart
                # This took...a lot of trial and error :(
                raise CustomRestartException(
                    "Time to restart the environment!")

            return actions.FunctionCall(_NO_OP, [])

        # BOILER PLATE Action-Space Guardrails
        # Used as a way of limiting the potential action space at the beginning of the game for the agent

        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]

        if obs.first():

            # Original logic, doesn't work properly...
            # player_y, player_x = (
            #     obs.observation["feature_screen"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            # self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            # # print("Player x and y are set to: ", self.player_x, self.player_y)

            self.cc_y, self.cc_x = (
                unit_type == _TERRAN_COMMANDCENTER).nonzero()

            # Using similar approach to Kane AI to Figure out where our home base is
            # Original reference is here of course:
            # https://raw.githubusercontent.com/skjb/pysc2-tutorial/master/Build%20a%20Zerg%20Bot/zerg_agent_step7.py
            if obs.first():

                player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                      features.PlayerRelative.SELF).nonzero()
                xmean = player_x.mean()
                ymean = player_y.mean()

                if xmean <= 31 and ymean <= 31:
                    self.base_top_left = True
                else:
                    self.base_top_left = False

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        # Original barracks_count and supply_depot_count from reference code had flaws when y was shared
        all_supply_depots = self.dqn_model.get_units_by_type(
            obs, units.Terran.SupplyDepot)
        supply_depot_count = len(
            [unit for unit in all_supply_depots if unit.alliance == features.PlayerRelative.SELF])

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        # Original barracks_count and supply_depot_count from reference code had flaws when y was shared
        # barracks_count = int(round(len(barracks_y) / 137))
        all_barracks = self.dqn_model.get_units_by_type(
            obs, units.Terran.Barracks)
        barracks_count = len(
            [unit for unit in all_barracks if unit.alliance == features.PlayerRelative.SELF])

        all_command_centers = self.dqn_model.get_units_by_type(
            obs, units.Terran.CommandCenter)
        command_center_count = len(
            [unit for unit in all_command_centers if unit.alliance == features.PlayerRelative.SELF])

        # Find our command centers:
        # command_center = self.dqn_model.get_units_by_type(obs, units.Terran.CommandCenter)
        # friendly_command_centers = [unit for unit in command_centers if unit.alliance == features.PlayerRelative.SELF]

        # If we haven't set our global CC yet, add it here
        if not self.command_center:
            self.command_center = self.dqn_model.get_units_by_type(
                obs, units.Terran.CommandCenter)

        # if self.command_center:
        #     print("Command centers are set to:", self.command_center)

        # Minerals
        self.minerals = obs.observation['player'][1]
        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]
        idle_worker_count = obs.observation['player'][7]
        army_count = obs.observation['player'][8]

        # unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.ENEMY]

        enemy_units = [
            unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.ENEMY]

        visible_units = [unit for unit in obs.observation.feature_units]
        # print("Visible Units are set to: ", visible_units)
        # print("But transformed, they are: ", self.dqn_model.transform_units_to_fixed_length(visible_units))

        supply_free = supply_limit - supply_used

        # Checking to see if we have available build queue for agent state
        barracks_available_build_queue = self.has_room_in_build_queue(obs)

        # We check to see if we need to do any reward shaping to avoid supply starvation...
        supply_depot_recently_built = False
        # print("Game step is: ", self.actual_root_level_steps_taken)

        if self.last_supply_depot_built_step is not None:
            if (self.actual_root_level_steps_taken - self.last_supply_depot_built_step) <= 60:
                # print("supply_depot_recently_built = True" )
                supply_depot_recently_built = True
            # else:
                # print("supply_depot_recently_built = False" )

        non_spatial_metrics = {
            'minerals': self.minerals,
            'supply_limit': supply_limit,
            'supply_used': supply_used,
            'supply_free': supply_free,
            'army_supply': army_supply,
            'worker_supply': worker_supply,
            'cc_count': command_center_count,
            'barracks_count': barracks_count,
            'supply_depot_count': supply_depot_count,
            'idle_worker_count': idle_worker_count,
            'army_count': army_count,
            'home_base_camera': self.last_camera_action == 6,
            'barracks_available_queue': barracks_available_build_queue,
            'supply_depot_recently_built': supply_depot_recently_built,
            'camera_location': self.last_camera_action,
            'game_step_progress': self.normalize(self.actual_root_level_steps_taken, 1, 1500),
            'game_score': self.normalize(obs.observation.score_cumulative.score, 1, 12000),
            # Prevents NaN from being sent to the model, breaking things
            'previous_action': self.previous_action or 0
        }

        # print("Non-spatial metrics are set to: ", non_spatial_metrics)
        # print("Non-spatial barracks count is: ", non_spatial_metrics["barracks_count"])

        current_state = {
            "non_spatial": np.zeros(300),
            "rgb_minimap": None
        }
        current_state["non_spatial"] = self.dqn_model.transform_non_spatial_to_fixed_length(
            visible_units, self.base_top_left, non_spatial_metrics, self.last_five_actions)
        current_state["rgb_minimap"] = self.dqn_model.transform_minimap(
            rgb_minimap, self.base_top_left)

        # print("The output of our spatial data is set to: ", )
        # for i in range(30):
        #     five_tuple = current_state["non_spatial"][i]
        #     print(five_tuple)

        ''' Current State should look like this when passed to the model:
            Non-spatial metrics are set to:  {'minerals': 105, 'supply_limit': 15, 'supply_used': 12, 'supply_free': 3, 'army_supply': 0, 'worker_supply': 12, 'cc_count': 1, 'barracks_count': 0, 'supply_depot_count': 0, 'idle_worker_count': 0, 'army_count': 0, 'home_base_camera': True, 'camera_location': 6, 'barracks_available_queue': False, 'supply_depot_recently_built': False, 'game_step_progress': 0.0066711140760507, 'game_score': 0.0920076673056088, 'previous_action': 0}
            The output of our spatial data is set to: 
            [105.   0.   0.   0.   0.]
            [15.  0.  0.  0.  0.]
            [12.  0.  0.  0.  0.]
            [3. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]
            [12.  0.  0.  0.  0.]
            [1. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]
            [1. 0. 0. 0. 0.]
            [6. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]
            [0.00667111 0.         0.         0.         0.        ]
            [0.09200767 0.         0.         0.         0.        ]
            [0. 0. 0. 0. 0.]
            [0. 0. 6. 0. 0.]
            [45.  1. 45. 17. 15.]
            [45.  1. 45.  8. 23.]
            [4.83e+02 3.00e+00 1.00e+04 7.00e+00 1.20e+01]
            [3.42e+02 3.00e+00 1.00e+04 3.30e+01 8.00e+00]
            [1.8e+01 1.0e+00 1.5e+03 2.2e+01 3.3e+01]
            [3.41e+02 3.00e+00 1.00e+04 1.70e+01 1.20e+01]
            [4.83e+02 3.00e+00 1.00e+04 2.10e+01 8.00e+00]
            [45.  1. 45. 12. 27.]
        '''

        # Push s/a/r/s_next our replay buffer
        # Per-Step rewards are calculated using the get_normalized_reward function - see it for details
        if self.previous_action is not None:
            self.actual_root_level_steps_taken += 1
            # print("Pushing to the replay buffer: ", self.previous_action, self.get_normalized_reward(obs, self.previous_action, non_spatial_metrics))
            self.dqn_model.store_transition(self.previous_state,
                                            self.previous_action, self.get_normalized_reward(obs, self.previous_action, non_spatial_metrics, self.last_five_actions), current_state)

        # Do in-game training of the model for every 100 root actions the agent takes
        if self.actual_root_level_steps_taken % 100 == 0 and len(self.dqn_model.training_buffer) > self.dqn_model.training_buffer_requirement:
            # print("Beginning in-game training for the model.")
            self.in_game_training_iterations += 1
            self.dqn_model.learn(self.target_network)

        # Simple intent state tracker
        if not hasattr(self, 'intended_action'):
            self.intended_action = None

        if self.move_number == 0:
            self.move_number += 1

            # this is where we store arbitrary actions which the agent is not allowed to take this game step
            excluded_actions = []
            # print("excluded_actions at the start are set to: ", excluded_actions)

            # --------------------
            # Action Mapping Reference
            # 0: 'donothing'
            # 1: 'buildsupplydepot'
            # 2: 'buildbarracks'
            # 3: 'buildmarine'
            # 4: 'attackunit'
            # 5: 'buildscv'
            # 6: 'resetcamera'
            # 7: 'mcselfexpansion'
            # 8: 'mcenemyexpansion'
            # 9: 'mcenemyprimary'
            # 10: 'attack_4_4'
            # 11: 'attack_12_4'
            # 12: 'attack_4_12'
            # 13: 'attack_12_12'
            # 14: 'attack_36_4'
            # 15: 'attack_44_4'
            # 16: 'attack_36_12'
            # 17: 'attack_44_12'
            # 18: 'attack_4_36'
            # 19: 'attack_12_36'
            # 20: 'attack_4_44'
            # 21: 'attack_12_44'
            # 22: 'attack_36_36'
            # 23: 'attack_44_36'
            # 24: 'attack_36_44'
            # 25: 'attack_44_44'
            # --------------------

            # Modified, self-generated code to scale supply depot creation
            # We guard supply depot builds by camera location - need to make sure we're at home base before building
            if supply_free > 4 or self.last_camera_action != 6 or self.minerals < 100:
                excluded_actions.append(1)

            # We guard barracks builds by camera location - need to make sure we're at home base before building
            # Also check to see that we have enough minerals
            if barracks_count > 5 or self.last_camera_action != 6 or self.minerals < 150:
                excluded_actions.append(2)

            # Exclude marines from the build queue
            if supply_free == 0 or barracks_count < 1 or self.minerals < 50:
                # print("Marine Build Excluded. Supply is set to: ", supply_free," and barracks_count is: ", barracks_count)
                excluded_actions.append(3)

            # CUSTOM
            # If we don't see an enemy or if we've issued an attack order 2 steps ago, skip...
            # print("Before our conditional check on attacking, enemy unit length is: ", len(enemy_units), "our timer is: ", self.unit_attack_delay_timer, " and our supply is: ", army_supply)
            # print("This evalutes to: ", (len(enemy_units) == 0) or (self.unit_attack_delay_timer < 2) or (army_supply < 8))
            if (len(enemy_units) == 0) or (army_supply < 10):
                # print("excluding attack units")
                excluded_actions.append(4)

            # SCV Checks
            if worker_supply > 15 or self.scv_delay_timer < 7 or self.last_camera_action != 6 or self.minerals < 50 or supply_free < 4:
                excluded_actions.append(5)

            # Camera reset handling
            # print("Truthyness check: ", (self.camera_reset_required <= 0 or self.last_camera_action == 6) and self.camera_move_timer > 4)
            # print("self.last_camera_action is set to: ", self.last_camera_action)
            if self.last_camera_action == 6 or self.camera_move_timer < 5:
                excluded_actions.append(6)

            # Camera move handling
            if self.camera_move_timer < 12 or barracks_count < 4 or self.last_camera_action != 6:
                # print("Not moving camera, timer is: ", self.camera_move_timer, "and army_supply is: ", army_supply)
                excluded_actions.append(7)
                excluded_actions.append(8)
                excluded_actions.append(9)

            # modified original logic, waits for 12 marines before attacking
            # Excludes all minimap attack actions if we just issued one (at least...for now)
            # Post-bootstrap, it may be possible to relax these checks
            # Additionally, makes sure there are no visible units (action 4) that we can attack directly...
            if 4 not in excluded_actions or army_supply < 12 or self.attack_delay_timer < 5:
                # Add actions from 10-25 to excluded_actions if they are not present
                excluded_actions.extend(x for x in range(
                    10, 26) if x not in excluded_actions)

            # This section of the code aims to prevent the bot from repeatedly taking the same action over and over again,
            # which can result in an undesirable behavior like doing nothing for an extended period.

            # If the last five actions taken by the bot are identical...
            if len(set(self.last_five_actions)) == 1:

                # Retrieve the action that was repeated
                action_to_exclude = self.last_five_actions[0]

                # Check if this action isn't already in the list of excluded actions
                # and also ensure we aren't excluding too many actions, which could lead to potential crashes
                # or undesirable behaviors if no actions are left to choose from.
                if action_to_exclude not in excluded_actions and len(excluded_actions) < 25:
                    excluded_actions.append(action_to_exclude)

            # If our Guardrails are disabled, allow any action to be chosen
            if _TRAINING_GUARDRAILS_ENABLED == False:
                excluded_actions.clear()

            # DQN Selects the action
            rl_action = self.dqn_model.choose_action(
                current_state, excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action

            # Set our last camera action:
            if rl_action in [6, 7, 8, 9]:  # One of the camera actions
                self.last_camera_action = rl_action

            # Add the action to our tiny 5-element queue
            self.last_five_actions.append(rl_action)

            # DEBUG : Print all exclusions!
            # print("Our excluded actions for this step are: ", excluded_actions)
            # print("The model chose: ", rl_action)

            # using reference code for smart action implementation
            smart_action, x, y = self.splitAction(self.previous_action)

            # Quick reset for first-step as bots have issues with camera state
            if obs.first():
                # Set our smart action to our base
                smart_action == ACTION_RESET_CAMERA

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    # Checks to avoid out-of-bounds crashing
                    # E.g ValueError: Argument is out of range for 2/select_point (6/select_point_act [4]; 0/screen [0, 0]), got: [[2], (-8, 15)]
                    if 0 <= target[0] < 84 and 0 <= target[1] < 84:
                        # Update Intent State Tracking
                        if smart_action == ACTION_BUILD_BARRACKS:
                            self.intended_action = ACTION_BUILD_BARRACKS
                        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                            self.intended_action = ACTION_BUILD_SUPPLY_DEPOT
                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                    else:
                        print(f"SCV coordinates out of range: {target}")

            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    self.intended_action = ACTION_BUILD_MARINE
                    # Checks to avoid out-of-bounds crashing
                    # E.g ValueError: Argument is out of range for 2/select_point (6/select_point_act [4]; 0/screen [0, 0]), got: [[2], (-8, 15)]
                    if 0 <= target[0] < 84 and 0 <= target[1] < 84:
                        return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
                    else:
                        print(f"Barracks coordinates out of range: {target}")

            elif smart_action == ACTION_ATTACK or smart_action == ACTION_ATTACK_UNIT:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

            # CUSTOM SCV Build Code
            elif smart_action == ACTION_BUILD_SCV:
                # print("base is top left: ", self.base_top_left)
                safe_cc_x, safe_cc_y = self.dqn_model.get_command_center_coordinates(
                    obs)
                # print("New target should be: ", safe_cc_x, safe_cc_y)

                if safe_cc_x:
                    target = safe_cc_x, safe_cc_y

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            # To Do:
            # If get_command_center_coordinates returns None, None
            # This means we've probably lost our command center and need to build a new one
            # New action required
            # Alternative -> Try and repair it if it's taking damage

            # Custom camera reset code
            # AI can only act on units on its screen (unless using hotkeys/mappings...)
            # This avoids annoying stalls after moving the camera to attack a specific unit on the Screen
            # Coordinate flipping isn't ideal for camera movement unfortunately...
            elif smart_action == ACTION_RESET_CAMERA:
                # print("Camera counter is at: ",
                #       self.camera_reset_required, "resetting camera")
                self.camera_move_timer = 0

                if self.base_top_left:
                    # print("Spawned top left - moving camera")
                    return actions.FUNCTIONS.move_camera((22, 18))
                else:
                    # print("Spawned bottom right - moving camera")
                    return actions.FUNCTIONS.move_camera((43, 51))

            # Camera move logic - need to move the camera to 'see' enemy units to attack them directly
            # Self expansion
            # Enemy Primary
            # Enemy expansion

            # Move camera to our expansion
            elif smart_action == ACTION_MOVE_CAMERA_SELF_EXPANSION:
                # reset our counter so we go back to home base eventually
                self.camera_move_timer = 0

                if self.base_top_left:
                    # Top right quadrant center
                    return actions.FUNCTIONS.move_camera((43, 18))
                else:
                    # Bottom left quadrant center
                    return actions.FUNCTIONS.move_camera((22, 51))

            # Move camera to enemy's expansion
            elif smart_action == ACTION_MOVE_CAMERA_ENEMY_EXPANSION:
                # reset our counter so we go back to home base eventually
                self.camera_move_timer = 0
                if self.base_top_left:
                    # Bottom left quadrant center
                    return actions.FUNCTIONS.move_camera((22, 51))
                else:
                    # Top right quadrant center
                    return actions.FUNCTIONS.move_camera((43, 18))

            # Move camera to enemy's primary base
            elif smart_action == ACTION_MOVE_CAMERA_ENEMY_PRIMARY:
                # reset our counter so we go back to home base eventually
                self.camera_move_timer = 0
                if self.base_top_left:
                    # Bottom right quadrant center
                    return actions.FUNCTIONS.move_camera((43, 51))
                else:
                    # Top left quadrant center
                    return actions.FUNCTIONS.move_camera((22, 18))

        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = self.splitAction(self.previous_action)

            # end of boiler plate

            # Custom code / initial design similar to boiler plate

            # Used to ensure buildings are placed on the border, resulting in trapped units
            BORDER_PADDING = 6

            if self.intended_action == ACTION_BUILD_SUPPLY_DEPOT and not _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                # print("Resetting back to square one")
                self.move_number = 0

            elif self.intended_action == ACTION_BUILD_SUPPLY_DEPOT and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                self.last_supply_depot_built_step = self.actual_root_level_steps_taken
                if self.cc_y.any():
                    x_padding = random.randint(-30, 30)
                    y_padding = random.randint(-30, 30)
                    target_x = round(self.cc_x.mean()) - 35 + x_padding
                    target_y = round(self.cc_y.mean()) + y_padding

                    target = self.transformLocation(target_x, target_y)

                    # Ensure the coordinates are within valid bounds after transformation
                    target[0] = max(BORDER_PADDING, min(
                        target[0], 83 - BORDER_PADDING))
                    target[1] = max(BORDER_PADDING, min(
                        target[1], 83 - BORDER_PADDING))

                    # Start our timer for reward shaping

                    # print("Trying to build a supply depot at:", target)
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

            elif self.intended_action == ACTION_BUILD_BARRACKS and not _BUILD_BARRACKS in obs.observation['available_actions']:
                # print("Resetting back to square one")
                self.move_number = 0
                # for action in obs.observation.available_actions:
                #     print(actions.FUNCTIONS[action])

            elif self.intended_action == ACTION_BUILD_BARRACKS and _BUILD_BARRACKS in obs.observation['available_actions']:
                # print("Obs actions are:", obs.observation['available_actions'])
                # print("Inside BUILD_BARRACKS")
                if self.cc_y.any():
                    # print("Inside SCV Check")
                    x_padding = random.randint(-35, 35)
                    y_padding = random.randint(-35, 35)
                    target_x = round(self.cc_x.mean()) - 35 + x_padding
                    target_y = round(self.cc_y.mean()) + y_padding

                    target = self.transformLocation(target_x, target_y)

                    # Ensure the coordinates are within valid bounds after transformation
                    # Assuming screen size is 84x84
                    target[0] = max(BORDER_PADDING, min(
                        target[0], 83 - BORDER_PADDING))
                    target[1] = max(BORDER_PADDING, min(
                        target[1], 83 - BORDER_PADDING))

                    # print("Trying to build barracks at: ", target)

                    # print("Trying to build a barracks at:", target)
                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

            # CUSTOM SCV Build Logiic
            elif smart_action == ACTION_BUILD_SCV:
                # print("SCV Smart Action Set")
                # Zero out our build timer
                self.scv_delay_timer = 0
                if _TRAIN_SCV in obs.observation['available_actions']:
                    # print("Trying to train an SCV")

                    return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])

            # start of boiler plate code (small modifications like delay timers)
            elif self.intended_action == ACTION_BUILD_MARINE and not _TRAIN_MARINE in obs.observation['available_actions']:
                # print("Resetting back to square one")
                self.move_number = 0

            elif self.intended_action == ACTION_BUILD_MARINE and _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action == ACTION_ATTACK:
                do_it = True

                if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:

                    # Debugs
                    # print("Our base is top left: ", self.base_top_left)
                    # print("Our attack minimap location is: ", self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8)))
                    # print("Attacking: ", self.transformLocation(int(x), int(y)))
                    # Zero out our attack delay timer so we don't issue this repetitively (will wait 100 steps)
                    self.attack_delay_timer = 0

                    # return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x), int(y))])

            # Custom implementation to attack random enemy unit
            elif smart_action == ACTION_ATTACK_UNIT:
                if enemy_units and actions.FUNCTIONS.Attack_screen.id in obs.observation["available_actions"]:
                    # Select a random enemy unit
                    target_unit = random.choice(enemy_units)

                    # Check if the target unit is within the current view
                    if 0 <= target_unit.x < 84 and 0 <= target_unit.y < 84:
                        # Issue the attack command using screen coordinates
                        # print("Attacking enemy unit: ", target_unit, " at: ", self.transformLocationScreen(target_unit.x, target_unit.y), " with original coordinates: ", target_unit.x, target_unit.y)
                        return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, self.transformLocationScreen(target_unit.x, target_unit.y)])
                    else:
                        # Clamp the target camera coordinates to a valid range
                        target_x = max(0, min(target_unit.x, 83))
                        target_y = max(0, min(target_unit.y, 83))

                        # Move the camera to the clamped coordinates
                        # print(
                        #     "Moving camera to fix out-of-bounds issue for attack_screen: ", target_x, target_y)

                        # Reset our timers to 0
                        self.unit_attack_delay_timer = 0
                        self.attack_delay_timer = 0
                        self.last_camera_action = 0

                        return actions.FUNCTIONS.move_camera((target_x, target_y))

            self.intended_action = None

        elif self.move_number == 2:
            self.move_number = 0

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (
                        unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])

        return actions.FunctionCall(_NO_OP, [])

def main(args):
    agent1 = DQNAgent()
    agent1_name = "Abel_AI"

    # Depending on the training type, create agent2
    # Self-Reinforcement Learning
    if _TRAINING_TYPE == "self":
        agent2 = test_DQN_Agent()
        agent2_name = "Abel_AI"
    elif _TRAINING_TYPE == "kane":
        agent2 = KaneAI()
        agent2_name = "Kane_AI"
    elif _TRAINING_TYPE == "bot":
        agent2 = sc2_env.Bot(sc2_env.Race.terran,
                             sc2_env.Difficulty[_BOT_DIFFICULTY])
        agent2_name = f"{_BOT_DIFFICULTY}_bot"
    else:
        raise ValueError(f"Unknown TRAINING_TYPE: {_TRAINING_TYPE}")

    USE_FEATURE_UNITS = True
    RGB_SCREEN_SIZE = 84
    RGB_MINIMAP_SIZE = 64

    while True:
        try:
            with sc2_env.SC2Env(
                map_name="Simple64",
                players=[
                    sc2_env.Agent(sc2_env.Race.terran, name=agent1_name),
                    sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty[_BOT_DIFFICULTY]) if _TRAINING_TYPE == "bot" else sc2_env.Agent(
                        sc2_env.Race.terran, name=agent2_name),
                ],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RGB,
                    use_feature_units=USE_FEATURE_UNITS,
                    rgb_dimensions=features.Dimensions(
                        screen=RGB_SCREEN_SIZE, minimap=RGB_MINIMAP_SIZE),
                    feature_dimensions=features.Dimensions(
                        screen=84, minimap=64)
                ),
                step_mul=16,
                game_steps_per_episode=21000,
                visualize=False
            ) as env:
                run_loop.run_loop([agent1, agent2], env)

        except CustomRestartException:
            print("Caught restart signal. Restarting environment...")
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Attempting to restart the SC2 environment.")
            # The full SC2 environment will be recreated at the start of the next iteration of the main loop.
            continue

if __name__ == "__main__":
    app.run(main)
