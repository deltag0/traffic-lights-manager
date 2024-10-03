from pathlib import Path
from dataclasses import dataclass

import gymnasium as gym
import torch
import sumo_rl  # sumo_rl will be needed to setup some environments


from constants import DIR_PATH, RUNTIME
print(DIR_PATH + "SUMO_NET")

@dataclass
class Four_way:
    """
    The display variable should be initialized by the user to choose
    if the gui should appear or not.
    The Four_way class aggreagates all necessary information for the
    environment:
    - the gym environment,
    - the number of lanes
    - the max_occupancy in the lanes
    """
    display: bool = False
    env = four_way_intersection = gym.make('sumo-rl-v0', net_file=DIR_PATH + "/sumo_nets/4_way_intersection/4_way_map.net.xml",
                route_file=DIR_PATH + "/sumo_nets/4_way_intersection/4_way_route.rou.xml", single_agent=True,
                use_gui=display, num_seconds=RUNTIME)
    lanes = 4
    max_occupancy = 19
