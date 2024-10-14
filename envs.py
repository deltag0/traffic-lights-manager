import os
from pathlib import Path
from dataclasses import dataclass

import gymnasium as gym
import torch
from sumo_rl import SumoEnvironment  # sumo_rl will be needed to setup some environments


from constants import DIR_PATH, RUNTIME


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

    env = SumoEnvironment(
        net_file=os.path.join(DIR_PATH, "sumo_nets", "4_way_intersection", "4_way_map.net.xml"),
        route_file=os.path.join(DIR_PATH, "sumo_nets", "4_way_intersection", "4_way_route.rou.xml"),
        num_seconds=RUNTIME,
        use_gui=display
    )
    lanes = 4
    max_occupancy = 19
