import os
from pathlib import Path

import gymnasium as gym
import torch
from sumo_rl import SumoEnvironment  # sumo_rl will be needed to setup some environments


from constants import DIR_PATH, RUNTIME


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
    def __init__(self, display: bool = False, lanes: int = 4, max_occupancy: int = 10):
        self.display = display

        self.env = SumoEnvironment(
            net_file=os.path.join(DIR_PATH, "sumo_nets", "4_way_intersection", "4_way_map.net.xml"),
            route_file=os.path.join(DIR_PATH, "sumo_nets", "4_way_intersection", "4_way_route.rou.xml"),
            num_seconds=RUNTIME,
            use_gui=display,
            sumo_warnings=False
        )
        self.lanes = lanes
        self.max_occupancy = max_occupancy


class FourxFour:
    """
    The FourxFour class aggregates all necessary information for the 4x4 grid environment:
    - the SUMO environment,
    - the number of lanes,
    - the maximum occupancy per lane.
    The `display` variable is user-initialized to control GUI display.
    """
    def __init__(self, display: bool = False, lanes: int = 4, max_occupancy: int = 10):
        self.display = display

        self.env = SumoEnvironment(
            net_file=os.path.join(DIR_PATH, "sumo_nets", "4x4_Grid", "4x4.net.xml"),
            route_file=os.path.join(DIR_PATH, "sumo_nets", "4x4_Grid", "4x4.rou.xml"),
            num_seconds=RUNTIME,
            use_gui=display,
            sumo_warnings=False
        )
        self.lanes = lanes
        self.max_occupancy = max_occupancy


class TwoxTwo:
    """
    The TwoxTwo class aggregates all necessary information for the 2x2 grid environment:
    - the SUMO environment,
    - the number of lanes,
    - the maximum occupancy per lane.
    The `display` variable is user-initialized to control GUI display.
    """
    def __init__(self, display: bool = False, lanes: int = 4, max_occupancy: int = 10):
        self.display = display

        self.env = SumoEnvironment(
            net_file=os.path.join(DIR_PATH, "sumo_nets", "2x2_Grid", "2x2.net.xml"),
            route_file=os.path.join(DIR_PATH, "sumo_nets", "2x2_Grid", "2x2.rou.xml"),
            num_seconds=RUNTIME,
            use_gui=display,
            sumo_warnings=False
        )
        self.lanes = lanes
        self.max_occupancy = max_occupancy
