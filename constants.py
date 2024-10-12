from pathlib import Path


import torch


DIR_PATH = str(Path().resolve())
RUNTIME = 99999999  # maximum runtime in seconds for running model (training or testing)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATE_FORMAT = "%m-%d %H:%M:%S"
