import pandas as pd
import numpy as np
import cvxpy as cp
import os
import warnings
import mosek
from typing import Union
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(filename):
    return os.path.join(BASE_DIR, "data", filename)

def get_log_path(filename):
    return os.path.join(BASE_DIR, "results", filename)

@dataclass(frozen=True)
class SimulationConfig:

    file_spx: str = field(default_factory=lambda: get_data_path("SpxDaten.csv"))
    file_options: str = field(default_factory=lambda: get_data_path("BloombergOptionsDatenNeu.csv"))
    file_risk_free: str = field(default_factory=lambda: get_data_path("DSG1MO_fred.csv"))
    log_file: str = field(default_factory=lambda: get_log_path("simulation_log.csv"))
    
    n_wiederholungen: int = 3000
    use_seed: bool = True
    seed: int = 999
    worst_case: bool = False
    gamma: float = 40.0
    use_crra: bool = False
    big_array: bool = False
    n_assets: int = 8  
    eps: float = 1e-3
    
    d_window: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 30, 60])
    
    
    long_idx: List[int] = field(default_factory=lambda: [0, 2, 4, 6])
    short_idx: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    pair_idx: List[List[int]] = field(default_factory=lambda: [[0, 1], [2, 3], [4, 5], [6, 7]])
    bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 1.0)] * 8)
    w0: Optional[List[float]] = None

