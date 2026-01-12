import numpy as np
import time
from modules.config import SimulationConfig
from main import run_simulation  

N_RUNS = 1
START_SEED = 252
GAMMA = 10

print(f"Starte Batch-Run mit {N_RUNS} Durchläufen...")
for i in range(N_RUNS):
    current_seed = START_SEED + i
    current_gamma = GAMMA + i 
    print(f"\n--- Starte Run {i+1} von {N_RUNS}  ---")
    
    run_config = SimulationConfig(
        use_crra=True,
        #use_seed=True,
        #seed=START_SEED,
        gamma=current_gamma,
        big_array=True,
        n_wiederholungen=1000,
        bounds=[(0.0 , 0.1)] * 8,
        #worst_case=True,
        #pair_idx=False
        )
    try:
        run_simulation(config=run_config)
    except Exception as e:
        print(f"Fehler in Run {i+1}: {e}")
        continue

print("\nAlle Simulationen abgeschlossen!")