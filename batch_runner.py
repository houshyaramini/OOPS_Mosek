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
        use_crra=False,
        use_seed=True,
        gamma=15,
        n_wiederholungen=2000,
        big_array=True,

        bounds=[(0.0 , 1.0)] * 8
        )
    try:
        run_simulation(config=run_config)
    except Exception as e:
        print(f"Fehler in Run {i+1}: {e}")
        continue

print("\nAlle Simulationen abgeschlossen!")