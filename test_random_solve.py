import time
import sys
import os
import random

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ai.solver import solve

# Generate a random board that is likely unsolvable
bo = [[0 for _ in range(9)] for _ in range(9)]
random.seed(42)

for _ in range(25):
    r, c = random.randint(0, 8), random.randint(0, 8)
    v = random.randint(1, 9)
    # just place it
    bo[r][c] = v

print("Starting solve of random board...")
t0 = time.time()
res = solve(bo, limit=2)
print("Finished in", time.time() - t0, "seconds, solutions:", res)
