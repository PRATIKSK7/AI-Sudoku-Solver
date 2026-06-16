import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ai.solver import solve

bo = [
    [0]*9,
    [0]*9,
    [0]*9,
    [0]*9,
    [0]*9,
    [0]*9,
    [0]*9,
    [0]*9,
    [0]*9
]
bo[0][0] = 1
t0 = time.time()
print("Starting solve...")
res = solve(bo, limit=2)
print("Finished in", time.time() - t0, "seconds, solutions:", res)
