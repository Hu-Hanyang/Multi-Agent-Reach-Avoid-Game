import numpy as np

center = np.array([2.0, 6.0])
r = 0.8
current = np.array([2.0, 1.0])
heuristic = (np.linalg.norm(current - center)  - r)/ 0.8
print(heuristic)