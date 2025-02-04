import numpy as np
import time
data = [{"x": 1, "y": 2}, {"x": 3, "z": 4}, {"a": 5, "x": 6}, {"b": 7}]
start = time.time()
x_values = [d["x"] for d in data if "x" in d]
print(x_values, "Required Time: ", time.time()-start)  # Output: [1, 3, 6]
data = np.array(data)
x_values = np.array([d.get("x") for d in data if "x" in d])
print(x_values, "Required Time: ", time.time()-start)  # Output: [1, 3, 6]

