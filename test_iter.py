import pandas as pd
import numpy as np
import time

n = 3_000_000
s = pd.Series(np.random.randint(0, 100, n), index=[f"uuid-{i}" for i in range(n)])

d = {}
t0 = time.time()
for k, v in s.items():
    if k not in d or v < d[k]:
        d[k] = v
print(f"Time taken: {time.time()-t0:.2f}s")
