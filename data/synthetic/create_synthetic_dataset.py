import pandas as pd
import os
import numpy as np
import featuretools as ft
from woodwork.logical_types import Categorical, Double, Integer

seed = 0
out_path = 'raw_data'
os.makedirs(out_path, exist_ok=True)

np.random.seed(seed=seed)

df_a = pd.DataFrame({
    'aId': np.arange(1000),
    'cId': np.arange(1000),
    'bId': np.random.randint(0, 5000, size=(1000,)),
    'data1': np.ones(1000, dtype=int),
})
df_b = pd.DataFrame({
    'bId': np.arange(5000),
    'cId': np.random.randint(0, 1000, size=(5000)),
    'data1': np.random.randint(11, size=(5000,)),
})
df_c = pd.DataFrame({
    'cId': np.arange(1000),
    'data1': np.ones(1000),
})

df_a.to_csv(os.path.join(out_path, 'A.csv'), index=False)
df_b.to_csv(os.path.join(out_path, 'B.csv'), index=False)
df_c.to_csv(os.path.join(out_path, 'C.csv'), index=False)

