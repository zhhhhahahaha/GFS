import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils import construct_graph

out_path = 'processed_data/GFS'
os.makedirs(out_path, exist_ok=True)
tablelist = {}

df_a = pd.read_csv('raw_data/A.csv', index_col=False)
df_b = pd.read_csv('raw_data/B.csv', index_col=False)
df_c = pd.read_csv('raw_data/C.csv', index_col=False)
label = pd.read_csv('processed_data/DFS1/label.csv', index_col=False)
df_a = pd.concat([df_a, label], axis=1)
df_a.rename(columns={'label': 'TARGET'}, inplace=True)

tablelist['a'] = df_a
tablelist['b'] = df_b
tablelist['c'] = df_c

construct_graph('synthetic', tablelist, out_path, dummy_table=False)