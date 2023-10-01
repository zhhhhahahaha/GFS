import os
import pandas as pd
import numpy as np

df = pd.read_csv('processed_data/DFS1/C.csv', index_col=False)
true_indice = np.where(df['c.SUM(b.data1)'] >= 25)[0]
label = pd.DataFrame({
    'label': np.zeros(df.shape[0], dtype=int)
})
label.iloc[true_indice] = 1
label.to_csv(os.path.join('processed_data', 'DFS1', 'label.csv'), index=False)
label.to_csv(os.path.join('processed_data', 'DFS2', 'label.csv'), index=False)
