import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils import construct_graph, data_encoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--data_path", default="./raw_data", help="Path to save the data")
parser.add_argument("--out_path", default="./processed_data", help="Path to save the data")

args = parser.parse_args()

np.random.seed(args.seed)
tablelist = {}
dname = 'kdd15'

# load data
course = pd.read_csv(os.path.join(args.data_path, 'date.csv'), index_col=False)
enrollment = pd.read_csv(os.path.join(args.data_path, 'enrollment_train.csv'), index_col=False)
log = pd.read_csv(os.path.join(args.data_path, 'log_train.csv'), index_col=False)
object = pd.read_csv(os.path.join(args.data_path, 'object.csv'), index_col=False)
label = pd.read_csv(os.path.join(args.data_path, 'truth_train.csv'), index_col=False, header=None)

out_path = os.path.join(args.out_path, 'GFS')
os.makedirs(out_path, exist_ok=True)

# data preprocess
label.rename(columns={0:'enrollment_id', 1:'TARGET'}, inplace=True)
enrollment = enrollment.merge(label, on='enrollment_id', how='left')

log.rename(columns={'object':'module_id'}, inplace=True)
tmp_time = log['time'].apply(lambda x: x[:10])
log['time'] = tmp_time
object.drop('start', axis=1, inplace=True)

# encode data
tablelist['course'] = course
tablelist['enrollment'] = enrollment
tablelist['log'] = log
tablelist['object'] = object
tablelist = data_encoder(dname, tablelist, is_align=True, dummy_table=False)
print('Successfully encode the data')

# build graph
construct_graph(dname, tablelist, out_path, dummy_table=False)
