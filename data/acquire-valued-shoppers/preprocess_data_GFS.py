import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils import construct_graph, data_encoder
import argparse

def get_latest_transactions(df, num):
    return df.iloc[-num:]

parser = argparse.ArgumentParser()
parser.add_argument('--sample', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--data_path", default="./raw_data", help="Path to save the data")
parser.add_argument("--out_path", default="./processed_data", help="Path to save the data")

args = parser.parse_args()

np.random.seed(args.seed)
tablelist = {}
dname = 'acquire-valued-shoppers'

# load data
History = pd.read_csv(os.path.join(args.data_path, 'trainHistory.csv'), index_col=False)
offers = pd.read_csv(os.path.join(args.data_path, 'offers.csv'), index_col=False)
transactions = pd.read_csv(os.path.join(args.data_path, 'transactions.csv'), index_col=False)

if args.sample:
    History = History.sample(frac=0.1).reset_index(drop=True)
    transactions = transactions.merge(History['id'], how='inner', on='id')
    out_path = os.path.join(args.out_path, 'GFS_sample')
else:
    transactions = transactions.merge(History['id'], how='inner', on='id')
    out_path = os.path.join(args.out_path, 'GFS')

os.makedirs(out_path, exist_ok=True)

#data preprocess
History.drop('repeattrips', axis=1, inplace=True)
mapping = {'t': 1, 'f': 0}
History['repeater'] = History['repeater'].replace(mapping)
History.rename(columns={'repeater': 'TARGET'}, inplace=True)

# get latest transactions
transactions['date'] = pd.to_datetime(transactions['date'], format='%Y-%m-%d')
transactions = transactions.groupby('id', sort=False).apply(get_latest_transactions, 50).reset_index(drop=True)
transactions['date'] = transactions['date'].dt.strftime('%Y-%m-%d')

#encode data
tablelist['History'] = History
tablelist['offers'] = offers
tablelist['transactions'] = transactions
data_encoder(dname, tablelist, is_align=True, dummy_table=False)
print('Successfully encode the data')

# build graph
construct_graph(dname, tablelist, out_path, dummy_table=False)

