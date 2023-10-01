import os
import pandas as pd
import numpy as np
import sys
import argparse
sys.path.append('..')
from utils import data_encoder, construct_graph

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--data_path", default="./raw_data", help="Path to save the data")
parser.add_argument("--out_path", default="./processed_data", help="Path to save the data")

args = parser.parse_args()

np.random.seed(args.seed)
tablelist = {}
dname = 'diginetica'

# load data
product_categories = pd.read_csv(os.path.join(args.data_path, 'product-categories.csv'), index_col=False)
products = pd.read_csv(os.path.join(args.data_path, 'products.csv'), index_col=False)
clicks = pd.read_csv(os.path.join(args.data_path, 'train-clicks.csv'), index_col=False)
item_views = pd.read_csv(os.path.join(args.data_path, 'train-item-views.csv'), index_col=False)
purchases = pd.read_csv(os.path.join(args.data_path, 'train-purchases.csv'), index_col=False)
queries_meta = pd.read_csv(os.path.join(args.data_path, 'train-queries-meta.csv'), index_col=False)
queries_result = pd.read_csv(os.path.join(args.data_path, 'train-queries-result.csv'), index_col=False)

out_path = os.path.join(args.out_path, 'GFS')
os.makedirs(out_path, exist_ok=True)

# data preprocess
products.drop('product_name_tokens', axis=1, inplace=True)
products = products.merge(product_categories, how='inner', on='itemId')

clicks['label'] = np.ones(clicks.shape[0])
clicks.drop('timeframe', axis=1, inplace=True)
clicks.drop_duplicates(subset=['queryId', 'itemId'], keep='first', inplace=True)
clicks.reset_index(drop=True, inplace=True)
queries_has_clicks = clicks['queryId'].drop_duplicates()
queries_has_clicks = queries_has_clicks.sample(frac=0.01, random_state=0).reset_index(drop=True)
clicks = clicks.merge(queries_has_clicks, how='inner', on='queryId')

item_views.drop('userId', axis=1, inplace=True)
purchases.drop('userId', axis=1, inplace=True)

queries_meta = queries_meta[queries_meta['is_test']==False]
queries_meta.drop(['userId', 'searchstring_tokens', 'is_test'], axis=1, inplace=True)
queries_meta = queries_meta.merge(queries_has_clicks, how='inner', on='queryId')

queries_result.rename(columns={'items':'itemId'}, inplace=True)
queries_result = queries_result.merge(queries_has_clicks, how='inner', on='queryId')

clicks = queries_result.merge(clicks, how='outer', on=['queryId', 'itemId'])
clicks = clicks.fillna(0)
clicks['label'] = clicks['label'].astype(int)

session = pd.merge(item_views['sessionId'].drop_duplicates(), purchases['sessionId'].drop_duplicates(), how='outer')

clicks.rename(columns={'label':'TARGET'}, inplace=True)

# encode data
tablelist['clicks'] = clicks
tablelist['queries_meta'] = queries_meta
tablelist['products'] = products
tablelist['session'] = session
tablelist['item_views'] = item_views
tablelist['purchases'] = purchases
tablelist = data_encoder(dname, tablelist, is_align=True, dummy_table=False)
print('Successfully encode the data')

# build graph
construct_graph(dname, tablelist, out_path, dummy_table=False)