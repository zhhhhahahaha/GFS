import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import argparse
from utils import data_encoder, construct_graph

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--data_path", default="./raw_data", help="Path to save the data")
parser.add_argument("--out_path", default="./processed_data", help="Path to save the data")

args = parser.parse_args()

np.random.seed(args.seed)
tablelist = {}
dname = 'outbrain-full'

# load data
clicks = pd.read_csv(os.path.join(args.data_path, 'clicks_train.csv'))
events = pd.read_csv(os.path.join(args.data_path, 'events.csv'))
page_views = pd.read_csv(os.path.join(args.data_path, 'page_views_sample.csv'))
ad_content = pd.read_csv(os.path.join(args.data_path, 'promoted_content.csv'))
doc_meta = pd.read_csv(os.path.join(args.data_path, 'documents_meta.csv'))
doc_ent = pd.read_csv(os.path.join(args.data_path, 'documents_entities.csv'))
doc_topics = pd.read_csv(os.path.join(args.data_path, 'documents_topics.csv'))
doc_cat = pd.read_csv(os.path.join(args.data_path, 'documents_categories.csv'))

out_path = os.path.join(args.out_path, 'GFS')
os.makedirs(out_path, exist_ok=True)

# data preprocess
## group by sample on display
display_id = clicks['display_id'].drop_duplicates().reset_index(drop=True)
display_id = display_id.sample(frac=0.01, random_state=0).reset_index(drop=True)

clicks = clicks.merge(display_id, on='display_id', how='inner')
click_map = {'f0': 0, 'f1': 1}
clicks['clicked'] = clicks['clicked'].replace(click_map)

ad_id = clicks['ad_id'].drop_duplicates()

events = events.merge(display_id, how='inner', on='display_id')
events['timestamp'] = pd.to_datetime(events['timestamp']+1465876799998, unit='ms').dt.strftime('%Y-%m-%d')

user = pd.DataFrame(events['uuid'].drop_duplicates().reset_index(drop=True))
page_views = page_views.merge(user, how='inner', on='uuid')
page_views['timestamp'] = pd.to_datetime(page_views['timestamp']+1465876799998, unit='ms').dt.strftime('%Y-%m-%d')

ad_content = ad_content.merge(ad_id, how='inner', on='ad_id')

doc_meta.drop(['publish_time'], axis=1, inplace=True)

clicks.rename(columns={'clicked': 'TARGET'}, inplace=True)

# encode data
tablelist['clicks'] = clicks
tablelist['events'] = events
tablelist['user'] = user
tablelist['page_views'] = page_views
tablelist['ad_content'] = ad_content
tablelist['doc_meta'] = doc_meta
tablelist['doc_ent'] = doc_ent
tablelist['doc_topics'] = doc_topics
tablelist['doc_cat'] = doc_cat
tablelist = data_encoder(dname, tablelist, is_align=True, dummy_table=False)
print('Successfully encode the data')

# build graph
construct_graph(dname, tablelist, out_path, dummy_table=False)