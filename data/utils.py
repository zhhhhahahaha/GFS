import category_encoders as ce
import pandas as pd
import numpy as np
import os
import time
from typing import Dict
from config import encode_config, graph_config

# the encoded number start with 1 (OrdinalEncoder start with 1)
def data_encoder(dname, tablelist: Dict[str, pd.DataFrame], is_align=False, dummy_table=False):
    encoder = {}
    mk = {}
    align = {}
    align_num = 0
    # encode primary key and aggregate multiple key
    for table, tableinfo in encode_config[dname].items():
        df = tablelist[table]
        if tableinfo['PK'] != None:
            enc = ce.OrdinalEncoder(cols=tableinfo['PK'])
            df[tableinfo['PK']] = enc.fit_transform(df[tableinfo['PK']])
            encoder[tableinfo['PK']] = enc
            # primary key will not be used for embedding so we do not need to align it
        if tableinfo['MK'] != None:
            for col in tableinfo['MK']:
                if col not in mk:
                    mk[col] = df[col]
                else:
                    mk[col] = mk[col].append(df[col])

    #add encoder of multiple key
    for key, value in mk.items():
        enc = ce.OrdinalEncoder(cols=key)
        enc = enc.fit(value)
        encoder[key] = enc
        if is_align:
            align[key] = align_num
            align_num += enc.category_mapping[0]['mapping'].max()
    
    # encode other columns
    for table, tableinfo in encode_config[dname].items():
        df = tablelist[table]
        for col, colcont in df.items():
            # we do not to deal with continuous feature
            if col == tableinfo['PK']:
                continue
            elif col in encoder:
                enc = encoder[col]
                df[col] = enc.transform(df[col])
                df[col] = df[col].astype(int)
                if is_align and col in align: # if this column is used as primary in other table, we do not need to align it
                    anum = align[col]
                    df[col] = df[col] + anum
                    unkown_idx = df[df[col]==(anum-1)].index.tolist()
                    missing_idx = df[df[col]==(anum-2)].index.tolist()
                    df.loc[unkown_idx, col] = -1
                    df.loc[missing_idx, col] = -2
            elif col in tableinfo['Date']:
                time_data = df[col].values
                time_data = [time.strptime(i, "%Y-%m-%d") for i in time_data]
                df.drop(col, axis=1, inplace=True)

                year_name = col+'_year'
                df.insert(df.shape[1], year_name, [i.tm_year for i in time_data])
                df[year_name] = df[year_name] - df[year_name].min() + 1 
                num = df[year_name].max()
                df[year_name] += align_num
                align_num += num

                month_name = col+'_month'
                df.insert(df.shape[1], month_name, [i.tm_mon for i in time_data])
                df[month_name] += align_num
                align_num += 12

                day_name = col+'_day'
                df.insert(df.shape[1], day_name, [i.tm_mday for i in time_data])
                df[day_name] += align_num
                align_num += 31

                wday_name = col+'_wday'
                df.insert(df.shape[1], wday_name, [i.tm_wday for i in time_data])
                df[wday_name] += (align_num + 1)
                align_num += 7
            elif colcont.dtype == 'object':
                enc = ce.OrdinalEncoder(cols=col)
                df[col] = enc.fit_transform(df[col])
                df[col] = df[col].astype(int)
                if is_align:
                    anum = align_num
                    df[col] = df[col] + anum
                    unkown_idx = df[df[col]==(anum-1)].index.tolist()
                    missing_idx = df[df[col]==(anum-2)].index.tolist()
                    df.loc[unkown_idx, col] = -1
                    df.loc[missing_idx, col] = -2
                    align_num += enc.category_mapping[0]['mapping'].max()
    if is_align:
        print(f'total number of categorical values is: {align_num}')
    return tablelist

def construct_graph(dname, tablelist: Dict[str, pd.DataFrame], path, dummy_table=False):
    if dummy_table:
        schema = graph_config[dname][1]
    else:
        schema = graph_config[dname][0]
    # construct node file
    for node, nodeinfo in schema.items():
        if nodeinfo['is_table'] == True:
            df = tablelist[node]
            node_df = pd.DataFrame({'node_id': np.arange(df.shape[0])})

            if nodeinfo['Cat']!= None:
                node_df['Cat'] =  df[nodeinfo['Cat']].apply(lambda x: f'{",".join(map(str, x))}', axis=1)
            if nodeinfo['Con']!= None:
                node_df['Con'] =  df[nodeinfo['Con']].apply(lambda x: f'{",".join(map(str, x))}', axis=1)
            if 'TARGET' in df.columns:
                node_df['label'] = df['TARGET']
            
            node_df.to_csv(os.path.join(path, f"{node}.csv"), index=False)
        else: # dummy table, need to be figure out later
            node_val = set()
            for table in nodeinfo['exist']:
                node_val.update(tablelist[table][node])
            node_val = np.array(list(node_val))
            node_df = pd.DataFrame({
                'node_id': np.arange(node_val.shape[0]),
                node: node_val, # for this type of node, feature name is the same with the node name
            })
            table_df = pd.DataFrame({
                node: node_val
            })
            tablelist[node] = table_df
            node_df.to_csv(os.path.join(path, f"{node}.csv"), index=False)
    
    # construct edge file
    for node in schema:
        tablelist[node]['_row_id_'] = np.arange(tablelist[node].shape[0])
    for node, nodeinfo in schema.items():
        if nodeinfo['is_table'] == True and nodeinfo['FK'] != None:
            for fktable, fkcolumn in nodeinfo['FK'].items():
                src_view = tablelist[node][['_row_id_', fkcolumn]]
                src_view = src_view.rename(columns={'_row_id_': 'src_id'})
                # drop src value == -1 or -2
                mask = (src_view[fkcolumn] != -1) & (src_view[fkcolumn] != -2)
                src_view = src_view[mask]

                dst_view = tablelist[fktable][[fkcolumn, '_row_id_']]
                dst_view = dst_view.rename(columns={'_row_id_': 'dst_id'})

                # attach dst_id from dst_view with the same fkcolmn value
                edge_df = pd.merge(src_view, dst_view, on=fkcolumn, how='left')
                edge_df = edge_df.drop(fkcolumn, axis=1)

                edge_df.to_csv(os.path.join(path, f"{node}_{fktable}.csv"), index=False)

                