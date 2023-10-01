import dgl
import numpy as np
import os
import torch

from dgl import AddReverse
from sklearn.preprocessing import MinMaxScaler


def load_data(
        dname, 
        tarn, 
        batch_size,
        graph_config, 
        use_PNA=True, 
        num_workers=8, 
        path='../data', 
        seed=99, 
        depth=1, 
        embedding_size=16,
        embed_len=1):
    data_path = os.path.join(path, dname, 'processed_data', 'GFS')

    dataset = dgl.data.CSVDataset(data_path)
    g = dataset[0]
    total_num = g.nodes[tarn].data['label'].shape[0]

    # generate train/valid/test indice
    if dname == 'outbrain-full':
        k = int(total_num/40)
    elif dname == 'acquire-valued-shoppers':
        k = int(total_num/5)
    elif dname == 'diginetica':
        k = int(total_num/45)
    elif dname == 'home-credit':
        k = int(total_num/15)
    elif dname == 'kdd15':
        k = int(total_num/5)
    elif dname == 'synthetic':
        k = int(total_num/5)
    elif dname == 'synthetic2':
        k = int(total_num/5)
    
    indice = np.zeros((total_num, 1))
    indice[:k] = 1
    indice[k:2*k] = 2
    np.random.seed(seed)
    indice = np.random.permutation(indice)
    train_indice = np.where(indice==0)[0]
    valid_indice = np.where(indice==2)[0]
    test_indice = np.where(indice==1)[0]

    # split a training graph (exclude all the nodes relate to test nodes)
    valid_g = split_train_graph(g, tarn, test_indice)
    exclude_indice = np.sort(np.concatenate((valid_indice, test_indice)))
    train_g = split_train_graph(g, tarn, exclude_indice)

    # get new valid indice
    valid_indice = torch.tensor(valid_indice)
    invmap = torch.zeros(total_num, dtype=torch.int64)
    invmap[valid_g.nodes[tarn].data[dgl.NID]] = torch.arange(valid_g.nodes[tarn].data[dgl.NID].shape[0])
    new_valid_indice = invmap[valid_indice]

    # transform to undirected graph
    transform = AddReverse()
    g = transform(g)
    valid_g = transform(valid_g)
    train_g = transform(train_g)

    train_g = remove_isolated_nodes(train_g)

    # normalize continuous data
    scaler = MinMaxScaler()
    for nodes, value in train_g.ndata['Con'].items():
        if len(value.shape) == 1:
            value = value.reshape(-1, 1)
        train_g.nodes[nodes].data['Con'] = torch.tensor(scaler.fit_transform(value), dtype=torch.float32)
        if len(g.nodes[nodes].data['Con'].shape) == 1:
            g.nodes[nodes].data['Con'] = g.nodes[nodes].data['Con'].reshape(-1, 1)
        g.nodes[nodes].data['Con'] = torch.tensor(scaler.transform(g.nodes[nodes].data['Con']), dtype=torch.float32)
        if len(valid_g.nodes[nodes].data['Con'].shape) == 1:
            valid_g.nodes[nodes].data['Con'] = valid_g.nodes[nodes].data['Con'].reshape(-1, 1)
        valid_g.nodes[nodes].data['Con'] = torch.tensor(scaler.transform(valid_g.nodes[nodes].data['Con']), dtype=torch.float32)

    # initialize the nodes embedding (row embedding)
    # remark: In the later(after initialization in forward), the ['h'] dimensions are num_nodes, row_embed_num, embedding_size, 
    # but it does not matter here
    # initialize here to deal with the situation that some nodes do not have its own features
    for nodes in train_g.ntypes:
        g.nodes[nodes].data['h'] = torch.zeros(g.num_nodes(nodes), embed_len, embedding_size)
        valid_g.nodes[nodes].data['h'] = torch.zeros(valid_g.num_nodes(nodes), embed_len, embedding_size)
        train_g.nodes[nodes].data['h'] = torch.zeros(train_g.num_nodes(nodes), embed_len, embedding_size)

    # generate graph structure
    meta_g = g.metagraph()
    meta_nodes = {}
    meta_nodes = construct_meta(meta_g, tarn, depth, meta_nodes, g)
    canonical_etypes = g.canonical_etypes

    if use_PNA:
        cal_PNA_scaler(train_g, meta_nodes, graph_config)

    # load data
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(depth)
    train_loader = dgl.dataloading.DataLoader(
        train_g,
        {tarn: torch.arange(train_g.num_nodes(tarn))},
        sampler,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = dgl.dataloading.DataLoader(
        valid_g,
        {tarn: new_valid_indice},
        sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = dgl.dataloading.DataLoader(
        g,
        {tarn: test_indice},
        sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, valid_loader, test_loader, meta_nodes, canonical_etypes


def split_train_graph(g: dgl.DGLHeteroGraph, tarn, test_indice, max_depth=10):
    # we need to exclude all the test nodes and test nodes' ancestors
    exclude_sg, _ = dgl.khop_in_subgraph(g, {tarn: test_indice}, k=max_depth)
    node_id_dict = exclude_sg.ndata[dgl.NID]

    node_mask = {node_name: torch.ones(g.num_nodes(node_name), dtype=torch.bool) for node_name in g.ntypes}
    for node_name, indices in node_id_dict.items():
        node_idx = indices
        node_mask[node_name][node_idx] = False
    
    sg = dgl.node_subgraph(g, node_mask, relabel_nodes=True)

    return sg

# This function is only for the setting that we have made the directed graph to undirected graph
def remove_isolated_nodes(g: dgl.DGLGraph):
    meta_g = g.metagraph()
    for ntype in g.ntypes:
        neighbors = list(meta_g.adj[ntype].keys()) # srcnode: ntype, dstnode: neighbor
        degree = torch.zeros(g.num_nodes(ntype))
        for neighbor in neighbors:
            etype = list(meta_g.get_edge_data(ntype, neighbor).keys())[0]
            degree += g.out_degrees(torch.arange(g.num_nodes(ntype)), etype=etype)
        remove_idx = torch.where(degree==0)[0]
        g = dgl.remove_nodes(g, remove_idx, ntype=ntype)

    return g


def cal_PNA_scaler(train_g: dgl.DGLHeteroGraph, meta_nodes, graph_config):
    for nodes_name in meta_nodes.keys():
        scalers = {}
        if meta_nodes[nodes_name]['neighbor_agg'] == None:
            continue
        for neighbor in meta_nodes[nodes_name]['neighbor_agg']:
            if graph_config[nodes_name]['FK'] != None and neighbor in graph_config[nodes_name]['FK'].keys():
                continue
            else:
                etype = list(train_g.metagraph().get_edge_data(neighbor, nodes_name).keys())[0]
                degrees = train_g.in_degrees(torch.arange(train_g.num_nodes(nodes_name)), etype=etype)
                scaler = torch.log(degrees+1).sum()/degrees.shape[0]
                scalers[neighbor] = scaler
        meta_nodes[nodes_name]['avg_d'] = scalers


def construct_meta(meta_g, tarn, total_depth, meta_nodes, g):
    # BFS traversal
    q = []
    q.append((tarn, total_depth))

    # we do not use the embedding of target node for final prediction, we only use target nodes' embedding
    # to join to other table, meta_nodes records the information used for forming row embedding, so that
    # we need to record the information of target nodes second times we visit it
    tarn_visit_times = 0 
    while len(q) != 0:
        nodes_name, depth = q.pop(0)

        if nodes_name == tarn:
            if tarn_visit_times == 2:
                continue
        else:
            if nodes_name in meta_nodes:
                continue

        self_domain_con_num = 0
        self_domain_cat_num = 0
        if 'Con' in g.nodes[nodes_name].data:
            if len(g.nodes[nodes_name].data['Con'].shape) == 1:
                self_domain_con_num = 1
            else: 
                self_domain_con_num = g.nodes[nodes_name].data['Con'].shape[1]
        if 'Cat' in g.nodes[nodes_name].data:
            if len(g.nodes[nodes_name].data['Cat'].shape) == 1:
                self_domain_cat_num = 1
            else:
                self_domain_cat_num = g.nodes[nodes_name].data['Cat'].shape[1]

        if depth == 0:
            # target nodes must need aggregation function, but it may not need information from other table to form row embedding
            # we explain it above on why we need has_traverse_tarn
            if nodes_name == tarn:
                if tarn_visit_times == 0:
                    tarn_visit_times += 1
                    # just initialize meta_nodes[tarn], to avoid the key error when the tarn will not be visit twice
                    meta_nodes[nodes_name] = {
                        'num_other_domain': 0,
                        'num_con_domain': self_domain_con_num,
                        'num_cat_domain': self_domain_cat_num,
                        'neighbor_agg': list(meta_g.adj[nodes_name].keys()),
                    }
                else:
                    tarn_visit_times += 1
            else:
                meta_nodes[nodes_name] = {
                    'num_other_domain': 0, # this record is used for forming row embedding, not for the final prediction
                    'num_con_domain': self_domain_con_num,
                    'num_cat_domain': self_domain_cat_num,
                    'neighbor_agg': None,
                }
        else:
            for neighbor_name in meta_g.adj[nodes_name].keys():
                q.append((neighbor_name, depth-1))
            if nodes_name == tarn:
                if tarn_visit_times == 0:
                    tarn_visit_times += 1
                    # just initialize meta_nodes[tarn], to avoid the key error when the tarn will not be visit twice
                    meta_nodes[nodes_name] = {
                        'num_other_domain': 0,
                        'num_con_domain': self_domain_con_num,
                        'num_cat_domain': self_domain_cat_num,
                        'neighbor_agg': list(meta_g.adj[nodes_name].keys()),
                    }
                else:
                    tarn_visit_times += 1
                    meta_nodes[nodes_name]['num_other_domain'] = len(meta_g.adj[nodes_name].keys())
            else:
                meta_nodes[nodes_name] = {
                    'num_other_domain': len(meta_g.adj[nodes_name].keys()), # this record is used for forming row embedding, not for the final prediction
                    'num_con_domain': self_domain_con_num,
                    'num_cat_domain': self_domain_cat_num,
                    'neighbor_agg': list(meta_g.adj[nodes_name].keys()) # maybe not all neighbor_agg need aggregate function, can just simply join
                }
    return meta_nodes