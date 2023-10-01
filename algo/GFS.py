import torch
import torch.nn as nn
import dgl
import sys
import dgl.function as fn

sys.path.append('..')
from algo.GFS_Module import RowEmb_mod, Con_mod, Agg_func
from GFS.algo.Predictor import DeepFM, Transformer
from collections import OrderedDict
from dgl.heterograph import DGLBlock

EPS = 1e-5

class GFS(nn.Module):
    def __init__(
            self,
            tarn,
            meta_nodes,
            graph_config,
            canonical_etypes,
            num_feat,
            embedding_size,
            embed_mode,
            embed_dropout,
            agg_mode,
            agg_dropout,
            aggregators,
            scalers,
            p_hidden,
            p_dropout,
            predictor,
            # fttransformer hyperparameter
            num_blocks=3,
            p_num_blocks=3,
            num_heads=8,
            ):
        super(GFS, self).__init__()
        self.predictor_name = predictor
        self.meta_nodes = meta_nodes
        # the number of embeddings for each nodes(row)
        self.embed_num = len(embed_mode)
        self.tarn = tarn
        self.aggregators = aggregators
        self.graph_config = graph_config
        self.embedding_size = embedding_size
        
        # raw categorical feature embedding
        self.w = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(self.w.weight.data)

        # continuous feature embedding
        self.con = nn.ModuleDict()
        for nodes_name in meta_nodes.keys():
            if meta_nodes[nodes_name]['num_con_domain'] > 0:
                self.con[nodes_name] = Con_mod(
                    num_cols=meta_nodes[nodes_name]['num_con_domain'],
                    embedding_size=embedding_size,
                )

        # nodes(row) embedding function
        self.node_embed = nn.ModuleDict()

        for nodes_name in meta_nodes.keys():
            num_fields = \
            meta_nodes[nodes_name]['num_con_domain']+meta_nodes[nodes_name]['num_cat_domain']+meta_nodes[nodes_name]['num_other_domain'] * self.embed_num
            if num_fields > 0:
                self.node_embed[nodes_name] = RowEmb_mod(
                    num_fields=num_fields,
                    embedding_size=embedding_size,
                    dropout=embed_dropout,
                    mode_list=embed_mode,
                    num_blocks=num_blocks,
                )
        
        # aggregation function
        self.agg_func = nn.ModuleDict()
        for nodes_name in meta_nodes.keys():
            nodes_agg_func = nn.ModuleDict()
            if meta_nodes[nodes_name]['neighbor_agg'] == None:
                continue
            for neighbor in meta_nodes[nodes_name]['neighbor_agg']:
                # neighbor -> nodes is one to many relation, just concat the nodes embedding
                if graph_config[nodes_name]['FK']!=None and neighbor in graph_config[nodes_name]['FK'].keys():
                    continue
                # neighbor -> nodes is many to one relation, need aggregation function
                else:
                    avg_d = meta_nodes[nodes_name]['avg_d'][neighbor]
                    nodes_agg_func[neighbor] = nn.ModuleList([Agg_func(embedding_size, agg_dropout, mode=agg_mode, avg_d=avg_d, aggregators=aggregators, scalers=scalers)
                                                               for i in range(self.embed_num)])
            self.agg_func[nodes_name] = nodes_agg_func
        
        # used for multi_update_all
        # we need to predefine it inorder to make sure the traversal order of etype_dict is the same when we apply 
        # multi_update_all, so that the stack order will be the same for each types of nodes everywhere.
        # we do this because we need to form nodes embedding for each type of nodes, and the nodes' column order
        # must be consistent everywhere.
        Aggregation = {
            'min': fn.min,
            'max': fn.max,
            'mean': fn.mean,
        }

        self.etype_agg_dict = {}
        for agg in aggregators:
            self.etype_agg_dict[agg] = OrderedDict()

        for src_node_type, edge_type, dst_node_type in canonical_etypes:
            for agg in aggregators:
                if agg == 'std':
                    self.etype_agg_dict[agg][edge_type] = (fn.copy_u('h2', 'mes'), fn.mean('mes', src_node_type+'_std'))
                else:
                    self.etype_agg_dict[agg][edge_type] = (fn.copy_u('h', 'mes'), Aggregation[agg]('mes', src_node_type+'_'+agg))
        
        
        num_fields = meta_nodes[tarn]['num_con_domain'] + meta_nodes[tarn]['num_cat_domain'] + len(meta_nodes[tarn]['neighbor_agg']) * self.embed_num
        if predictor == 'DeepFM':
            self.predictor = DeepFM(num_fields, embedding_size, p_hidden, p_dropout)
        elif predictor == 'fttransformer':
            self.cls = nn.Parameter(torch.randn(1, 1, embedding_size))
            self.predictor = Transformer(
                dim=embedding_size,
                depth=p_num_blocks,
                heads=num_heads,
                attn_dropout=p_dropout,
                ff_dropout=p_dropout,
                include_first_norm=False,
            )
            self.fc_out = nn.Sequential(
                nn.LayerNorm(embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, 1),
            )
            
    def forward(self, blocks:DGLBlock):
        h = {}
        # initializa row embedding 'h', we form row embedding for every nodes, and the information needed from other
        # types of node is set are zero vectors.
        for nodes_name in blocks[0].ndata['h'].keys():
            # deal with the nodes' type that haven't been traverse
            if blocks[0].num_src_nodes(nodes_name) == 0:
                continue

            input = []
            if 'Cat' in blocks[0].nodes[nodes_name].data:
                cat_feat = self.w(blocks[0].nodes[nodes_name].data['Cat'])
                # deal with situation that is only one categorical column
                if len(cat_feat.shape) == 2:
                    cat_feat = cat_feat.unsqueeze(1)
                input.append(cat_feat)
            if 'Con' in blocks[0].nodes[nodes_name].data:
                con_feat = self.con[nodes_name](blocks[0].nodes[nodes_name].data['Con'])
                # deal with situation that is only one continuous column
                if len(con_feat.shape) == 2:
                    con_feat = con_feat.unsqueeze(1)
                input.append(con_feat)

            # deal with special situation that nodes do not have its own features
            if len(input) == 0:
                value = blocks[0].nodes[nodes_name].data['h']
                h[nodes_name] = value
                continue

            input = torch.concat(input, dim=1)

            # information from other nodes are set to zero vectors
            if input.shape[1] < self.node_embed[nodes_name].num_fields:
                num_columns_zero = self.node_embed[nodes_name].num_fields - input.shape[1]
                input = torch.concat((input, torch.zeros(input.shape[0], num_columns_zero, input.shape[2]).to(input.device)), dim=1)
            
            nodes_embed = self.node_embed[nodes_name](input)
            h[nodes_name] = nodes_embed

        for i, block in enumerate(blocks):
            block.ndata['h'] = h
            for agg in self.aggregators:
                if agg == 'std':
                    h2 = {}
                    for nodes_name in block.ndata['h'].keys():
                        if nodes_name in h:
                            h2[nodes_name] = h[nodes_name] * h[nodes_name]
                        else:
                            h2[nodes_name] = torch.zeros(block.num_nodes(nodes_name), self.embed_num, self.embedding_size).to(h[self.tarn].device)
                    block.ndata['h2'] = h2
                    
                block.multi_update_all(
                    etype_dict=self.etype_agg_dict[agg],
                    cross_reducer=self.cross_reducer,
                )
            h = {}

            for nodes_name in block.dsttypes:
                # if there isn't some type of nodes in destination, this node's type
                # will still appear in block.dsttypes
                if block.num_dst_nodes(nodes_name) == 0 :
                    continue
                
                # aggregate the information from other tables
                mes = []
                for neighbor in self.meta_nodes[nodes_name]['neighbor_agg']:
                    # neighbor -> nodes is one to many relation, just concat the nodes embedding
                    if self.graph_config[nodes_name]['FK']!= None and neighbor in self.graph_config[nodes_name]['FK'].keys():
                        mes.append(block.dstnodes[nodes_name].data[neighbor+'_mean']) # assuming that at least we have mean aggregation
                    # neighbor -> nodes is many to one relation, need aggregation function
                    else:
                        agg_mes = [[] for _ in range(self.embed_num)]
                        for agg in self.aggregators:
                            tmp = block.dstnodes[nodes_name].data[neighbor+'_'+agg]

                            if agg == 'std':
                                tmp_mean = block.dstnodes[nodes_name].data[neighbor+'_mean']
                                tmp = torch.sqrt(torch.relu(tmp-tmp_mean * tmp_mean) + EPS)

                            for j in range(self.embed_num):
                                agg_mes[j].append(tmp[:, j])
                        for j in range(self.embed_num):
                            tmp = torch.concat(agg_mes[j], dim=1)
                            etype = list(block.metagraph().get_edge_data(neighbor, nodes_name).keys())[0]
                            d = block.in_degrees(torch.arange(block.num_dst_nodes(nodes_name)).to(block.device), etype=etype)
                            mes.append(self.agg_func[nodes_name][neighbor][j](tmp, d).unsqueeze(1))
                mes = torch.concat(mes, dim=1)

                input = []
                if 'Cat' in block.dstnodes[nodes_name].data:
                    cat_feat = self.w(block.dstnodes[nodes_name].data['Cat'])
                    # deal with situation that is only one categorical column
                    if len(cat_feat.shape) == 2:
                        cat_feat = cat_feat.unsqueeze(1)
                    input.append(cat_feat)
                if 'Con' in block.dstnodes[nodes_name].data:
                    con_feat = self.con[nodes_name](block.dstnodes[nodes_name].data['Con'])
                    # deal with situation that is only on continuous column
                    if len(con_feat.shape) == 2:
                        con_feat = con_feat.unsqueeze(1)
                    input.append(con_feat)

                # In the final block we do not need to compute the row embedding
                if i == len(blocks)-1:
                    input.append(mes)
                    input = torch.concat(input, dim=1)
                    output = input
                else:
                    if self.meta_nodes[nodes_name]['num_other_domain'] != 0:
                        input.append(mes)
                    # deal with special situation that nodes do not have its own features and do not need to form row embedding
                    if len(input) == 0:
                        output = block.dstnodes[nodes_name].data['h']
                    else:
                        input = torch.concat(input, dim=1)
                        output = self.node_embed[nodes_name](input)
                h[nodes_name] = output
        label = blocks[-1].dstnodes[self.tarn].data['label']
        pred_input = h[self.tarn]

        if self.predictor_name == 'DeepFM':
            pred_output = self.predictor(pred_input)
        elif self.predictor_name == 'fttransformer':
            pred_output = torch.cat([self.cls.expand(pred_input.shape[0], 1, -1), pred_input], 1)
            pred_output = self.predictor(pred_output)
            pred_output = pred_output[:, 0]
            pred_output = torch.sigmoid(self.fc_out(pred_output).view(-1))

        return pred_output, label

    def message_func(self, edges):
        return {'mes': edges.src['h']}
    
    def create_reduce_func(self, src_name, dst_name):
        # all the messages from different edge types must return the same feature keys so that the cross reduce funcion can reduce
        # (after reading dgl's *multi_update_all* api source code)
        def reduce_dunc(nodes):
            # many to one relation
            if src_name in self.agg_func[dst_name].keys():
                agg_func = self.agg_func[dst_name][src_name]
                all_edges_message = nodes.mailbox['mes']
                message = []
                for i in range(all_edges_message.shape[2]):
                    message.append(agg_func[i](all_edges_message[:, :, i, :]).unsqueeze(1))
                message = torch.concat(message, dim=1)
            # one to many relation
            else:
                message = nodes.mailbox['mes'].squeeze(1)
            
            # message: [batch_size, embedding_size]
            return {'mes': message}
        
        return reduce_dunc

    def cross_reducer(self, flist):
        return flist[0]