import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import dgl
import wandb
import pickle

import os
import sys
import random
sys.path.append('..')

from GFS.data.dataset_GFS import load_data
from sklearn.metrics import roc_auc_score
from GFS.algo.GFS import GFS
from data.config import graph_config
from torchmetrics import AUROC
from tqdm import tqdm

def seed_all(seed, gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluation(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        criterion = nn.BCELoss()
        predictions = []
        labels = []
        evaluation_loss = []
        for input_nodes, output_nodes, blocks in data_loader:
            blocks = [blk.to(device) for blk in blocks]
            pred, label = model(blocks)

            eval_loss = criterion(pred, label.float())
            pred = pred.squeeze().detach().cpu().numpy().astype('float64')
            label = label.detach().cpu().numpy()

            evaluation_loss.append(eval_loss.item())
            predictions.append(pred)
            labels.append(label)
        log_loss = np.mean(evaluation_loss)
        predictions = np.concatenate(predictions, 0)
        labels = np.concatenate(labels, 0)
        return roc_auc_score(y_score=predictions, y_true=labels), log_loss, predictions

def train():
    if args.use_wandb:
        if not args.repeat:
            run = wandb.init()
        wd = wandb.config.wd
        lr = wandb.config.lr
        row_emb_dropout = wandb.config.row_emb_dropout
        agg_dropout = wandb.config.agg_dropout
        p_dropout = wandb.config.p_dropout
        p_num_blocks = wandb.config.p_num_blocks
        num_blocks = wandb.config.num_blocks
    else:
        wd = args.wd
        lr = args.lr
        row_emb_dropout = args.row_emb_dropout
        agg_dropout = args.agg_dropout
        p_dropout = args.p_dropout
        p_num_blocks = args.p_num_blocks
        num_blocks = args.num_blocks

    train_loader, valid_loader, test_loader, meta_nodes, canonical_etypes = load_data(
        dname=args.dataset, 
        tarn=tarn, 
        batch_size=args.batch_size, 
        graph_config=graph_config[args.dataset][0],
        use_PNA=(args.agg_mode=='PNA'),
        num_workers=args.num_workers, 
        seed=args.seed, 
        depth=args.depth, 
        embedding_size=args.embedding_size,
        embed_len=len(args.row_emb_mode))

    model = GFS(
        tarn=tarn,
        meta_nodes=meta_nodes,
        graph_config=graph_config[args.dataset][0],
        canonical_etypes=canonical_etypes,
        num_feat=num_feat[args.dataset],
        embedding_size=args.embedding_size,
        embed_mode=args.row_emb_mode,
        embed_dropout=row_emb_dropout,
        agg_mode=args.agg_mode,
        agg_dropout=agg_dropout,
        aggregators=args.aggregators,
        scalers=args.scalers,
        p_hidden=args.p_hidden,
        p_dropout=p_dropout,
        predictor=args.predictor,
        num_blocks=num_blocks,
        p_num_blocks=p_num_blocks,
    ).to(device)

    criterion = nn.BCELoss()
    metric = AUROC(task='binary')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # start training
    print('Start training')
    best_auc = 0.0
    kill_cnt = 0
    for epoch in range(args.epochs):
        train_loss = []
        metric.reset()
        model.train()
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            t.set_description(f'Epoch: {epoch+1}/{args.epochs}')
            for step, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
                blocks = [blk.to(device) for blk in blocks]
                logits, label = model(blocks)

                # compute loss
                tr_loss = criterion(logits, label.float())
                tr_auc = metric(logits.detach(), label.long().detach())
                train_loss.append(tr_loss.item())

                # backward
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()

                t.update()
                t.set_postfix({
                    'train loss': f'{tr_loss.item():.4f}',
                    'train_auc': f'{tr_auc.item():.4f}'
                })
        train_loss = np.mean(train_loss)
        train_auc = metric.compute().item()
        metric.reset()

        validate_auc, validate_log_loss, _ = evaluation(model, valid_loader, device)

        print("In epoch {}, Train Loss: {:.5}, Train AUC: {:.5}, Valid AUC: {:.5}, Valid Log Loss: {:.5}\n".format(epoch+1, train_loss, train_auc, validate_auc, validate_log_loss))
        log_file.write("In epoch {}, Train Loss: {:.5}, Train AUC: {:.5}, Valid AUC: {:.5}, Valid Log Loss {:.5}\n".format(epoch+1, train_loss, train_auc, validate_auc, validate_log_loss))
        if args.use_wandb:
            wandb.log({
                'epoch': epoch+1,
                'Train Loss': train_loss,
                'Train AUC': train_auc,
                'Valid AUC': validate_auc,
                'Valid Log Loss': validate_log_loss,
            })

        #validate
        if validate_auc > best_auc:
            best_auc = validate_auc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.out_path, 'model', 'GFS', 'GFS'+'_'+args.num+'_seed:'+str(args.seed)))
            kill_cnt = 0
            print("saving model...")
            log_file.write("saving model...\n")
        else:
            kill_cnt += 1
            if kill_cnt >= args.early_stop:
                print('early stop.')
                log_file.write('early stop.\n')
                print("best epoch: {}, Valid AUC: {:.5}\n".format(best_epoch+1, best_auc))
                log_file.write("best epoch: {}, Valid AUC: {:.5}\n".format(best_epoch+1, best_auc))
                if args.use_wandb:
                    wandb.log({
                        'best_validate_auc': best_auc,
                    })
                break
            
    # test use the best model
    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.out_path, 'model', 'GFS', 'GFS'+'_'+args.num+'_seed:'+str(args.seed))))
    test_auc, test_log_loss, test_predictions = evaluation(model, test_loader, device)
    print("Test AUC: {:.5}\n".format(test_auc))
    log_file.write("Test AUC: {:.5}\n".format(test_auc))
    if args.use_wandb:
        wandb.log({
            'Test AUC': test_auc,
            'Test Log Loss': test_log_loss,
        })
    return test_auc, test_log_loss, test_predictions

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='acquire-valued-shoppers')
    parser.add_argument("--data_path", default="../data", help="Path to save the data")
    parser.add_argument("--out_path", default="../output", help="Path to save the output")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of processes to construct batches")
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--use_wandb', action='store_true')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument("--num", default='0', type=str, help="Model log name.")
    parser.add_argument("--gpu", type=int, default="0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    parser.add_argument("--wd", type=float, default=0.0, help="L2 Regularization for Optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='Exponential decay of learning rate')

    # GFS hyperparameter
    parser.add_argument('--embedding_size', default=16, type=int, help='Initial dimension size for entities.')
    parser.add_argument('--row_emb_mode', default=['FM','Con_MLP'], nargs='+', help='row embedding mode for GFS, you can choose\
                        from DeepFM, Sum_MLP, Con_MLP, FM, fttransformer and any combination of them')
    parser.add_argument('--agg_mode', default='PNA', choices=['PNA', 'Sum_MLP'], help='aggregation mode for GFS')
    parser.add_argument('--aggregators', default=['mean', 'max', 'min', 'std'], nargs='+', help="PNA aggregator, you can choose from ['mean', 'max', 'min', 'std'], 'std' must behind 'mean'" )
    parser.add_argument('--scalers', default=['identity', 'amplification', 'attenuation'], nargs='+', help="PNA scalers, you can choose from ['identity', 'amplification', 'attenuation']")
    parser.add_argument('--row_emb_dropout', default=0.0, type=float, help='row embedding dropout for GFS.')
    parser.add_argument('--agg_dropout', default=0.0, type=float, help='aggregation function dropout for GFS.')
    parser.add_argument('--depth', default=3, type=int, help='Depth for GFS')
    parser.add_argument('--num_blocks', default=3, type=int)

    # predictor hyperparameter
    parser.add_argument('--predictor', default='DeepFM', choices=['DeepFM', 'fttransformer'])
    parser.add_argument('--p_hidden', default=500, nargs='*', type=int, help='Hidden dimension size for prediction model.')
    parser.add_argument('--p_dropout', default=0.0, type=float, help='Dropout for prediction model.')
    parser.add_argument('--early_stop', default=3, type=int, help='Patience for early stop.')
    parser.add_argument('--p_num_blocks', default=3, type=int)

    # repeat experiment
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--repeat_num', type=int, default=5)
    parser.add_argument('--repeat_target', type=str)
    parser.add_argument('--id', type=str)
    parser.add_argument('--ablation',action='store_true')
    parser.add_argument('--ablation_dis', type=str)

    # parallelize sweep
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--sweep_id', type=str)

    # hyperparameter study
    parser.add_argument('--hp_study', action='store_true')
    parser.add_argument('--target', type=str)
    parser.add_argument('--hp_id', type=str)

    args = parser.parse_args()
    print(args)

    num_feat = {
        'acquire-valued-shoppers': 37561+1,
        'outbrain-full': 1367458+1,
        'diginetica': 14877+1,
        'home-credit': 381+1,
        'kdd15': 119479+1,
        'synthetic': 1+1,
        'synthetic2': 1+1,
    }
    target_node = {
        'acquire-valued-shoppers': 'History',
        'outbrain-full': 'clicks',
        'diginetica': 'clicks',
        'home-credit': 'applications',
        'kdd15': 'enrollment',
        'synthetic': 'a',
        'synthetic2': 'a',
    }
    tarn = target_node[args.dataset]

    # prepare the dataset and device
    args.out_path = os.path.join(args.out_path, args.dataset)
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    print(f"Device is {device}.")


    seed_all(args.seed, device)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'train_log', 'GFS'), exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'model', 'GFS'), exist_ok=True)
    log_name = 'GFS'+'_'+args.num+'_seed:'+str(args.seed) + ".txt"
    log_file = open(os.path.join(args.out_path, 'train_log', 'GFS', log_name), "w+")
    log_file.write(str(args))

    if args.use_wandb:
        if args.repeat:
            api = wandb.Api()
            run = api.run(args.repeat_target)
            config = run.config

            # just due to some changing hyperparameter problems
            config['p_num_blocks'] = 3
            config['num_blocks'] = 3
            
            if args.ablation:
                final_out_path = os.path.join(args.out_path, 'GFS'+args.predictor, args.ablation_dis, 'final'+args.id)
            else:
                final_out_path = os.path.join(args.out_path, 'GFS'+args.predictor, 'final'+args.id)
            os.makedirs(final_out_path, exist_ok=True)
            all_test_auc = []
            all_test_log_loss = []

            for i in range(args.repeat_num):
                if args.ablation:
                    run = wandb.init(project=args.dataset+'_GFS_'+args.predictor, config=config, group=args.ablation_dis+'_final'+args.id, job_type="eval")
                else:
                    run = wandb.init(project=args.dataset+'_GFS_'+args.predictor, config=config, group='final'+args.id, job_type="eval")
                test_auc, test_log_loss, test_predictions = train()
                all_test_auc.append(test_auc)
                all_test_log_loss.append(test_log_loss)
                wandb.finish()
                with open(os.path.join(final_out_path, 'test'+str(i)+'.pkl'), 'wb') as f:
                    pickle.dump(test_predictions, f)

            mean_auc = np.mean(all_test_auc)
            std_auc = np.std(all_test_auc)
            mean_log_loss = np.mean(all_test_log_loss)
            std_log_loss = np.std(all_test_log_loss)
            with open(os.path.join(final_out_path, 'results.txt'), 'w') as f:
                f.write('Mean Test AUC: {:.5f}\n'.format(mean_auc))
                f.write('Std Test AUC: {:.5f}\n'.format(std_auc))
                f.write('Mean Log Loss: {:.5f}\n'.format(mean_log_loss))
                f.write('Std Test AUC: {:.5f}\n'.format(std_log_loss))
        elif args.parallelize:
            wandb.agent(sweep_id=args.sweep_id, function=train)
        elif args.hp_study:
            api = wandb.Api()
            run = api.run(args.target)

            config = run.config
            # just due to some changing hyperparameter problems
            config['p_num_blocks'] = 3
            config['num_blocks'] = 3
            
            lr_range = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
            for i,lr in enumerate(lr_range):
                config['lr'] = lr
                run = wandb.init(project=args.dataset+'_GFS_'+args.predictor, config=config, group='hp_study'+args.hp_id, job_type="eval")
                train()
                wandb.finish()

        else:
            sweep_configuration = {
                'project': args.dataset+'_GFS_depth'+str(args.depth),
                'method': 'bayes',
                'name': 'DeepFM no std',
                'metric': {
                    'goal': 'maximize',
                    'name': 'best_validate_auc',
                },
                'parameters':{
                    'wd':{
                        'distribution': 'log_uniform_values',
                        'max': 1e-3,
                        'min': 1e-7,
                    },
                    'lr':{
                        'distribution': 'log_uniform_values',
                        'max': 1e-3,
                        'min': 1e-5,
                    },
                    'agg_dropout':{
                        'distribution': 'uniform',
                        'max': 0.2,
                        'min': 0,
                    },
                    'row_emb_dropout':{
                        'distribution': 'uniform',
                        'max': 0.2,
                        'min': 0,
                    },
                    'p_dropout': {
                        'distribution': 'uniform',
                        'max': 0.2,
                        'min': 0,
                    },
                    'p_num_blocks':{
                        'values': [3],
                    },
                    'num_blocks': {
                        'values': [3],
                    }
                }
            }
            sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.dataset+'_GFS_depth'+str(args.depth))
            wandb.agent(sweep_id, function=train)
    else:
        train()