import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import  average_precision_score, roc_auc_score

from optim.loss import EarlyStopping
import os

def train(args, logger, data, model, path):
    checkpoints_path = path

    logger.info('Start training')
    logger.info(
        f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(
        f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)

    adj = data['g'].adjacency_matrix().to_dense().cuda()
    loss_fn = nn.MSELoss()
    model.train()

    # 创立矩阵以存储结果曲线
    arr_epoch = np.arange(args.n_epochs)
    arr_loss = np.zeros(args.n_epochs)
    arr_valauc = np.zeros(args.n_epochs)
    arr_testauc = np.zeros(args.n_epochs)
    savedir = './embeddings/Dominant/' + args.dataset + '/{}'.format(args.abclass)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    max_valauc = 0
    epoch_max = 0
    dur = []
    for epoch in range(args.n_epochs):
        # model.train()
        # if epoch %5 == 0:
        t0 = time.time()
        # forward
        z, re_x, re_adj = model(data['g'], data['features'])
        if args.module == 'Dominant':
            loss = Recon_loss(re_x, re_adj, adj, data['features'], data['train_mask'], loss_fn, 'AX', 0.2)
        else:
            loss = Recon_loss(re_x, re_adj, adj, data['features'], data['train_mask'], loss_fn, 'A', 0)

        # 保存训练loss
        arr_loss[epoch] = loss.item()
        #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur = time.time() - t0

        auc, ap, val_loss = fixed_graph_evaluate(args, model, data, adj, data['val_mask'],0.2)
        # 保存验证集AUC
        arr_valauc[epoch] = auc
        # 保存验证集AUC

        if auc > max_valauc:
            max_valauc = auc
            epoch_max = epoch
            torch.save(model.state_dict(), checkpoints_path)
            np.save(os.path.sep.join(
                [savedir + '/embeddings_max.npy']),
                z[data['test_mask']].data.cpu().numpy())
            np.save(os.path.sep.join([savedir + '/label_max.npy'.format(args.n_epochs)]),
                    data['labels'][data['test_mask']].cpu().numpy())

        print(
            "Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f}  | Val AUROC {:.4f}  | "
            "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item() * 100000,
                                          auc,  data['n_edges'] / np.mean(dur) / 1000))

    if args.early_stop:
        print('loading model before testing.')
        model.load_state_dict(torch.load(checkpoints_path))

        # if epoch%100 == 0:

    auc, ap, _ = fixed_graph_evaluate(args, model, data, adj, data['test_mask'],0.2)
    test_dur = 0
    # 保存测试集AUC
    arr_testauc[epoch] = auc
    # 保存测试集AUC
    best_epoch = epoch_max
    print("Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(test_dur, auc, ap))


    return auc, ap, best_epoch


def Recon_loss(re_x, re_adj, adj, x, mask, loss_fn, mode,alpha):
    # S_loss: structure loss A_loss: Attribute loss
    if mode == 'A':
        return loss_fn(re_x[mask], x[mask])
    if mode == 'X':
        return loss_fn(re_adj[mask], adj[mask])
    if mode == 'AX':
        return (1-alpha)*loss_fn(re_x[mask], x[mask]) + alpha*loss_fn(re_adj[mask], adj[mask])


def anomaly_score(re_x, re_adj, adj, x, mask, loss_fn, mode,alpha):
    if mode == 'A':
        A_scores = F.mse_loss(re_x[mask], x[mask], reduction='none')
        return torch.mean(A_scores, 1)
    if mode == 'X':
        S_scores = F.mse_loss(re_adj[mask], adj[mask], reduction='none')
        return torch.mean(S_scores, 1)

    if mode == 'AX':
        A_scores = F.mse_loss(re_x[mask], x[mask], reduction='none')
        S_scores = F.mse_loss(re_adj[mask], adj[mask], reduction='none')
        return (1-alpha)*torch.mean(A_scores, 1) + alpha*torch.mean(S_scores, 1)


def fixed_graph_evaluate(args, model, data, adj, mask,alpha):
    loss_fn = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        labels = data['labels'][mask]

        loss_mask = mask.bool() & data['labels'].bool()
        z, re_x, re_adj = model(data['g'], data['features'])
        if args.module == 'GAE':
            loss = Recon_loss(re_x, re_adj, adj, data['features'], loss_mask, loss_fn, 'X', alpha)
            scores = anomaly_score(re_x, re_adj, adj, data['features'], mask, loss_fn, 'X', alpha)
        else:
            loss = Recon_loss(re_x, re_adj, adj, data['features'], loss_mask, loss_fn, 'AX',alpha)

            scores = anomaly_score(re_x, re_adj, adj, data['features'], mask, loss_fn, 'AX',alpha)


        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

    return auc, ap, loss