import time
import numpy as np
import torch
import logging
# from dgl.contrib.sampling.sampler import NeighborSampler
# import torch.nn as nn
# import torch.nn.functional as F


from optim.loss import loss_function, init_center, get_radius, EarlyStopping

from utils.evaluate import fixed_graph_evaluate
import os

def train(args, logger, data, model, path):
    checkpoints_path = path
    logger.info('Start training')
    logger.info(
        f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(
        f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    # initialize data center

    input_feat = data['features']
    input_g = data['g']

    data_center = init_center(args, input_g, input_feat, model)
    if args.gpu < 0:
        radius = torch.tensor(0)
    else:
        radius = torch.tensor(0, device=f'cuda:{args.gpu}')  # radius R initialized with 0 by default.
    # 创立矩阵以存储结果曲线
    arr_epoch = np.arange(args.n_epochs)
    arr_loss = np.zeros(args.n_epochs)
    arr_valauc = np.zeros(args.n_epochs)
    arr_testauc = np.zeros(args.n_epochs)
    max_valauc = 0
    epoch_max = 0
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        # model.train()
        # if epoch %5 == 0:
        t0 = time.time()
        # forward
        savedir = './embeddings/OCGCN/' + args.dataset + '/{}'.format(args.abclass)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        outputs = model(input_g, input_feat)
        # print('model:',args.module)
        # print('output size:',outputs.size())

        loss, dist, _ = loss_function(args.nu, data_center, outputs, radius, data['train_mask'])
        # 保存训练loss
        arr_loss[epoch] = loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if args.gpu < 0:
            radius.data = torch.tensor(get_radius(dist, args.nu))
        else:
            radius.data = torch.tensor(get_radius(dist, args.nu), device=f'cuda:{args.gpu}')


        auc, ap, f1, acc, precision, recall, val_loss = fixed_graph_evaluate(args, checkpoints_path, model, data_center,
                                                                             data, radius, data['val_mask'])
        # 保存验证集AUC
        arr_valauc[epoch] = auc
        if auc > max_valauc:
            max_valauc = auc
            epoch_max = epoch
            torch.save(model.state_dict(), checkpoints_path)
            auc_test, ap_test, f1_test, acc_test, precision_test, recall_test, test_loss = fixed_graph_evaluate(args, checkpoints_path, model, data_center,
                                                                             data, radius, data['test_mask'])
            # torch.save(model.state_dict(), path)
            np.save(os.path.sep.join(
                [savedir+'/embeddings_max.npy']),
                outputs[data['test_mask']].data.cpu().numpy())
            np.save(os.path.sep.join([savedir + '/label_max.npy'.format(args.n_epochs)]),
                    data['labels'][data['test_mask']].cpu().numpy())
        print(
            "Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f}  | Val AUROC {:.4f} | Test AUROC {:.4f} | Test AP {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item() * 100000,
                                          auc, auc_test, ap_test, data['n_edges'] / np.mean(dur) / 1000))

        if args.early_stop:
            if stopper.step(auc, val_loss.item(), model, epoch, checkpoints_path):
                break

        if args.early_stop:
            if stopper.step(auc, val_loss.item(), model, epoch, checkpoints_path):
                break

    if args.early_stop:
        print('loading model before testing.')
        model.load_state_dict(torch.load(checkpoints_path))
    model.load_state_dict(torch.load(checkpoints_path))
    auc, ap, f1, acc, precision, recall, loss = fixed_graph_evaluate(args, checkpoints_path, model, data_center, data,
                                                                     radius, data['test_mask'])
    test_dur = 0
    # 保存测试集AUC
    arr_testauc[epoch] = auc
    # 保存测试集AUC
    print("Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(test_dur, auc, ap))
    # print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    # logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
    # logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    # logger.info('\n')
    test_auc = auc
    test_ap = ap
    best_epoch = epoch_max
    # np.savez('SAGE-2.npz',epoch=arr_epoch,loss=arr_loss,valauc=arr_valauc,testauc=arr_testauc)

    return test_auc,test_ap,best_epoch


