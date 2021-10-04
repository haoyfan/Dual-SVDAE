import time
import numpy as np
import torch
import os
from optim.loss import loss_function_dualAE,init_center_dual,get_radius,EarlyStopping
from utils.evaluate import fixed_graph_evaluate_dualAE


def train(args, logger, data, model, path):
    checkpoints_path = path
    logger.info('Start training')
    logger.info(
        f'dropout:{args.dropout}, nu1:{args.nu1},nu2:{args.nu2},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

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
    if args.gpu < 0:
        adj = data['g'].adjacency_matrix().to_dense().cpu()
    else:
        adj = data['g'].adjacency_matrix().to_dense().cuda()
    data_center_A, data_center_S = init_center_dual(args, input_g, input_feat, model)
    if args.gpu < 0:
        radius_A = torch.tensor(0)
        radius_S = torch.tensor(0)
    else:
        radius_A = torch.tensor(0, device=f'cuda:{args.gpu}')  # radius R initialized with 0 by default.
        radius_S = torch.tensor(0, device=f'cuda:{args.gpu}')
    savedir = './embeddings/SVDAE/' + args.dataset + '/{}'.format(args.abclass)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
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
        t0 = time.time()

        outputs_A, outputs_S, rec, re_adj = model(input_g, input_feat)
        loss, dist_A, dist_S, score = loss_function_dualAE(args, rec, re_adj, adj, data['features'],
                                                       data_center_A, data_center_S, outputs_A, outputs_S, radius_A,
                                                       radius_S, data['train_mask'])
        # 保存训练loss
        arr_loss[epoch] = loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        # radius.data=torch.tensor(get_radius(dist, args.nu), device=)
        if args.gpu < 0:
            radius_A.data = torch.tensor(get_radius(dist_A, args.nu))
            radius_S.data = torch.tensor(get_radius(dist_S, args.nu))
        else:
            radius_A.data = torch.tensor(get_radius(dist_A, args.nu), device=f'cuda:{args.gpu}')
            radius_S.data = torch.tensor(get_radius(dist_S, args.nu), device=f'cuda:{args.gpu}')
        auc, ap, f1, acc, precision, recall, val_loss,_,_ = fixed_graph_evaluate_dualAE( \
            args, checkpoints_path, model, data_center_A, data_center_S, data, adj, radius_A, radius_S,
            data['val_mask'])
        np.save(os.path.sep.join([savedir+'/embeddingsS_{}.npy'.format(args.n_epochs)]), outputs_S.data.cpu().numpy())
        np.save(os.path.sep.join([savedir + '/label_{}.npy'.format(args.n_epochs)]), data['labels'].cpu().numpy())
        # 保存验证集AUC
        arr_valauc[epoch] = auc
        if auc > max_valauc:
            max_valauc = auc
            epoch_max = epoch
            torch.save(model.state_dict(), checkpoints_path)


            auc_test, ap_test, f1_test, acc_test, precision_test, recall_test, test_loss,oA, oS = fixed_graph_evaluate_dualAE( \
                args, checkpoints_path, model, data_center_A, data_center_S, data, adj, radius_A, radius_S,
                data['test_mask'])
            np.save(os.path.sep.join(
                [savedir+  '/embeddingsS_max.npy']),
                oS[ data['test_mask']].data.cpu().numpy())
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
        print('loading model before testing.')
        model.load_state_dict(torch.load(checkpoints_path))

    model.load_state_dict(torch.load(checkpoints_path))
    auc, ap, f1, acc, precision, recall, loss,A,S = fixed_graph_evaluate_dualAE(args, checkpoints_path, model,
                                                                            data_center_A, data_center_S, data, adj,
                                                                            radius_A, radius_S, data['test_mask'])

    np.save(os.path.sep.join(
        [savedir + '/embeddingsA_last.npy']),
        A[data['test_mask']].data.cpu().numpy())
    np.save(os.path.sep.join(
        [savedir + '/embeddingsS_last.npy']),
        S[data['test_mask']].data.cpu().numpy())
    np.save(os.path.sep.join([savedir + '/label.npy'.format(args.n_epochs)]),
            data['labels'][data['test_mask']].cpu().numpy())
    test_dur = 0
    # 保存测试集AUC
    arr_testauc[epoch_max] = auc
    # 保存测试集AUC
    print("Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(test_dur, auc, ap))

    test_auc = auc
    test_ap = ap
    best_epoch = epoch_max

    return test_auc, test_ap, best_epoch


