from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve
import torch
from optim.loss import loss_function,loss_function_AE,loss_function_dualAE,anomaly_score
import numpy as np
import torch.nn as nn
import time
import matplotlib as plt

def fixed_graph_evaluate(args, path, model, data_center, data, radius, mask):
    model.eval()
    with torch.no_grad():
        labels = data['labels'][mask]
        loss_mask = mask.bool() & data['labels'].bool()

        # test_t0 = time.time()
        outputs = model(data['g'], data['features'])

        # print(loss_mask.)
        _, scores = anomaly_score(data_center, outputs, radius, mask)
        # test_dur = time.time()-test_t0
        loss, _, _ = loss_function(args.nu, data_center,  outputs, radius, loss_mask)
        # print("Test Time {:.4f}".format(test_dur))

        labels = labels.cpu().numpy()
        # dist=dist.cpu().numpy()
        scores = scores.cpu().numpy()
        pred = thresholding(scores, 0)

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        acc = accuracy_score(labels, pred)
        recall = recall_score(labels, pred)
        precision = precision_score(labels, pred)
        f1 = f1_score(labels, pred)

        return auc, ap, f1, acc, precision, recall, loss
def fixed_graph_evaluate_AE(args,path,model, data_center,data,radius,mask):

    model.eval()
    with torch.no_grad():
        labels = data['labels'][mask]
        loss_mask=mask.bool() & data['labels'].bool()

        outputs,rec = model(data['g'],data['features'])

        loss,_,scores=loss_function_AE(args.nu,rec,data['features'],data_center, outputs,radius,mask)

 
        labels=labels.cpu().numpy()

        scores=scores.cpu().numpy()
        pred=thresholding(scores,0)

        auc=roc_auc_score(labels, scores)
        ap=average_precision_score(labels, scores)

        acc=accuracy_score(labels,pred)
        recall=recall_score(labels,pred)
        precision=precision_score(labels,pred)
        f1=f1_score(labels,pred)

        return auc,ap,f1,acc,precision,recall,loss


def fixed_graph_evaluate_dualAE(args, path, model, data_center_A, data_center_S, data, adj, radius_A, radius_S,
                                mask):
    model.eval()
    with torch.no_grad():
        labels = data['labels'][mask]
        loss_mask = mask.bool() & data['labels'].bool()

        outputs_A, outputs_S, rec, re_adj = model(data['g'], data['features'])

        loss, _, _, scores = loss_function_dualAE(args, rec, re_adj, adj, data['features'],
                                                  data_center_A, data_center_S, outputs_A, outputs_S,
                                                  radius_A, radius_S, mask)
        # print("Test Time {:.4f}".format(test_dur))

        labels = labels.cpu().numpy()
        # dist=dist.cpu().numpy()
        scores = scores.cpu().numpy()
        pred = thresholding(scores, 0)

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        acc = accuracy_score(labels, pred)
        recall = recall_score(labels, pred)
        precision = precision_score(labels, pred)
        f1 = f1_score(labels, pred)

        return auc, ap, f1, acc, precision, recall, loss,outputs_A, outputs_S

def thresholding(recon_error,threshold):
    ano_pred=np.zeros(recon_error.shape[0])
    for i in range(recon_error.shape[0]):
        if recon_error[i]>threshold:
            ano_pred[i]=1
    return ano_pred

def baseline_evaluate(datadict,y_pred,y_score,val=True):
    
    if val==True:
        mask=datadict['val_mask']
    if val==False:
        mask=datadict['test_mask']

    auc=roc_auc_score(datadict['labels'][mask],y_score)
    ap=average_precision_score(datadict['labels'][mask],y_score)
    acc=accuracy_score(datadict['labels'][mask],y_pred)
    recall=recall_score(datadict['labels'][mask],y_pred)
    precision=precision_score(datadict['labels'][mask],y_pred)
    f1=f1_score(datadict['labels'][mask],y_pred)

    return auc,ap,f1,acc,precision,recall

def do_hist(scores, true_labels, directory, dataset, random_seed, display=False):
    plt.figure()
    idx_inliers = (true_labels == 0)
    idx_outliers = (true_labels == 1)
    hrange = (min(scores), max(scores))
    plt.hist(scores[idx_inliers], 50, facecolor=(0, 1, 0, 0.5),
             label="Normal samples", density=True, range=hrange)
    plt.hist(scores[idx_outliers], 50, facecolor=(1, 0, 0, 0.5),
             label="Anomalous samples", density=True, range=hrange)
    plt.title("Distribution of the anomaly score")
    plt.legend()
    if display:
       plt.show()
    else:
        plt.savefig(directory + 'histogram_{}_{}.png'.format(random_seed, dataset),
                    transparent=False, bbox_inches='tight')
        plt.savefig(directory + 'histogram_{}_{}.pdf'.format(random_seed, dataset),
                    transparent=False, bbox_inches='tight')
        plt.close()

