import torch    
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
def data_norm(A):
    max_A = torch.max(A).expand_as(A)
    min_A = torch.min(A).expand_as(A)  # nn.MSELoss(rec[mask], data[mask])

    A = (A - min_A) / (max_A - min_A)
    return A
def loss_function(nu,data_center,outputs,radius=0,mask=None):
    dist,scores=anomaly_score(data_center,outputs,radius,mask)
    loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss,dist,scores

def loss_function_AE(nu,rec,data,data_center,outputs,radius=0,mask=None):
    dist,scores=anomaly_score(data_center,outputs,radius,mask)
    loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))+torch.mean((rec[mask]-data[mask])**2)
    return loss,dist,scores

##loss of Dual-SVDAE
def loss_function_dualAE(args,rec,re_adj,adj,data,data_center_A, data_center_S, outputs_A,outputs_S,radius_A=0,radius_S = 0,mask=None):
    dist_A, scores_A = anomaly_score(data_center_A, outputs_A, radius_A, mask)
    dist_S, scores_S = anomaly_score(data_center_S, outputs_S, radius_S, mask)
    scores_A = F.normalize(scores_A, p=1, dim=0)
    scores_S = F.normalize(scores_S, p=1, dim=0)
    scores = args.beta * scores_A + (1 - args.beta) * scores_S
    loss_A = radius_A ** 2 + (1 / args.nu1) * torch.mean(torch.max(torch.zeros_like(scores_A), scores_A)) + torch.mean((rec[mask] - data[mask]) ** 2)  #
    loss_S = radius_S ** 2 + (1 / args.nu2) * torch.mean(torch.max(torch.zeros_like(scores_S), scores_S)) + torch.mean((re_adj[mask] - adj[mask]) ** 2)  #
    loss = args.beta * loss_A + (1 - args.beta) * loss_S
    return loss, dist_A, dist_S, scores

def anomaly_score(data_center,outputs,radius=0,mask= None):
    if mask == None:
        dist = torch.sum((outputs - data_center) ** 2, dim=1)
    else:
        dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)
    scores = dist - radius ** 2
    return dist,scores
def init_center(args,input_g,input_feat, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    if args.gpu<0 :
        c = torch.zeros(args.n_hidden)
    else:
        c = torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')

    model.eval()
    with torch.no_grad():
        if args.module in [ 'GCN', 'GraphSAGE', 'SVDD_Attr','SVDD_Stru']:
            outputs = model(input_g,input_feat)
        else:
            outputs,rec = model(input_g, input_feat)
        # get the inputs of the batch

        n_samples = outputs.shape[0]
        c =torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def init_center_dual(args,input_g,input_feat, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    if args.gpu<0 :
        c_A = torch.zeros(args.n_hidden)
        c_S = torch.zeros(args.n_hidden)
    else:
        c_A = torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')
        c_S = torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')
    model.eval()
    with torch.no_grad():
        outputs_A, outputs_S,rec,re_adj = model(input_g, input_feat)
        # get the inputs of the batch

        n_samples_A = outputs_A.shape[0]
        c_A =torch.sum(outputs_A, dim=0)
        n_samples_S = outputs_S.shape[0]
        c_S =torch.sum(outputs_S, dim=0)
    c_A /= n_samples_A
    c_S /= n_samples_S

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c_A[(abs(c_A) < eps) & (c_A < 0)] = -eps
    c_A[(abs(c_A) < eps) & (c_A > 0)] = eps

    c_S[(abs(c_S) < eps) & (c_S < 0)] = -eps
    c_S[(abs(c_S) < eps) & (c_S > 0)] = eps
    return c_A,c_S
def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    radius=np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    # if radius<0.1:
    #     radius=0.1
    return radius

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.lowest_loss = None
        self.early_stop = False

    def step(self, acc,loss, model,epoch,path):
        score = acc
        cur_loss=loss
        if (self.best_score is None) or (self.lowest_loss is None):
        #if self.lowest_loss is None:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.save_checkpoint(acc,loss,model,path)
        #elif cur_loss > self.lowest_loss:
        elif (score < self.best_score) and (cur_loss > self.lowest_loss):
            self.counter += 1
            if self.counter >= 0.8*(self.patience):
                print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.save_checkpoint(acc,loss,model,path)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, acc,loss,model,path):
        '''Saves model when validation loss decrease.'''
        print('model saved. loss={:.4f} AUC={:.4f}'. format(loss,acc))
        torch.save(model.state_dict(), path)
