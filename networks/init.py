import torch.nn.functional as F
from networks.SVDD import SVDD_Attr
from networks.GCN import GCN
from networks.SVDAE import SVDAE
from networks.GAE import GAE
from networks.GraphSAGE import GraphSAGE
from networks.Dominant import Dominant
def init_model(args,input_dim,input_sdim):
    # create GCN model
    if args.module== 'GCN':
        model = GCN(None,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout)
    if args.module== 'SVDAE':
        model = SVDAE(None,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout)
    if args.module =='SVDD_Attr':
        model = SVDD_Attr(None,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout)
    if args.module =='SVDD_stru':
        model = GCN(None,
                input_sdim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout)
    if args.module== 'GraphSAGE':
        model = GraphSAGE(None,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout,
                aggregator_type='pool')

    if args.module== 'GAE':
        model = GAE(None,
                input_dim,
                n_hidden=args.n_hidden*2,
                n_classes=args.n_hidden,
                n_layers=args.n_layers,
                activation=F.relu,
                dropout=args.dropout)
    if args.module== 'Dominant':
        model = Dominant(None,
                input_dim,
                n_hidden=args.n_hidden*2,
                n_classes=args.n_hidden,
                n_layers=args.n_layers,
                activation=F.relu,
                dropout=args.dropout)


    if args.gpu < 0:
        cuda = False
    else:
        cuda = True

    if cuda:
        model.cuda()

    print(f'Parameter number of {args.module} Net is: {count_parameters(model)}')

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)