Source code of paper "[Deep Dual Support Vector Data Description for Anomaly Detection on Attributed Networks](https://doi.org/10.1002/int.22683)".
 

### Run Model Training and Evaluation

**Dual-SVDAE**:
```bash
python main.py --dataset cora --module SVDAE --nu1 0.2 --nu2 0.2 --beta 0.4 --lr 0.001 --n-hidden 32 --n-layers 2 --weight-decay 0.0005 --n-epochs 5000 
```

**OC-SVM(Raw)**:
```bash
python main.py --dataset cora --module OCSVM --mode A 
```


**OC-SVM (DW)**:
```bash
python main.py --dataset cora --module OCSVM --mode X 
```

**Deep-SVDD (Attr)**:
```bash
python main.py --dataset cora --module SVDD_Attr --nu 0.2 --lr 0.002 --n-hidden 32 --n-layers 2 --weight-decay 0.0005 --n-epochs 2000 
```

**Deep-SVDD (Stru)**:
```bash
python main.py --dataset cora --module SVDD_Stru --nu 0.2 --lr 0.002 --n-hidden 32 --n-layers 2 --weight-decay 0.0005 --n-epochs 2000 
```

**GAE**:
```bash
python main.py --dataset cora --module GAE -lr 0.002 --n-hidden 32 --n-layers 2  --n-epochs 2000 
```

**Dominant**:
```bash
python main.py --dataset cora --module Dominant -lr 0.002 --n-hidden 32 --n-layers 2 --n-epochs 2000 
```

**OC-GNN**:
```bash
python main.py --dataset cora --module --nu 0.2 --lr 0.002 --n-hidden 32 --n-layers 2 ---n-epochs 2000 
```

### Requirements:
pytorch>=1.4
DGL>=0.4.2
sklearn>=0.20.1
numpy>=1.16
networkx>=2.1
Pyod>=0.7.6
tensorflow>=1.4.0,<=1.12.0
gensim==3.6.0
DGL>=0.4.2



### Cite
If you make use of this code in your own work, please cite our paper.
```
@article{zhang2021deep,
  title={Deep dual support vector data description for anomaly detection on attributed networks},
  author={Zhang, Fengbin and Fan, Haoyi and Wang, Ruidong and Li, Zuoyong and Liang, Tiancai},
  journal={International Journal of Intelligent Systems},
  year={2021},
  doi={https://doi.org/10.1002/int.22683},
  publisher={Wiley Online Library}
}
```