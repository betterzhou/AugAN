# Improving Generalizability of Graph Anomaly Detection Models via Data Augmentation (AugAN)

## 1.Introduction
This repository contains code for paper "[Improving Generalizability of Graph Anomaly Detection Models via Data Augmentation](https://ieeexplore.ieee.org/abstract/document/10119211)" (TKDE 2023).

## 2. Usage
### Requirements:
+ pytorch==1.7.0
+ dgl==0.7.2
+ scikit-learn
+ scipy
+ pandas
+ networkx

### Datasets:
Users can create datasets with the code.
+ python create_datasets.py

Please check the data statistics and control the overlapped nodes in each sub-graph.
The processed datasets are put into the ./sub_G_datasets/ folder.

### Data Format:
The input data for AugAN is a '.mat' file with 'gnd' (ground-truth), 'Attributes' (attributes), and 'Network' (graph structure).

### Example:
+ cd ./src/
+ python main.py --epoch=2001 --dataset=AD_ms_academic_cs_sub --meta_lr=1e-05 --update_lr=1e-05 --known_outliers_num=20 --batch_size=128 --alpha=0.1 --remain_prop=0.5 --seed=1
+ python main.py --epoch=2001 --dataset=AD_dblp_sub --meta_lr=1e-05 --update_lr=1e-05 --known_outliers_num=20 --batch_size=128 --alpha=0.3 --remain_prop=0.8 --seed=1

## 3. Citation
Please kindly cite the paper if you use the code or any resources in this repo:
```bib
@article{zhou2023improving,
  title={Improving generalizability of graph anomaly detection models via data augmentation},
  author={Zhou, Shuang and Huang, Xiao and Liu, Ninghao and Zhou, Huachi and Chung, Fu-Lai and Huang, Long-Kai},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}
```
